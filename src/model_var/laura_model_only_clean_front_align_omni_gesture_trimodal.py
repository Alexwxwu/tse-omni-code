import logging
from typing import Any, List, Tuple, Dict, Optional, Union
import torch
import torch.nn as nn
from funcodec.modules.embedding import PositionalEncoding, ScaledPositionalEncoding
from funcodec.modules.nets_utils import (
    subsequent_mask, make_pad_mask, th_accuracy, pad_list
)
from funcodec.train.abs_espnet_model import AbsESPnetModel
import torch.nn.functional as F
from funcodec.torch_utils.device_funcs import force_gatherable
from funcodec.losses.label_smoothing_loss import LabelSmoothingLoss
from copy import deepcopy

from funcodec.models.audio_generation.laura_model import QuantizerCodebook
from funcodec.bin.codec_inference import Speech2Token
from transformers import RobertaTokenizer, RobertaModel

Mel_len = 63
gesture_FPS = 15
lip_FPS = 25 
gesture_aux_upsample_factor = Mel_len/gesture_FPS
lip_aux_upsample_factor = Mel_len/lip_FPS
gesture_codec_upsample_factor = lip_FPS/gesture_FPS
 
class GatedCrossAttPrompt(nn.Module):
    def __init__(self, text_dim, mix_dim, prompt_dim, n_heads=4):
        super().__init__()
        # 统一到同一注意力维度
        self.text_proj = nn.Linear(text_dim, prompt_dim)
        self.mix_proj = nn.Linear(mix_dim, prompt_dim)
        self.attn = nn.MultiheadAttention(prompt_dim, n_heads, batch_first=True)

        # 门控，将原始 text 和 attended context 融合
        self.gate = nn.Linear(prompt_dim * 2, prompt_dim)
        self.out_ln = nn.LayerNorm(prompt_dim)

    def forward(self, text_emb, mix_emb):
        """
        text_emb: (B, D_text)  # pooled text
        mix_emb:  (B, T, D_mix)  # sequence of mixture frames (codec emb)
        return:   (B, 1, D_prompt)  # 1 个 mix-conditioned text prompt token
        """
        B, T, _ = mix_emb.shape

        # 投影到同一维度
        q = self.text_proj(text_emb).unsqueeze(1)   # (B, 1, Dp)
        kv = self.mix_proj(mix_emb)                 # (B, T, Dp)

        # cross-attention: text query 看整段 mix
        attn_out, _ = self.attn(query=q, key=kv, value=kv)  # (B, 1, Dp)

        # 门控融合：原始 text vs attended context
        both = torch.cat([q, attn_out], dim=-1)  # (B, 1, 2*Dp)
        g = torch.sigmoid(self.gate(both))       # (B, 1, Dp)
        fused = g * attn_out + (1 - g) * q       # (B, 1, Dp)
        fused = self.out_ln(fused)               # (B, 1, Dp)

        return fused  # 作为 1 个 prompt token

     
class VisualConv1D(nn.Module):
    def __init__(self):
        super(VisualConv1D, self).__init__()
        relu = nn.ReLU()
        norm_1 = nn.BatchNorm1d(512)
        dsconv = nn.Conv1d(512,
                           512,
                           3,
                           stride=1,
                           padding=1,
                           dilation=1,
                           groups=512,
                           bias=False)
        prelu = nn.PReLU()
        norm_2 = nn.BatchNorm1d(512)
        pw_conv = nn.Conv1d(512, 512, 1, bias=False)

        self.net = nn.Sequential(relu, norm_1, dsconv, prelu, norm_2, pw_conv)

    def forward(self, x):
        out = self.net(x)
        return out + x
    
    
class LauraTSE(AbsESPnetModel):
    """
    LauraTSE model from LauraGPT Backbone[1]. 

    [1] LauraGPT: Listen, Attend, Understand, and Regenerate Audio with GPT, 2023,
    https://arxiv.org/abs/2310.04673
    """
    def __init__(
            self,
            input_size,                     # seq size of text embeddings
            text_encoder: nn.Module,        # encode text inputs
            codec_encoder: nn.Module,       # predict codec_emb according to codec_1st
            vocab_size: int = 0,            # 0 for embedding inputs, > 0 for token inputs such as phoneme
            token_list: List[str] = None,   # None for embedding inputs, not None for token inputs
            pos_enc: str = "abs_pos",
            codec_conf: Dict = None,
            ignore_id: int = -1,
            length_normalized_loss: bool = True,
            lsm_weight: float = 0.1,
            codec_lm_conf: Dict = None,
            codec_sampling_ratio: float = 0.0,
            predict_nq: int = 1,
            pos_emb_type: str = "split",
    ):
        super().__init__()
        if pos_enc in ["sinusoidal", "abs_pos"]:
            pos_enc_class = PositionalEncoding
        elif pos_enc == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc is None:
            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity
        else:
            raise ValueError(f"unknown pos-enc option: {pos_enc}")
        assert pos_emb_type in ["split", "uni"], f"pos_emb_type must be split or uni rather than {pos_emb_type}"

        self.ignore_id = ignore_id
        self.codec_sampling_ratio = codec_sampling_ratio
        self.num_quantizers = num_quantizers = codec_conf.get("num_quantizers", 32)
        self.codebook_size = codebook_size = codec_conf.get("codebook_size", 1024)
        self.codebook_dim = codebook_dim = codec_conf.get("codebook_dim", 128)
        self.predict_nq = predict_nq
        self.pos_emb_func = pos_enc_class(self.codebook_dim, 0.1)
        self.pos_emb_type = pos_emb_type

        # 1. build text inputs related modules
        self.text_encoder = text_encoder
        self.text_enc_out_layer = nn.Linear(
            self.text_encoder.output_size() if text_encoder is not None else input_size,
            self.codebook_dim
        )


        self.transcript_tokenizer = RobertaTokenizer.from_pretrained('/mnt/users/hccl.local/wwu/roberta-base')
         # 1. build text inputs related modules
        self.transcript_encoder =  RobertaModel.from_pretrained('/mnt/users/hccl.local/wwu/roberta-base')
        for key, param in self.transcript_encoder.named_parameters():
            print("freeze: ",key)
            param.requires_grad = False
        
        self.transcript_enc_out_layer = nn.Linear(768, self.codebook_dim)


        self.text_dim = self.transcript_encoder.config.hidden_size  # e.g. 768
        self.mix_dim = 128  # 你的 mix emb 维度，比如 512
             # 或你的 LLM embedding dim
        self.text_mix_prompt = GatedCrossAttPrompt(
            text_dim=self.text_dim,
            mix_dim=self.mix_dim,
            prompt_dim=self.text_dim,
            n_heads=4,
        )


        # # 1. 文本 tokenizer（替代原来的 RobertaTokenizer）
        # # 2. 临时加载一次 Qwen2 模型，只为拿 input embedding 权重
        # print("load qwen emb layer!!!")
        # vocab_size, emb_dim = orig_emb.weight.shape          # [V_text, D_qwen]
        # # 3. 建立自己的 text_embed，并拷贝 Qwen 的 embedding 权重
        # with torch.no_grad():
        #     self.text_embed.weight.copy_(orig_emb.weight)

        # # 4. 冻结 text_embed（按你现在 “查表不训练” 的设定）
        # # 5. 如需映射到你的 codebook_dim，可以加一个线性层（替代原来的 transcript_enc_out_layer）
        # #   Qwen2.5-0.5B 的 hidden_size 是 896

        # # 如果你只想用 Qwen embedding 直接作为 LLM 的 prompt，不需要额外 encoder，可以只用上面两步：
        # #   ids -> self.text_embed -> [B, L, emb_dim]，再和 speech_emb 拼接给 self.codec_lm
        # # # 6. codec_lm 保持你现在的写法，只是把 lm_head 改成预测 speech vocab
        # # self.codec_lm = Qwen2ForCausalLM.from_pretrained(qwen_pretrain_path)
        # # self.codec_lm.lm_head = nn.Linear(
        # #     self.codec_lm.config.hidden_size,  # Qwen2.5-0.5B 是 896
        # #     self.lm_out_voc_size               # 你的 speech codec vocab size
        # # )

                
        # # visual blocks
        # # import pdb;pdb.set_trace()
        #                                           self.codebook_dim)


        self.visual_aux_encoder = nn.LSTM(30, hidden_size=128, num_layers=5, batch_first=True, bidirectional=True, dropout=0.3)
        self.visual_aux_enc_out_layer = nn.Linear(256, self.codebook_dim)
        self.visual_sync_encoder = nn.LSTM(30, hidden_size=128, num_layers=5, batch_first=True, bidirectional=True, dropout=0.3)
        self.visual_sync_enc_out_layer = nn.Linear(256, self.codebook_dim)


        self.av_fuse_mlp = nn.Linear(256, 128)
        self.cue_fuse = nn.Linear(128*3, 128)

        self.codec_emb_residual_scale = 1
        self.fused_residual_scale = 1

        self.vocab_size = vocab_size
        print(f"[DPRINT]: vocab size {self.vocab_size}")
        self.token_list = token_list
        if vocab_size > 0:
            self.token_embedding = torch.nn.Embedding(vocab_size, input_size)

        # 2. build Music language model related moduels
        self.sos_eos = 0
        self.task_id = 1
        self.sep = 2 # Special Token to separate reference and mixture
        # embedding for sos_eos and task id
        self.lm_embedding = torch.nn.Embedding(3, self.codebook_dim)
        self.lm_out_voc_size = (self.codebook_size + 1) * self.predict_nq
        self.codec_lm = self.build_codec_lm(codec_lm_conf)
         

        # 3. build fine codec predictor
        self.codec_encoder = codec_encoder
        self.codec_encoder_out_layer = nn.Linear(codec_encoder.output_size(), self.codebook_dim)

        self.quantizer_codebook = QuantizerCodebook(num_quantizers, codebook_size, codebook_dim)
        self.criterion_ce = LabelSmoothingLoss(
            size=self.lm_out_voc_size // self.predict_nq,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
            reduction=False,
        )

        self.criterion_ce_visual = LabelSmoothingLoss(
            size = 2004 ,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
            reduction=False,
        )
        self.length_normalized_loss = length_normalized_loss
        from funcodec.models.quantizer.costume_quantizer import CostumeQuantizer
        self.quantizer = CostumeQuantizer(
            input_size=self.codebook_dim,
            codebook_size=self.codebook_size,
            num_quantizers=32,
            ema_decay=0.99,
            kmeans_init=True,
            sampling_rate=16000,
            quantize_dropout=False,
            use_ddp=True,
        )

        # for key, param in self.hubert_encoder.named_parameters():

        # for key, param in self.wavlm_model.named_parameters():
         # Load Codec Model
        config_path = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/codec_config/config.yaml'
        model_ckpt = '/mnt/users/hccl.local/wwu/lauraTSE_code_refact/codec_config/model.pth'
        codec_kwargs = dict(
            config_file=config_path,
            model_file=model_ckpt,
        )
        #     **codec_kwargs,


        # for key, param in self.codec_model.named_parameters():

 
    def build_codec_lm(self, conf: Dict):
        name = conf.pop("name")
        if name == "transformer":
            from funcodec.lm.transformer_lm import TransformerEmbedLM
            if "text_vocab_size" in conf:
                lm_model = TransformerEmbedLM(
                    vocab_size=self.lm_out_voc_size,
                    **conf
                )
            else:
                lm_model = TransformerEmbedLM(
                    vocab_size=self.lm_out_voc_size,
                    text_vocab_size=self.lm_out_voc_size,
                    **conf
                )
        else:
            raise TypeError(f"Unknown codec decoder type {name}")

        conf["name"] = name
        return lm_model

    def _target_mask(self, lengths):
        ys_mask = ~make_pad_mask(lengths)
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def encode(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
    ):
        if self.text_encoder is not None:
            outs, out_lens, _ = self.text_encoder(text, text_lengths)
            outs = self.text_enc_out_layer(outs)
        else:
            if text.shape[-1] == self.codebook_dim:
                outs, out_lens = text, text_lengths
            else:
                outs = self.text_enc_out_layer(text)
                out_lens = text_lengths

        return outs, out_lens
    

    def transcript_aux_encode(
            self,
            text: torch.Tensor,
            device, 
    ):
 
        if type(text) == str:
            text_list = [text]
        # 将张量转换为字符串列表
        if len(text) >1:
            text_list = sum(text, [])
        else:
            text_list = text

        encoded_input = self.transcript_tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device)
        outs = self.transcript_encoder(**encoded_input).pooler_output
        outs = self.transcript_enc_out_layer(outs)


        # # 文本 -> BPE ids

        # # BPE ids -> Qwen embedding
        # # 如需再投到 codebook_dim

        return outs
    

    def transcript_aux_encode_gated_att(
            self,
            text: torch.Tensor,
            mix_emb: torch.Tensor,
            device, 
    ):
 
        if type(text) == str:
            text_list = [text]
        # 将张量转换为字符串列表
        if len(text) >1:
            text_list = sum(text, [])
        else:
            text_list = text

        encoded_input = self.transcript_tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device)
        self.use_bert_pool_emb = True
        if self.use_bert_pool_emb:
            text_pooled = self.transcript_encoder(**encoded_input).pooler_output  # (B, D_text)

            # 2) 用 gated cross-att 得到 mix-conditioned text prompt
            text_prompt_token = self.text_mix_prompt(text_pooled, mix_emb).squeeze(1)  # (B, 1, D_prompt)

            # 3) 把 prompt_token 拼在 mix_emb 前面，作为 LLM 输入
            outs = self.transcript_enc_out_layer(text_prompt_token)
        else:
            outs = self.transcript_encoder(**encoded_input).pooler_output
            outs = self.transcript_enc_out_layer(outs)


        # # 文本 -> BPE ids

        # # BPE ids -> Qwen embedding
        # # 如需再投到 codebook_dim

        return outs

    def visual_aux_encode(
                self,
                visual: torch.Tensor,
        ):
            if self.visual_aux_encoder is not None:
                outs,_ = self.visual_aux_encoder(visual)
                outs = self.visual_aux_enc_out_layer(outs)

            return outs.permute(0,2,1)
    def visual_sync_encode(
                self,
                visual: torch.Tensor,
        ):
            if self.visual_sync_encoder is not None:
                outs,_  = self.visual_sync_encoder(visual)
                outs = self.visual_sync_enc_out_layer(outs)

            return outs.permute(0,2,1)


    def build_llm_io_training(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec: Optional[torch.Tensor] = None,
            codec_lengths: Optional[torch.Tensor] = None,
            av_token: Optional[torch.Tensor] = None,
            visual_sync: Optional[torch.Tensor] = None,
            audio_enroll_index: Optional[torch.Tensor] = None,
            visual_enroll_index: Optional[torch.Tensor] = None,
            need_targets: bool = True,
    ):
        """build inputs and targets for language model

                Normally, this function is called in batchify_nll.
                Args:
                    text: (Batch, Length, Dim)
                    text_lengths: (Batch,)
                    codec: (Batch, Length)
                    codec_lengths: (Batch,)
                    need_targets: bool, whether provide targets
                """

 
        if need_targets:
            assert codec is not None and codec_lengths is not None, \
                "need_target=True, but codec or codec_length is None"

        sos_eos_emb = self.lm_embedding(torch.tensor([self.sos_eos], dtype=torch.int64, device=text.device))
        task_id_emb = self.lm_embedding(torch.tensor([self.task_id], dtype=torch.int64, device=text.device))
        codec_emb = None
        if codec is not None and codec_lengths is not None:
            codec_emb = self.calc_dense_vector(codec, codec_lengths)
        if visual_sync is not None and codec_lengths is not None:
            visual_sync = visual_sync[:,1:]
            visual_end_feat = torch.zeros((visual_sync.shape[0],1, visual_sync.shape[-1])).to(visual_sync.device)
            # visual_sync should align target time step!!!, so add eos!!!
            visual_sync = torch.cat((visual_sync, visual_end_feat), dim=1).to(visual_sync.device)

        # assert visual_sync.shape[1] == codec_emb.shape[1]+1
        #concat on channel: b*c*t
        
        inputs_list = []
        for sample_idx, text_len in enumerate(text_lengths):
            one_input = [sos_eos_emb, text[sample_idx, :text_len], task_id_emb]
            if sample_idx in visual_enroll_index:
                use_fused_sync_emb = True
                fused_sync_concat = torch.cat((visual_sync, codec_emb),dim=2).to(visual_sync.device)
                fused_sync_emb_residual = self.av_fuse_mlp(fused_sync_concat)
                fused_sync_emb = self.fused_residual_scale * fused_sync_emb_residual + self.codec_emb_residual_scale * codec_emb
                if use_fused_sync_emb and fused_sync_emb is not None:
                    one_input.append(fused_sync_emb[sample_idx, :codec_lengths[sample_idx]])
            else:
                if codec_emb is not None:
                    one_input.append(codec_emb[sample_idx, :codec_lengths[sample_idx]])
                
            inputs_list.append(torch.cat(one_input, dim=0).to(visual_sync.device))
        llm_inputs = pad_list(inputs_list, 0.0)
        llm_lengths = text_lengths + 2
        if codec_emb is not None:
            llm_lengths = llm_lengths + codec_lengths

         
        if not need_targets:
            return llm_inputs, llm_lengths

        bb, tt = text.shape[0], codec_lengths.max() + 1
        llm_targets = torch.zeros([bb, tt, self.predict_nq], dtype=torch.int64, device=text.device)
        for i, codec_len in enumerate(codec_lengths):
            llm_targets[i, :codec_len] = codec[i, :codec_len]
            llm_targets[i, codec_len] = self.codebook_size + self.sos_eos
        visual_targets = torch.zeros([bb, tt, 1], dtype=torch.int64, device=text.device)
        for i, codec_len in enumerate(codec_lengths):
            visual_targets[i, :codec_len, 0] = av_token[i, :codec_len]
            # visual_targets[i, codec_len] = self.codebook_size + self.sos_eos
            # visual_targets[i, codec_len] = self.codebook_size + self.sos_eos


        return (llm_inputs, llm_targets, visual_targets), (llm_lengths, codec_lengths + 1)


    def build_llm_io_inference(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec: Optional[torch.Tensor] = None,
            codec_lengths: Optional[torch.Tensor] = None,
            visual_sync: Optional[torch.Tensor] = None,
            need_targets: bool = False,
    ):
        """build inputs and targets for language model

                Normally, this function is called in batchify_nll.
                Args:
                    text: (Batch, Length, Dim)
                    text_lengths: (Batch,)
                    codec: (Batch, Length)
                    codec_lengths: (Batch,)
                    need_targets: bool, whether provide targets
                """

        if need_targets:
            assert codec is not None and codec_lengths is not None, \
                "need_target=True, but codec or codec_length is None"

        sos_eos_emb = self.lm_embedding(torch.tensor([self.sos_eos], dtype=torch.int64, device=text.device))
        task_id_emb = self.lm_embedding(torch.tensor([self.task_id], dtype=torch.int64, device=text.device))
        codec_emb = None
        if codec is not None and codec_lengths is not None:
            codec_emb = self.calc_dense_vector(codec, codec_lengths)
        

        if visual_sync is not None and codec is not None and codec_lengths is not None:
            #     # visual_sync should align target time step!!!, so add eos!!!
            #     # visual_sync = torch.cat((visual_sync, sos_eos_emb), dim=1).to(visual_sync.device)
            # else:
            #     # import pdb;pdb.set_trace()

            if (codec_lengths.item() >= visual_sync.shape[1]) == True:
                 
                # visual_sync should align target time step!!!, so add eos!!!
                visual_sync_pading_length = codec_lengths+1 - visual_sync.shape[1]
                visual_end_feat = torch.zeros((visual_sync.shape[0],visual_sync_pading_length, visual_sync.shape[-1])).to(visual_sync.device)
                visual_sync = torch.cat((visual_sync, visual_end_feat), dim=1).to(visual_sync.device)
            # else:
            #     # import pdb;pdb.set_trace()
            visual_sync_current = visual_sync[:,1:1+codec_lengths].to(visual_sync.device)
            

            # assert visual_sync.shape[1] == codec_emb.shape[1]+1
            #concat on channel: b * t * c
            fused_sync_concat = torch.cat((visual_sync_current, codec_emb),dim=2)
            fused_sync_emb_residual = self.av_fuse_mlp(fused_sync_concat)
            fused_sync_emb = self.fused_residual_scale * fused_sync_emb_residual + self.codec_emb_residual_scale * codec_emb
            
            use_fused_sync_emb = True
        else:
            use_fused_sync_emb = False
            fused_sync_emb = None

        
        inputs_list = []
        for i, text_len in enumerate(text_lengths):
            one_input = [sos_eos_emb, text[i, :text_len], task_id_emb]
            if use_fused_sync_emb and codec_emb is not None and fused_sync_emb is not None:
                one_input.append(fused_sync_emb[i, :codec_lengths[i]])
            else:
                if codec_emb is not None:
                    one_input.append(codec_emb[i, :codec_lengths[i]])
             

            inputs_list.append(torch.cat(one_input, dim=0))
        llm_inputs = pad_list(inputs_list, 0.0)
        llm_lengths = text_lengths + 2
        if codec_emb is not None:
            llm_lengths = llm_lengths + codec_lengths

         
        if not need_targets:
            return llm_inputs, llm_lengths

        bb, tt = text.shape[0], codec_lengths.max() + 1
        llm_targets = torch.zeros([bb, tt, self.predict_nq], dtype=torch.int64, device=text.device)
        for i, codec_len in enumerate(codec_lengths):
            llm_targets[i, :codec_len] = codec[i, :codec_len]
            llm_targets[i, codec_len] = self.codebook_size + self.sos_eos
        visual_targets = torch.zeros([bb, tt, 1], dtype=torch.int64, device=text.device)
        # for i, codec_len in enumerate(codec_lengths):
        #     # import pdb;pdb.set_trace()
        #     visual_targets[i, :codec_len, 0] = av_token[i, :codec_len]
        #     # visual_targets[i, codec_len] = self.codebook_size + self.sos_eos
        #     # visual_targets[i, codec_len] = self.codebook_size + self.sos_eos


        return (llm_inputs, llm_targets, visual_targets), (llm_lengths, codec_lengths + 1)

    def build_llm_io(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec: Optional[torch.Tensor] = None,
            codec_lengths: Optional[torch.Tensor] = None,
            need_targets: bool = True,
    ):
        """build inputs and targets for language model

                Normally, this function is called in batchify_nll.
                Args:
                    text: (Batch, Length, Dim)
                    text_lengths: (Batch,)
                    codec: (Batch, Length)
                    codec_lengths: (Batch,)
                    need_targets: bool, whether provide targets
                """

        if need_targets:
            assert codec is not None and codec_lengths is not None, \
                "need_target=True, but codec or codec_length is None"

        sos_eos_emb = self.lm_embedding(torch.tensor([self.sos_eos], dtype=torch.int64, device=text.device))
        task_id_emb = self.lm_embedding(torch.tensor([self.task_id], dtype=torch.int64, device=text.device))
        codec_emb = None
        if codec is not None and codec_lengths is not None:
            codec_emb = self.calc_dense_vector(codec, codec_lengths)
        inputs_list = []
        for i, text_len in enumerate(text_lengths):
            one_input = [sos_eos_emb, text[i, :text_len], task_id_emb]
            if codec_emb is not None:
                one_input.append(codec_emb[i, :codec_lengths[i]])
            inputs_list.append(torch.cat(one_input, dim=0))
        llm_inputs = pad_list(inputs_list, 0.0)
        llm_lengths = text_lengths + 2
        if codec_emb is not None:
            llm_lengths = llm_lengths + codec_lengths

        if not need_targets:
            return llm_inputs, llm_lengths

        bb, tt = text.shape[0], codec_lengths.max() + 1
        llm_targets = torch.zeros([bb, tt, self.predict_nq], dtype=torch.int64, device=text.device)
        for i, codec_len in enumerate(codec_lengths):
            llm_targets[i, :codec_len] = codec[i, :codec_len]
            llm_targets[i, codec_len] = self.codebook_size + self.sos_eos

        return (llm_inputs, llm_targets), (llm_lengths, codec_lengths + 1)


    def nll(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        codec: Optional[torch.Tensor] = None,
        codec_lengths: Optional[torch.Tensor] = None,
        av_token: Optional[torch.Tensor] = None,
        visual_sync: Optional[torch.Tensor] = None,
        audio_enroll_index: Optional[torch.Tensor] = None,
        visual_enroll_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor, torch.Tensor]:
        """Compute negative log likelihood(nll)

        Normally, this function is called in batchify_nll.
        Args:
            text: (Batch, Length, Dim)
            text_lengths: (Batch,)
            codec: (Batch, Length)
            codec_lengths: (Batch,)
        """
        batch_size = text.size(0)
        # For data parallel
        text = text[:, :text_lengths.max()]
        codec = codec[:, :codec_lengths.max()]
        av_token = av_token[:, :codec_lengths.max()]
        visual_sync =  visual_sync[:, :codec_lengths.max()]

        # build inputs and targets for language model
        (sequence, target, visual_targets), (x_lengths, y_lengths) = self.build_llm_io_training(
            text, text_lengths,
            codec, codec_lengths,
            av_token,
            visual_sync,
            audio_enroll_index, 
            visual_enroll_index,
            need_targets=True
        )

        # 2a. Forward Language model
        # x: (Batch, Length) -> y: (Batch, Length, NVocab)
        sequence = sequence[:, :x_lengths.max()]
        target = target[:, :y_lengths.max()]
        visual_targets = visual_targets[:, :y_lengths.max()]
        y, y_visual, _ = self.codec_lm(sequence, x_lengths, text_lengths+1)
        bb, tt, tt_v = y.shape[0], y.shape[1], y_visual.shape[1]
        y = y.reshape(bb, tt, self.predict_nq, -1)
    
        # 2b. Extract real logits
        logits_list = []
        logits_list_visual = []
        for i, (text_len, codec_len) in enumerate(zip(text_lengths, codec_lengths)):
            logits_list.append(y[i, text_len + 1:text_len + 2 + codec_len])
            logits_list_visual.append(y_visual[i, text_len + 1:text_len + 2 + codec_len])
        logits = pad_list(logits_list, 0.0)
        logits_visual = pad_list(logits_list_visual, 0.0)

        # 3. Calc negative log likelihood
        tt = logits.shape[1]
        nll = self.criterion_ce(logits.reshape(bb, tt * self.predict_nq, -1),target.reshape(bb, tt * self.predict_nq))
        nll_visual = self.criterion_ce_visual(
            logits_visual,
            visual_targets.squeeze(-1),
        )
        nll = nll.sum(-1)
         
        # nll: (BxL,) -> (BxL,)
        nll.masked_fill_(make_pad_mask(y_lengths * self.predict_nq).to(nll.device).view(-1), 0.0)
        # nll: (BxL,) -> (B, L)
        nll = nll.reshape(batch_size, -1).reshape(batch_size, tt, self.predict_nq)

        nll_visual = nll_visual.sum(-1)
        nll_visual = nll_visual.reshape(batch_size, -1).reshape(batch_size, tt, 1)


        return nll, nll_visual, logits, target, codec_lengths+1

    
    def cal_codec_emb(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec_prob: torch.Tensor,
            codec_lengths: torch.Tensor,
    ):
        first_nq_emb = None
        for i in range(self.predict_nq):
            one_emb = torch.matmul(codec_prob[:, :, i], self.quantizer_codebook.embed[i:i+1].detach()) #[B,T,1024] * [1, 1024, D] = [B, T, D]
            if first_nq_emb is None:
                first_nq_emb = one_emb
            else:
                first_nq_emb = first_nq_emb + one_emb
        model_inputs = []
        for i, (text_len, codec_len) in enumerate(zip(text_lengths, codec_lengths)):
            if self.pos_emb_type == "split":
                one_in = [
                    self.pos_emb_func(text[i:i+1, :text_len]).squeeze(0),
                    self.pos_emb_func(first_nq_emb[i:i+1, :codec_len]).squeeze(0)
                ]
            else:
                one_in = [text[i, :text_len], first_nq_emb[i, :codec_len]]
            model_inputs.append(torch.cat(one_in, dim=0))
        model_input_lengths = text_lengths + codec_lengths
        model_inputs = pad_list(model_inputs, 0.0)
        model_inputs = model_inputs[:, :model_input_lengths.max()]

        model_outs, model_outs_lens, _ = self.codec_encoder(model_inputs, model_input_lengths)
        model_outs = self.codec_encoder_out_layer(model_outs)

        outs = torch.zeros([text.shape[0], codec_lengths.max(), self.codebook_dim], requires_grad=True).to(text)
        for i, (text_len, codec_len) in enumerate(zip(text_lengths, codec_lengths)):
            outs[i, :codec_len] = model_outs[i, text_len: text_len+codec_len]

        return outs, codec_lengths

    def calc_reg_loss(self, prediction, target, length):
        loss_mask = ~make_pad_mask(length, target)
        l1_loss = F.l1_loss(prediction, target, reduction="none")
        l1_loss = (l1_loss * loss_mask).sum() / loss_mask.sum()
        l2_loss = 0.5 * F.mse_loss(prediction, target, reduction="none")
        l2_loss = (l2_loss * loss_mask).sum() / loss_mask.sum()

        return l1_loss * 0.5 + l2_loss * 0.5, l1_loss, l2_loss
    def calc_ssl_loss(self, prediction, target):

        l2_loss = torch.mean(F.mse_loss(prediction, target, reduction="none"))

        return  l2_loss

    def calc_dense_vector(self, codec, codec_lengths):
        """
        Args:
            codec: (B, T, Nq)
            codec_lengths: (B, )
        """
        with torch.no_grad():
            return self.quantizer_codebook(codec, codec_lengths)

    def prob_sampler(
            self,
            logits: torch.Tensor,
            codec: torch.Tensor,
            codec_lengths: torch.Tensor,
    ):
        """ Sampling ground-truth prob to replace wrongly predicted prob
        Args:
            logits: (B, T, N, V)
            codec: (B, T, N)
            codec_lengths: (B,)
        """
        assert logits.shape[1] == codec.shape[1], \
            f"lengths of logits and codec mismatch: {logits.shape[1]} and {codec.shape[1]}"
        bb, tt = logits.shape[0], logits.shape[1]
        valid_mask = (~make_pad_mask(codec_lengths)).view(bb, tt, 1, 1).to(logits.device)

        soft_prob = torch.softmax(logits, dim=-1)
        pred_token = torch.argmax(soft_prob, dim=-1)
        hard_prob = F.one_hot(pred_token, self.codebook_size).float()
        # go-through gradient estimation
        pred_prob = soft_prob + (hard_prob - soft_prob).detach()
        if self.codec_sampling_ratio == 0.0:
            return pred_prob * valid_mask

        gt_prob = F.one_hot(
            torch.clamp(codec, 0, self.codebook_size - 1),
            self.codebook_size
        ).float()
        if self.codec_sampling_ratio == 1.0:
            return gt_prob * valid_mask

        # bb, tt, nn
        correct_mask = (pred_token == codec)
        # higher codec_sampling_ratio means less prediction usage
        sampling_mask = torch.rand_like(correct_mask.float()) > self.codec_sampling_ratio
        # for correct tokens or (wrong tokens without sampling), we use predictions
        input_mask = (torch.logical_or(
            correct_mask,
            torch.logical_and(~correct_mask, sampling_mask))
        ).unsqueeze(-1)
        prob = input_mask * pred_prob + (~input_mask) * gt_prob

        # masking out the padding part
        return prob * valid_mask

    def forward(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            aux: torch.Tensor,
            aux_lengths: torch.Tensor, 
            codec: torch.Tensor,
            codec_lengths: torch.Tensor,
            av_token:torch.Tensor,
            visual_aux:torch.Tensor,
            visual_sync:torch.Tensor,
            transcript_aux:torch.Tensor,
            tgts:torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            text: (B, L, D) The mixture Log-Mel Spectrogram
            text_lengths: (B,)
            aux: (B, L, D) ## The referene Log-Mel Spectrogram
            aux_lengths: (B,)
            codec: (B, L, N) ## The target clean codec
            codec_lengths: (B,)
        """
        text = text[:, :text_lengths.max()]
        aux = aux[:, :aux_lengths.max()]
        codec = codec[:, :codec_lengths.max()].long()
        visual_sync = visual_sync[:, :codec_lengths.max()]
        av_token = av_token[:,:codec_lengths.max()].long()


        # 1. encode text and ref
        text, text_lengths = self.encode(text, text_lengths)
        aux, aux_lengths = self.encode(aux, aux_lengths) # [B, T, D]
        tgts, tgts_lengths = self.encode(tgts, text_lengths) 
        tgts_aux_part = tgts[:,:aux.shape[-2],:].to(tgts.device)
        transcript_aux = self.transcript_aux_encode_gated_att(text=transcript_aux, mix_emb=text, device=visual_aux.device) # [B,D]

        transcript_aux = transcript_aux.unsqueeze(1).expand(transcript_aux.shape[0],  aux_lengths.max(), -1).to(tgts.device)  # [B, T_max, D]

       
        visual_aux = self.visual_aux_encode(visual_aux)
        if visual_aux.shape[-1]< aux_lengths.max():
            visual_aux = F.interpolate(visual_aux, int((gesture_aux_upsample_factor * visual_aux.shape[-1])), mode='linear').to(text.device)
            visual_aux = F.pad(visual_aux, (0, aux_lengths.max() - visual_aux.shape[-1])).to(text.device)
        else:
            visual_aux = visual_aux[:,:,:aux_lengths.max()]
        visual_aux = visual_aux.permute(0,2,1)


        visual_sync = self.visual_sync_encode(visual_sync)
        if visual_sync.shape[-1]< codec_lengths.max():
            visual_sync = F.interpolate(visual_sync, int((gesture_codec_upsample_factor * visual_sync.shape[-1])), mode='linear').to(text.device)
            visual_sync = F.pad(visual_sync, (0, codec_lengths.max() - visual_sync.shape[-1])).to(text.device)
        else:
            visual_sync = visual_sync[:,:,:codec_lengths.max()]
        visual_sync = visual_sync.permute(0,2,1)
         
         
        sep_emb = self.lm_embedding(torch.tensor([self.sep], dtype=torch.int64, device=text.device)) # [1, D]
        
        inputs_list = [] # [[T1,D], [T2,D]]
        llm_lengths = []
        batch_size = text.size(0)

        # 7 种组合: A, V, T, A+V, A+T, V+T, A+V+T
        combo_tensor = torch.tensor([
            [1, 0, 0],  # A
            [0, 1, 0],  # V
            [0, 0, 1],  # T
            [1, 1, 0],  # A+V
            [1, 0, 1],  # A+T
            [0, 1, 1],  # V+T
            [1, 1, 1],  # A+V+T
        ], device=text.device, dtype=torch.bool)  # [7, 3]

        # 确保每个batch包含所有组合（如果可能）
        if batch_size >= 7:
            # 先包含所有7种组合
            idx = torch.arange(7, device=text.device)
            
            # 如果batch_size > 7，随机补充剩余的位置
            if batch_size > 7:
                extra_idx = torch.randint(0, 7, (batch_size - 7,), device=text.device)
                idx = torch.cat([idx, extra_idx])
            
            # 打乱顺序
            idx = idx[torch.randperm(batch_size, device=text.device)]
        else:
            # 如果batch_size < 7，随机选择不重复的组合
            idx = torch.randperm(7, device=text.device)[:batch_size]

        cue_mask = combo_tensor[idx]  # [B, 3], 对应 [A, V, T]
        # ===== 记录使用 Visual cue 的样本下标 =====
        # cue_mask[:, 1] 对应是否使用 V
        visual_used_mask = cue_mask[:, 1]                 # [B] bool
        visual_indices   = visual_used_mask.nonzero(as_tuple=True)[0].to(text.device)      # 用到 V 的样本下标
        non_visual_indices = (~visual_used_mask).nonzero(as_tuple=True)[0].to(text.device)   # 不含 V 的样本下标
        for text_idx in range(batch_size):
            _t = text[text_idx][:text_lengths[text_idx].item()]          # [T, D]
            _a = aux[text_idx][:aux_lengths[text_idx].item()]            # [T, D]
            _v = visual_aux[text_idx][:aux_lengths[text_idx].item()]     # [T, D]
            _u = transcript_aux[text_idx]                                # [D] 或 [T,D]

            T_len, D = _a.shape
            device = _a.device

            # 3 个模态的帧级特征，默认 0
            a_feat = torch.zeros(T_len, D, device=device)
            v_feat = torch.zeros(T_len, D, device=device)
            u_feat = torch.zeros(T_len, D, device=device)

            use_A = cue_mask[text_idx, 0].item()
            use_V = cue_mask[text_idx, 1].item()
            use_T = cue_mask[text_idx, 2].item()

            if use_A:
                a_feat = _a                          # [T, D]
            if use_V:
                v_feat = _v                          # [T, D]
            if use_T:
                # 如果 transcript 是句级 [D]，扩成 [T,D]
                if _u.dim() == 1:
                    u_feat = _u.unsqueeze(0).expand(T_len, -1)
                else:
                    u_feat = _u[:T_len]              # 已是 [T, D]

            # [T, 3D] -> Linear(3D->D) -> [T, D]
            fuse_input = torch.cat([a_feat, v_feat, u_feat], dim=-1)  # [T, 3D]
            _cue = self.cue_fuse(fuse_input)                          # [T, D]

            # sep_emb: [D] 或 [1,D]
            sep = sep_emb.unsqueeze(0) if sep_emb.dim() == 1 else sep_emb  # [1,D] / [K,D]
            one_input = torch.cat([_cue, sep, _t], dim=0)                  # [T+1+T', D]
            # 后续把 one_input 堆回 batch …
            inputs_list.append(one_input)
            llm_lengths.append(len(one_input))
        llm_inputs = pad_list(inputs_list, 0.0)
        llm_lengths = torch.tensor(llm_lengths, dtype = torch.long, device = text.device)
        
        ## Assign it to text
        text = llm_inputs
        text_lengths = llm_lengths
        text = text[:, :text_lengths.max()]

        # 2. generate the first `predict_nq` codec groups
        # nll, nll_visual, logits, target, target_lengths = self.nll(text, text_lengths, codec[:, :, :self.predict_nq], codec_lengths, av_token, visual_sync)
        nll, nll_visual, logits, target, target_lengths = self.nll(text, text_lengths, codec[:, :, :self.predict_nq], codec_lengths, av_token, visual_sync, audio_enroll_index = non_visual_indices, visual_enroll_index = visual_indices)

        output_mask = ~make_pad_mask(target_lengths, maxlen=target_lengths.max()).to(text.device).unsqueeze(-1)
        total, batch_size = output_mask.sum() * self.predict_nq, nll.shape[0] * self.predict_nq
        denom = total if self.length_normalized_loss else batch_size
        nll_loss = (nll * output_mask).sum() / denom
        
        denom_visual =  denom / 2
        nll_visual_loss =  (nll_visual).sum() / denom_visual

        # 3. generate dense codec vectors
        # logits: [B, T-1(remove eos), n_q, 1024,]
        # sampling codec prob
        prob = self.prob_sampler(
            # remove <eos> from logits
            logits[:, :-1, :self.predict_nq, :self.codebook_size], 
            codec[:, :, :self.predict_nq], # [B, T, n_q]
            codec_lengths
        ) # [B, T, n_q, 1024]
         
        codec_emb, codec_emb_lens = self.cal_codec_emb(text, text_lengths, prob, codec_lengths)
        # 4. loss calculation
        target_emb = self.calc_dense_vector(codec, codec_lengths)

        # # 5. align on semantic

        reg_loss, l1_loss, l2_loss = self.calc_reg_loss(codec_emb, target_emb, codec_lengths)
        av_front_align_mse_loss = self.calc_ssl_loss(visual_aux, tgts_aux_part)

        # 3 loss :111
        loss = reg_loss + nll_loss 
        # 10* visual loss + speech ce loss + speech reg_loss


        stats = dict(
            loss=loss.detach(),
            nll_visual_loss = nll_visual_loss.detach(),
            nll_loss=nll_loss.detach(),
            reg_loss=reg_loss.detach(),
            reg_l1_loss=l1_loss.detach(),
            reg_l2_loss=l2_loss.detach(),
            av_front_align_mse_loss = av_front_align_mse_loss.detach(),
            batch_size=text.shape[0],
            seq_length=text_lengths.max() + codec_lengths.max(),
        )

        # 5. accuracy calculation
        with torch.no_grad():
            cc = logits.shape[-1]
            for i in range(self.predict_nq):
                acc = th_accuracy(
                    logits[:, :, i, :].reshape(-1, cc),
                    target[:, :, i],
                    self.ignore_id
                )
                stats[f"out_acc_{i+1}"] = acc

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def sampling_ids(
            self,
            weighted_scores: torch.Tensor,
            sampling: Union[bool, int, float] = True,
            beam_size: int = 1,
    ):
        if isinstance(sampling, bool):
            if sampling:
                top_ids = weighted_scores.softmax(dim=0).multinomial(beam_size, replacement=True)
            else:
                top_ids = weighted_scores.topk(beam_size)[1]
        elif isinstance(sampling, int):
            temperature = 1

            probs = (weighted_scores / temperature).softmax(dim=0)
            prob, indices = probs.topk(sampling)
            # prob, indices = weighted_scores.softmax(dim=0).topk(sampling)
            sampling_ids = prob.multinomial(beam_size, replacement=True)
            top_ids = indices[sampling_ids]
        elif isinstance(sampling, float):
            prob, indices = [], []
            cum_prob = 0.0
            sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
            for i in range(len(sorted_idx)):
                if cum_prob < sampling:
                    cum_prob += sorted_value[i]
                    prob.append(sorted_value[i])
                    indices.append(sorted_idx[i])
                else:
                    break
            prob = torch.tensor(prob).to(weighted_scores)
            indices = torch.tensor(indices, dtype=torch.long).to(weighted_scores.device)
            sampling_ids = prob.multinomial(beam_size, replacement=True)
            top_ids = indices[sampling_ids]
        else:
            raise NotImplementedError(f"Not implemented for {type(sampling)} sampling")

        return top_ids
    

    def decode_codec(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            max_length: int = 30 * 25,
            sampling: Union[bool, int, float] = True,
            beam_size: int = 1,
            continual: List = None,
    ) -> torch.Tensor:
        device = text.device
        out_tokens = [] if continual is None else deepcopy(continual)
        sos_eos_emb = self.lm_embedding(torch.tensor([[self.sos_eos]], dtype=torch.int64, device=device)) # [1,1,D]
        task_id_emb = self.lm_embedding(torch.tensor([[self.task_id]], dtype=torch.int64, device=device)) # [1,1,D]
        prompt = torch.cat([sos_eos_emb, text, task_id_emb], dim=1)
        state = None
        
        for i in range(max_length):
            if len(out_tokens) > 0:
                codec_prompt = torch.tensor([out_tokens], dtype=torch.int64, device=device)
                codec_lengths = torch.tensor([len(out_tokens)], dtype=torch.int64, device=device)
                # if any quantizer output is eos
                if torch.any(codec_prompt[:, -1] == (self.codebook_size+self.sos_eos)):
                    break
                seq_input, _ = self.build_llm_io(
                    text, text_lengths,
                    codec_prompt, codec_lengths,
                    need_targets=False
                )
            else:
                seq_input, _ = self.build_llm_io(
                    text, text_lengths, None, None,
                    need_targets=False
                )

            # not use state, since has not aligned
            pred, _ = self.codec_lm.score(seq_input[0], state, prompt[0])
            # sampling all `nq` token ids
            pred = pred.reshape(self.predict_nq, -1)
            top_ids = []
            for k in range(self.predict_nq):
                top_ids.append(self.sampling_ids(pred[k], sampling, beam_size)[0].item())
            out_tokens.append(top_ids)
        # remove eos token
        if torch.any(torch.tensor(out_tokens[-1], dtype=torch.int64) == self.codebook_size+self.sos_eos):
            out_tokens = out_tokens[:-1]

        return torch.tensor([out_tokens], dtype=torch.int64, device=device) # [1,T,n_q]


    def decode_codec_visual_cue(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            visual_sync: torch.Tensor,
            max_length: int = 30 * 25,
            sampling: Union[bool, int, float] = True,
            beam_size: int = 1,
            continual: List = None,
    ) -> torch.Tensor:
        device = text.device
        out_tokens = [] if continual is None else deepcopy(continual)
        sos_eos_emb = self.lm_embedding(torch.tensor([[self.sos_eos]], dtype=torch.int64, device=device)) # [1,1,D]
        task_id_emb = self.lm_embedding(torch.tensor([[self.task_id]], dtype=torch.int64, device=device)) # [1,1,D]
        prompt = torch.cat([sos_eos_emb, text, task_id_emb], dim=1)
        state = None
        
        for i in range(max_length):
            if len(out_tokens) > 0:
                codec_prompt = torch.tensor([out_tokens], dtype=torch.int64, device=device)
                codec_lengths = torch.tensor([len(out_tokens)], dtype=torch.int64, device=device)
                # if any quantizer output is eos
                if torch.any(codec_prompt[:, -1] == (self.codebook_size+self.sos_eos)):
                    break
                # seq_input, _ = self.build_llm_io_inference(
                #     text, text_lengths,
                #     codec_prompt, codec_lengths,
                # build inputs and targets for language model
                sequence , x_lengths = self.build_llm_io_inference(
                    text, text_lengths,
                    codec_prompt, codec_lengths,
                    visual_sync,
                    need_targets=False
                )
            else:
                sequence , x_lengths = self.build_llm_io_inference(
                    text, text_lengths,
                    None, None,
                    visual_sync,
                    need_targets=False
                )

            # not use state, since has not aligned
            pred, _ = self.codec_lm.score(sequence[0], state, prompt[0])
            # sampling all `nq` token ids
            pred = pred.reshape(self.predict_nq, -1)
            top_ids = []
            for k in range(self.predict_nq):
                top_ids.append(self.sampling_ids(pred[k], sampling, beam_size)[0].item())
            out_tokens.append(top_ids)
        # remove eos token
        if torch.any(torch.tensor(out_tokens[-1], dtype=torch.int64) == self.codebook_size+self.sos_eos):
            out_tokens = out_tokens[:-1]

        return torch.tensor([out_tokens], dtype=torch.int64, device=device) # [1,T,n_q]

    def syn_audio(
            self,
            codec: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec_model,
            continual_length=None,
    ):
        if codec ==[]:
            return None
        codec = codec[:, :, :self.predict_nq]
        prob = F.one_hot(
            torch.clamp(codec, 0, self.codebook_size-1),
            self.codebook_size
        ).float()
        codec_lengths = torch.tensor([codec.shape[1]], dtype=torch.int64, device=text.device)
        codec_emb, codec_emb_lens = self.cal_codec_emb(text, text_lengths, prob, codec_lengths)
        _, _, recon_wav, _ = codec_model(codec_emb[:, continual_length:], run_mod="decode_emb")

        return recon_wav

    def collect_feats(
            self,
            text: torch.Tensor,
            text_lengths: torch.Tensor,
            codec: torch.Tensor,
            codec_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        feats, feats_lengths = codec, codec_lengths

        return {"feats": feats, "feats_lengths": feats_lengths}
