from torch import nn
import torch
from argparse import Namespace
from funcodec.torch_utils.load_pretrained_model import load_pretrained_model
from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from funcodec.utils.misc import statistic_model_parameters
from funcodec.bin.codec_inference import Speech2Token
from _funcodec import build_model
import torch.nn.functional as F
import numpy as np


 

Mel_len = 63
gesture_FPS = 15
lip_FPS = 25 
gesture_aux_upsample_factor = Mel_len/gesture_FPS
lip_aux_upsample_factor = Mel_len/lip_FPS
gesture_codec_upsample_factor = lip_FPS/gesture_FPS
 

def load_model(model, state_dict):
    model_dict = model.state_dict()
    # 1. 检查是否有 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 1.1 如果当前模型是多卡（有 'module.' 前缀）但加载的参数没有 'module.' 前缀
        if k.startswith("module.") and not any(
            key.startswith("module.") for key in model_dict
        ):
            new_key = k[len("module.") :]  # 去掉 'module.' 前缀
            new_state_dict[new_key] = v
            print("loading: ", k)

        # 1.2 如果当前模型是单卡（没有 'module.' 前缀）但加载的参数有 'module.' 前缀
        elif not k.startswith("module.") and any(
            key.startswith("module.") for key in model_dict
        ):
            new_key = "module." + k  # 添加 'module.' 前缀
            new_state_dict[new_key] = v
            print("loading: ", k)

        # 1.3 当前模型和加载的参数前缀一致
        else:
            new_state_dict[k] = v
            print("loading: ", k)
            
    # 2. 检查模型结构是否一致
    for k, v in model_dict.items():
        if k in new_state_dict:
            try:
                model_dict[k].copy_(new_state_dict[k])
            except Exception as e:
                print(f"Error in copying parameter {k}: {e}")
        else:
            # pdb.set_trace()
            print(f"Parameter {k} not found in checkpoint. Skipping...")
    # 3. 更新模型参数
    model.load_state_dict(model_dict)

    return model


class TSExtraction:
    def __init__(self, args: Namespace, model_ckpt: str, device, logger):
        # Load Laura GPT Model #
        model: nn.Module = build_model(args)
        model.to(device)
        print(model)
        print(args.init_param)
        for p in args.init_param:
            load_pretrained_model(
                model=model,
                init_param=p,
                ignore_init_mismatch=True,
                # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
                #   in PyTorch<=1.4
                map_location=device,
            )
        logger.info("model: {}".format(model))
        logger.info(
            "model parameter number: {}".format(statistic_model_parameters(model))
        )

        # Load Ckpt #
        ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
        if 'pth' in model_ckpt:
            load_model(model, ckpt["model_state_dict"])
            # model.load_state_dict(ckpt["model_state_dict"])
        else:
            # model.load_state_dict(ckpt["model"])
            load_model(model, ckpt["model"])
        model.eval()
        self.model = model
        logger.info("model loaded successfully!")

        # model.load_state_dict(ckpt["model_state_dict"])
        # model.eval()
        # logger.info("model loaded successfully!")

        # Load Codec Model
        codec_kwargs = dict(
            config_file=args["codec_config_file"],
            model_file=args["codec_model_file"],
            device=device,
        )
        self.codec_model = Speech2Token.from_pretrained(
            model_tag=None,
            **codec_kwargs,
        )

        # sampling and beam_size
        self.sampling = args.sampling
        self.beam_size = args.beam_size


        # # remove weight norm in the model and set to eval mode
        # self.bigvgan_model.remove_weight_norm()


    @torch.no_grad()
    def __call__(self, mix_mel:torch.Tensor, ref_mel:torch.Tensor, visual_sync:torch.Tensor, transcript_aux:torch.Tensor, enroll_setting: str, enroll_length:int, enroll_stage: str, tgt_scp_path:None):
        """
        This function can also be used as TSE Inference.
        mix_mel the mep spec of the mixture: [1, T, D]
        ref_mel is the reference mel : [1, T, D]
        """

        continual = None
        continual_length = None
         
        # 1. Encode mix mel and ref mel
        mix_mel_lens = torch.tensor([mix_mel.size(1)], dtype=torch.long, device=mix_mel.device) # [1]
        aux_mel_lens = torch.tensor([ref_mel.size(1)], dtype=torch.long, device=ref_mel.device) # [1]
        visual_sync_lens =  torch.tensor([visual_sync.size(1)], dtype=torch.long, device=ref_mel.device) # [1]
        visual_sync = visual_sync.float().to(ref_mel.device)
        visual_aux = visual_sync[:, :int(enroll_length * gesture_FPS)].float().to(ref_mel.device)

        mix, _ = self.model.encode(mix_mel, mix_mel_lens) # [1,T,D]
        aux, aux_lengths = self.model.encode(ref_mel, aux_mel_lens) # [1,T,D]
 
        visual_aux = self.model.visual_aux_encode(visual_aux)
        # visual_aux[:] = 0
        if visual_aux.shape[-1]< aux_mel_lens:
            visual_aux = F.interpolate(visual_aux, int((gesture_aux_upsample_factor * visual_aux.shape[-1])), mode='linear').to(ref_mel.device)
            visual_aux = F.pad(visual_aux, (0, aux_mel_lens - visual_aux.shape[-1])).to(ref_mel.device)
        else:
            visual_aux = visual_aux[:,:,:aux_mel_lens]

        visual_aux = visual_aux.permute(0,2,1)
        visual_sync = self.model.visual_sync_encode(visual_sync)
        # visual_sync[:] = 0
        if visual_sync.shape[-1] < (4 * lip_FPS):
            visual_sync = F.interpolate(visual_sync, int((gesture_codec_upsample_factor * visual_sync.shape[-1])), mode='linear').to(ref_mel.device)
             
            visual_sync = F.pad(visual_sync, (0, (4 * lip_FPS) - visual_sync.shape[-1])).to(ref_mel.device)
        else:
            visual_sync = visual_sync[:,:,:(4 * lip_FPS)]
        visual_sync = visual_sync.permute(0,2,1)
         
        # visual_sync[:] = 0
        transcript_aux = self.model.transcript_aux_encode([transcript_aux], visual_aux.device) # [B,D]
        transcript_aux = transcript_aux.unsqueeze(1).expand(transcript_aux.shape[0],  aux_lengths.max(), -1).to(visual_aux.device)  # [B, T_max, D]
        sep = self.model.lm_embedding(torch.tensor([[self.model.sep]], dtype = torch.int64, device = visual_aux.device)) # [1,1,D]


        T_len, D = ref_mel.shape[-2],  ref_mel.shape[-1]
        device = ref_mel.device

        # 3 个模态的帧级特征，默认 0
        a_feat = torch.zeros(1, T_len, D, device=device)
        v_feat = torch.zeros(1, T_len, D, device=device)
        u_feat = torch.zeros(1, T_len, D, device=device)

        if 'visual_enroll' in enroll_setting:
            v_feat = visual_aux
            if enroll_setting =='audio_visual_transcript_enroll':
                a_feat = aux
                u_feat = transcript_aux
                fuse_input = torch.cat([a_feat, v_feat, u_feat], dim=-1)  # [T, 3D]
                _cue = self.model.cue_fuse(fuse_input)   
                text_outs = torch.cat([_cue, sep, mix], dim = 1) # [1, T', D]
            elif enroll_setting =='visual_transcript_enroll':
                u_feat = transcript_aux
                fuse_input = torch.cat([a_feat, v_feat, u_feat], dim=-1)  # [T, 3D]
                _cue = self.model.cue_fuse(fuse_input)   
                text_outs = torch.cat([_cue, sep, mix], dim = 1) # [1, T', D]
            elif enroll_setting =='audio_visual_enroll':
                a_feat = aux
                fuse_input = torch.cat([a_feat, v_feat, u_feat], dim=-1)  # [T, 3D]
                _cue = self.model.cue_fuse(fuse_input)   
                text_outs = torch.cat([_cue, sep, mix], dim = 1) # [1, T', D]
            else:  # visual_enroll
                fuse_input = torch.cat([a_feat, v_feat, u_feat], dim=-1)  # [T, 3D]
                _cue = self.model.cue_fuse(fuse_input)   
                text_outs = torch.cat([_cue, sep, mix], dim = 1) # [1, T', D]

            text_out_lens = torch.tensor([text_outs.size(1)], dtype=torch.long, device=text_outs.device) # [1]
            # 2. decode first codec group
            decoded_codec = self.model.decode_codec_visual_cue(
                text_outs,
                text_out_lens,
                visual_sync,
                max_length=30 * 25,
                sampling=self.sampling,
                beam_size=self.beam_size,
                continual=continual,
            )
        else:
            if enroll_setting =='audio_transcript_enroll':
                a_feat = aux
                u_feat = transcript_aux
                fuse_input = torch.cat([a_feat, v_feat, u_feat], dim=-1)  # [T, 3D]
                _cue = self.model.cue_fuse(fuse_input)   
                text_outs = torch.cat([_cue, sep, mix], dim = 1) # [1, T', D]
            elif enroll_setting =='transcript_enroll':
                u_feat = transcript_aux
                fuse_input = torch.cat([a_feat, v_feat, u_feat], dim=-1)  # [T, 3D]
                _cue = self.model.cue_fuse(fuse_input)   
                text_outs = torch.cat([_cue, sep, mix], dim = 1) # [1, T', D]

            else: # audio_enroll
                a_feat = aux
                fuse_input = torch.cat([a_feat, v_feat, u_feat], dim=-1)  # [T, 3D]
                _cue = self.model.cue_fuse(fuse_input)   
                text_outs = torch.cat([_cue, sep, mix], dim = 1) # [1, T', D]

            text_out_lens = torch.tensor([text_outs.size(1)], dtype=torch.long, device=text_outs.device) # [1]
            decoded_codec = self.model.decode_codec(
                text_outs,
                text_out_lens,
                max_length=30 * 25,
                sampling=self.sampling,
                beam_size=self.beam_size,
                continual=continual,
            )


        if tgt_scp_path != None:
            tgt_codec = np.load(tgt_scp_path)
            decoded_codec_len = min(decoded_codec.shape[1], tgt_codec.shape[0])

            decoded_codec_first_layer = decoded_codec[0,:decoded_codec_len, 0].detach().cpu()
            decoded_codec_second_layer = decoded_codec[0,:decoded_codec_len, 1].detach().cpu()

            tgt_codec_first_layer = tgt_codec[:decoded_codec_len,0]
            tgt_codec_second_layer = tgt_codec[:decoded_codec_len,1]
            
            audio_codec_first_layer_acc = len(torch.where(tgt_codec_first_layer == decoded_codec_first_layer)[0])/ decoded_codec_len
            audio_codec_second_layer_acc = len(torch.where(tgt_codec_second_layer == decoded_codec_second_layer)[0])/ decoded_codec_len
        if tgt_scp_path != None:
            decoded_codec_gt = torch.tensor(tgt_codec[:100, :2]).unsqueeze(0).to(decoded_codec.device)


        # _, _, gen_speech_only_lm, _ = self.codec_model(
        #     decoded_codec[:, continual_length:], bit_width=None, run_mod="decode"
        # 3. predict embeddings
        #     decoded_codec,
        #     text_outs,
        #     text_out_lens,
        #     self.codec_model,
        if enroll_stage == 'ar_stage_only':
            text_outs_systhesis  = torch.cat([mix], dim = 1) # [1, T', D]
            text_outs_systhesis_lens =  torch.tensor([text_outs_systhesis.size(1)], dtype=torch.long, device=text_outs.device) # [1]
        else:
            text_outs_systhesis  = text_outs
            text_outs_systhesis_lens =  text_out_lens

         
        gen_speech = self.model.syn_audio(
            decoded_codec,
            text_outs_systhesis,
            text_outs_systhesis_lens,
            self.codec_model,
            continual_length=continual_length,
        )

        #     decoded_codec,
        #     text_outs_systhesis,
        #     text_outs_systhesis_lens,
        #     self.codec_model,
       
        # #generate waveform from mel
        # # import pdb;pdb.set_trace()
        # with torch.inference_mode():

        # you can convert the generated waveform to 16 bit linear PCM
        ret_val = dict(
            gen=gen_speech,
        )

        return (
            ret_val,
            decoded_codec,
        )  # {'gen':[1,1,T] }, [1,T,n_q]

