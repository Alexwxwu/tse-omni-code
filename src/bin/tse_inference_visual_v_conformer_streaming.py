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
    

    total_params = 0
    # 4. 计算总参数的大小以MB为单位，按模块父名称分类
    param_sizes = {}
    for name, param in model.named_parameters():
        param_size = param.numel() * param.element_size()  # bytes
        total_params += param_size
        
        # 解析模块名称
        parent_name = name.split('.')[0]  # 获取父模块名
        if parent_name not in param_sizes:
            param_sizes[parent_name] = 0
        param_sizes[parent_name] += param_size
        
    total_params_mb = total_params / (1024 ** 2)  # 转换为MB
    print(f"Total model parameters: {total_params_mb:.2f} MB")

    # 按父名称打印各模块的参数大小
    for parent_name, size in param_sizes.items():
        print(f"Parameters in '{parent_name}': {size / (1024 ** 2):.2f} MB")

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

    @torch.no_grad()
    def __call__(self, mix_mel:torch.Tensor, ref_mel:torch.Tensor, visual_sync:torch.Tensor, enroll_setting: str, enroll_length:int, enroll_stage: str, continual:list, tgt_scp_path:None):
        """
        This function can also be used as TSE Inference.
        mix_mel the mep spec of the mixture: [1, T, D]
        ref_mel is the reference mel : [1, T, D]
        """


        continual_length = None if continual is None else len(continual)
         
        # 1. Encode mix mel and ref mel
        mix_mel_lens = torch.tensor([mix_mel.size(1)], dtype=torch.long, device=mix_mel.device) # [1]
        aux_mel_lens = torch.tensor([ref_mel.size(1)], dtype=torch.long, device=ref_mel.device) # [1]
        visual_sync_lens =  torch.tensor([visual_sync.size(1)], dtype=torch.long, device=ref_mel.device) # [1]
        visual_sync = visual_sync.float().to(ref_mel.device)
        visual_aux = visual_sync[:, :int(enroll_length *25)].float().to(ref_mel.device)

        
        mix, _ = self.model.encode(mix_mel, mix_mel_lens) # [1,T,D]
        aux, aux_lengths = self.model.encode(ref_mel, aux_mel_lens) # [1,T,D]
 
        visual_aux = visual_aux.permute(0,2,1)

        if visual_aux.shape[-1]< aux_lengths.max():
            visual_aux = F.interpolate(visual_aux, int((2.52 * visual_aux.shape[-1])), mode='linear').to(mix.device)
            visual_aux = F.pad(visual_aux, (0, aux_lengths.max() - visual_aux.shape[-1])).to(mix.device)
        else:
            visual_aux = visual_aux[:,:,:aux_lengths.max()]

        visual_aux = visual_aux.permute(0,2,1)
        visual_aux = self.model.visual_aux_encode(visual_aux, aux_lengths)


        visual_sync = self.model.visual_sync_encode(visual_sync)
        sep = self.model.lm_embedding(torch.tensor([[self.model.sep]], dtype = torch.int64, device = mix_mel.device)) # [1,1,D]

         
        if enroll_setting == 'visual_enroll':
            text_outs = torch.cat([visual_aux, sep, mix], dim = 1) # [1, T', D]
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
            text_outs = torch.cat([aux, sep, mix], dim = 1) # [1, T', D]
            text_out_lens = torch.tensor([text_outs.size(1)], dtype=torch.long, device=text_outs.device) # [1]
            decoded_codec = self.model.decode_codec(
                text_outs,
                text_out_lens,
                max_length=30 * 25,
                sampling=self.sampling,
                beam_size=self.beam_size,
                continual=continual,
            )
        if tgt_scp_path:
            tgt_codec = np.load(tgt_scp_path)
            decoded_codec_len = min(decoded_codec.shape[1], tgt_codec.shape[0])

            decoded_codec_first_layer = decoded_codec[0,:decoded_codec_len, 0].detach().cpu()
            decoded_codec_second_layer = decoded_codec[0,:decoded_codec_len, 1].detach().cpu()

            tgt_codec_first_layer = tgt_codec[:decoded_codec_len,0]
            tgt_codec_second_layer = tgt_codec[:decoded_codec_len,1]
            
            audio_codec_first_layer_acc = len(torch.where(tgt_codec_first_layer == decoded_codec_first_layer)[0])/ decoded_codec_len
            audio_codec_second_layer_acc = len(torch.where(tgt_codec_second_layer == decoded_codec_second_layer)[0])/ decoded_codec_len


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
        ret_val = dict(
            gen=gen_speech,
        )
        if tgt_scp_path:

            return (
                ret_val,
                decoded_codec,
                audio_codec_first_layer_acc,
                audio_codec_second_layer_acc
            )  # {'gen':[1,1,T] }, [1,T,n_q]
        else:

             return (
                ret_val,
                decoded_codec,
            )  # {'gen':


