## Inference python scripts
import os
import sys 
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse
import logging
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import tqdm
import time
import numpy as np
from pathlib import Path

import torchaudio
import soundfile as sf
from utils.utils import AttrDict, update_args, setup_seed
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin'))
# Base repo path for auxiliary data lookups (override with LAURA_CODE_ROOT).
CODE_ROOT = os.environ.get('LAURA_CODE_ROOT', '/mnt/users/hccl.local/wwu/lauraTSE_code_refact')

from tse_inference_trimodal_gesture import TSExtraction

from utils.utils import get_source_list
from utils.mel_spectrogram import MelSpec

MEL_len = 63
gesture_FPS = 15
lip_FPS = 25 
aux_upsample_factor = MEL_len/lip_FPS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix_wav_scp", type=str, default = None)
    parser.add_argument("--ref_wav_scp", type=str, default = None)
    parser.add_argument("--visual_sync_scp", type=str, default = None)
    parser.add_argument("--text_direc", type=str, default = None)
     

    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--config", type=str)
    parser.add_argument("--model_ckpt", type=str)

    parser.add_argument('--codec_model_file', type=str, required=True)
    parser.add_argument('--codec_config_file', type=str, required=True)

    parser.add_argument("--sampling", default=25, type=int)
    parser.add_argument("--beam_size", default=1, type=int)
    
    parser.add_argument("--max_aux_ds", default=2, type=int)
    parser.add_argument("--enroll_setting", default= 'visual_enroll', type=str)
    parser.add_argument("--max_infer_length", default= 4, type=int)
    parser.add_argument("--limit_infer_length", default=False, type=bool)
    parser.add_argument("--enroll_stage", default='ar_stage_only', type=str)
    parser.add_argument("--Laura", default=1, type=str)

    ## DDP
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument(
        "--gpus", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    )
    parser.add_argument("--model_name", type=str, default=None,
                        help="model name in MODEL_REGISTRY (src/model_registry.py); "
                             "overrides env LAURA_MODEL_NAME / default")
    args = parser.parse_args()
    return args

def main(args):
    print(args)
    args.init_param = [f"{args.codec_model_file}:quantizer.rq.model:quantizer_codebook"]
    os.makedirs(args.output_dir, exist_ok=True)
    setup_seed(1234, 0)
    # mp.spawn(inference, args=(args,), nprocs=args.num_proc, join=True)
    inference(rank=0, args=args)
    print("done!")

 
def pick_random_span(self, s: str, k_min: int = 3, k_max: int = 6) -> str:
            words = s.strip().split()
            n = len(words)
            if n == 0:
                return ""
            # 句子很短，直接返回全部
            if n <= k_min:
                return " ".join(words)
            # 实际可用的最大长度不超过句长
            k_max_eff = min(k_max, n)
            k_min_eff = min(k_min, k_max_eff)
            # 随机选择一个长度
            import random
            k = random.randint(k_min_eff, k_max_eff)
            # 起始下标随机，但要保证 end 不越界
            max_start = n - k
            start = random.randint(0, max_start)
            end = start + k
            return " ".join(words[start:end])

def inference(rank, args):
    # update args to contain config
    update_args(args, args.config)
    args = AttrDict(**vars(args))
    args.output_dir = Path(args.output_dir)
    limit_infer_length = args.limit_infer_length
    limit_infer_length = True
    
    print(f"args: {args}")
    args.max_aux_ds = 2
    # device setup
     
    device = args.gpus[rank % len(args.gpus)]
    # data for each process setup

    mix_wav_ids, mix_wav_paths = get_source_list(args.mix_wav_scp, ret_name=True)
    ref_wav_ids, ref_wav_paths = get_source_list(args.ref_wav_scp, ret_name=True)
    visual_sync_ids, visual_sync_paths = get_source_list(args.visual_sync_scp, ret_name=True)
    if ('vox2_en' in args.mix_wav_scp) or ('lrs3_en' in args.mix_wav_scp) or ('ygd_en' in args.mix_wav_scp):
        mix_wav_dict = {id: path for id, path in zip(mix_wav_ids, mix_wav_paths)}
        ref_wav_dict = {id: path for id, path in zip(ref_wav_ids, ref_wav_paths)}
        visual_sync_dict = {id: path for id, path in zip(visual_sync_ids, visual_sync_paths)}

        text_direc = args.text_direc

        scp_list = []
        for id in mix_wav_ids:
            mix_wav_path = mix_wav_dict.get(id)
            ref_wav_path = ref_wav_dict.get(id)
            visual_sync_path = visual_sync_dict.get(id)
            if ('vox2_en' in args.mix_wav_scp) or ('lrs3_en' in args.mix_wav_scp):
                text_path = text_direc + mix_wav_path.split('/')[-3] + '/s1/' + mix_wav_path.split('/')[-1].replace('.wav','.txt') 
            else:
                text_path = text_direc + '/'.join(visual_sync_path.split('/')[-3:]).replace('.npy','.txt') 
            # with open(text_path, 'r', encoding='utf-8') as file:
                # 从整句里截取一个中间的 3 词 span
             # 从整句里随机截取一个长度在 [3,6] 的连续词 span
            # transcripts_enrolls.append(full_text)

            if mix_wav_path and ref_wav_path and visual_sync_path and text_path:
                scp_list.append([mix_wav_path, ref_wav_path, visual_sync_path,text_path])
            else:
                print(f"ID {id} not found in mix_wav_ids or ref_wav_ids.")

    else:
        scp_list = [] # [ [mix_wav, ref_wav, ref_codec], [...] ]
        for id in mix_wav_ids:
            mix_wav_path = mix_wav_paths[mix_wav_ids.index(id)]
            ref_wav_path = ref_wav_paths[ref_wav_ids.index(id)]
            scp_list.append([mix_wav_path, ref_wav_path])     

    scp = scp_list[rank::args.num_proc]
    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # load model
    logger.info("loading model")
    
    tse = TSExtraction(args, args.model_ckpt, device, logger)
    # mel spec
    mel_spec = MelSpec(**args.mel_config)
 

    # Inference
    total_rtf = 0.0
    with torch.no_grad(), tqdm.tqdm(scp, desc=f"[inferencing...rank {rank}]") as pbar:
        for paths in pbar:
            mix_wav_path, ref_wav_path, visual_sync_path, text_path = paths
            # 2. Save audio
            base_name = Path(mix_wav_path).stem + ".wav"
            save_path = args.output_dir / base_name
            if os.path.exists(save_path):
                continue

            # 0. Mix Mel -> [1, T,]
            audio, sr = torchaudio.load(mix_wav_path)  # [1,T]
            if limit_infer_length == True:
                audio = audio[:, :int(args.max_infer_length * 16000)]
            mask = torch.tensor([audio.size(1)], dtype=torch.long)
            mix_mel, _ = mel_spec.mel(audio, mask)
            mix_mel = mix_mel.to(device)
        
            # 1. Ref Mel -> [1,T,D]
            audio, sr = torchaudio.load(ref_wav_path)  # [1,T]
            if args.max_aux_ds is not None:
                audio = audio[:, -int(args.max_aux_ds * 16000):]
    
            mask = torch.tensor([audio.size(1)], dtype=torch.long)
            ref_mel, _ = mel_spec.mel(audio, mask)
            ref_mel = ref_mel.to(device)

            with open(text_path, 'r', encoding='utf-8') as file:
                full_text = file.readlines()[0].split('\n')[0]
            transcript_aux = full_text
            if 'avhubert' in args.visual_sync_scp:
                visual_sync = np.load(visual_sync_path)
                visual_sync =  torch.from_numpy(visual_sync).to(torch.long)# [T,N]
            
            else:
                if  ('YGD'in visual_sync_path) or ('ygd_en'in visual_sync_path):
                    lip_setting = False
                else:
                    lip_setting = True
                if lip_setting:
                    visual_sync = np.load(visual_sync_path)
                    visual_sync =  torch.from_numpy(visual_sync).to(torch.float)# [T,N]
                    visual_sync = visual_sync.unsqueeze(0).to(device)
            
                else:
                    gesture_upsample_factor = lip_FPS/gesture_FPS
                    visual_sync = np.load(visual_sync_path)
                    visual_sync = visual_sync.reshape(visual_sync.shape[0], 30)
                     
                    visual_sync =  torch.from_numpy(visual_sync).to(torch.float)# [T,N]
                    visual_sync = visual_sync.unsqueeze(0).to(device)
                    visual_sync = visual_sync.permute(0,2,1)
                    visual_sync = F.interpolate(visual_sync, int((gesture_upsample_factor * visual_sync.shape[-1])), mode='linear').to(device)
                    visual_sync = F.pad(visual_sync, (0, int(args.max_infer_length * lip_FPS) - visual_sync.shape[-1])).to(device)
                    visual_sync = visual_sync.permute(0,2,1)
                    
                    # visual_sync[:,50:] = 0

            # 1. Inference
            start = time.time()
            if ('Voxceleb2'in mix_wav_path) or ('vox2'in mix_wav_path):
                tgt_scp_dir = f'{CODE_ROOT}/data_dpo/funcodec/vox2_en/test_scale/0/'
                tgt_scp_path = tgt_scp_dir + mix_wav_path.split('/')[-1].replace('.wav','.npy')
            elif ('YGD'in mix_wav_path) or ('ygd_en'in mix_wav_path):
                tgt_scp_dir = f'{CODE_ROOT}/data_dpo/funcodec/ygd_en/test/0/'
                tgt_scp_path = tgt_scp_dir + visual_sync_path.split('/')[-1].replace('.wav','.npy')
            else:
                tgt_scp_dir = f'{CODE_ROOT}/data_dpo/funcodec/lrs3_en_new/test/0/'
                tgt_scp_path = tgt_scp_dir + mix_wav_path.split('/')[-1].replace('.wav','.npy')
            output_tuple = tse(mix_mel, ref_mel, visual_sync, transcript_aux, enroll_setting = args.enroll_setting, enroll_length = args.max_aux_ds, enroll_stage = args.enroll_stage,tgt_scp_path=None)
            if output_tuple[0] == 0:
                print("none!!!")
                continue
            output = output_tuple[0]["gen"].squeeze()  # [T]
             
            rtf = (time.time() - start) / (len(output) / sr)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # 2. Save audio
            base_name = Path(mix_wav_path).stem + ".wav"
            save_path = args.output_dir / base_name

            if isinstance(output, torch.Tensor):
                out_np = output.detach().cpu().numpy()
            else:
                out_np = output

            if isinstance(audio, torch.Tensor):
                audio_np = audio.detach().cpu().numpy()
            else:
                audio_np = audio

            sf.write(save_path,normalize(out_np, audio_np.squeeze()),samplerate=sr,)
            
            # sf.write(
            #     save_path,
            #     normalize(output.cpu().numpy(), audio.numpy().squeeze()),
    logger.info(
        f"Finished generation of {len(scp)} utterances (RTF = {total_rtf / len(scp):.03f})."
    )

def normalize(output: np.ndarray, mixture: np.ndarray):
    norm = np.linalg.norm(mixture, np.inf)
    return output * norm / np.max(np.abs(output))

if __name__ == "__main__":
    args = parse_args()
    main(args)
