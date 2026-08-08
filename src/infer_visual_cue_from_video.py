## Inference python scripts
import os
import sys 
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import cv2 as cv
import argparse

import logging
import torch
import torch.multiprocessing as mp
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

from tse_inference_visual import TSExtraction

from utils.utils import get_source_list
from utils.mel_spectrogram import MelSpec

# External SEMO-621 repo (video frontend / perturbation helpers); override with SEMO_DIR.
_SEMO_DIR = os.environ.get('SEMO_DIR', '/mnt/users/hccl.local/wwu/SEMO-621')
sys.path.append(_SEMO_DIR)
from visual_frontend.visual_frontend import VisualFrontend
sys.path.append(os.path.join(_SEMO_DIR, 'data_preparation'))
from Visual_perturb import *

def process_video_with_mask(videoFile, test_ds_rate, params, mask_type, mask_start, mask_length, partition):
    """
    处理带有mask的视频并提取特征
    """
    roiSize = params["roiSize"]
    normMean = params["normMean"]
    normStd = params["normStd"]
    vf = params["vf"]
    device = params["device"]
    
    # 获取遮挡物（如果是部分遮挡）
    occluder_img = None
    alpha_mask = None
    if mask_type == 1:
        obj_dir = '/mnt/users/hccl.local/wwu/SEMO-621/data_preparation/Asset/object_image_sr'
        obj_mask_dir = '/mnt/users/hccl.local/wwu/SEMO-621/data_preparation/Asset/object_mask_x4'
        _, occluder_img, occluder_mask = get_occluders(obj_dir, obj_mask_dir, state=partition)
        alpha_mask = np.expand_dims(occluder_mask, axis=2)
        alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
    
    # 读取视频并应用mask
    captureObj = cv.VideoCapture(videoFile)
    roiSequence = []
    roiSequence_vclean = []
    frame_idx = 0
    
    while captureObj.isOpened():
        ret, frame = captureObj.read()
        if not ret:
            break
        
        # 提取基础ROI
        grayed = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grayed = grayed / 255
        grayed = cv.resize(grayed, (roiSize * 2, roiSize * 2))
        base_roi = grayed[
            int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
        ]
        if mask_type == -1:
            roiSequence_vclean.append(base_roi)
            frame_idx += 1
        # 应用mask
        else:
            masked_roi = apply_mask_to_frame(
                base_roi, test_ds_rate, mask_type, frame, roiSize, partition, 
                mask_start, mask_length, frame_idx, occluder_img, alpha_mask
            )
            
            roiSequence.append(masked_roi)
            frame_idx += 1
    
    captureObj.release()
    
    # 提取视觉特征
    if mask_type!= -1 and len(roiSequence) > 0:
        inp = np.stack(roiSequence, axis=0)
        inp = np.expand_dims(inp, axis=[1, 2])
        inp = (inp - normMean) / normStd
        inputBatch = torch.from_numpy(inp).float().to(device)
        
        vf.eval()
        with torch.no_grad():
            outputBatch = vf(inputBatch)
        out = torch.squeeze(outputBatch, dim=1).to(device)
        return out
    else:

        inp = np.stack(roiSequence_vclean, axis=0)
        inp = np.expand_dims(inp, axis=[1, 2])
        inp = (inp - normMean) / normStd
        inputBatch = torch.from_numpy(inp).float().to(device)
        
        vf.eval()
        with torch.no_grad():
            outputBatch = vf(inputBatch)
        out = torch.squeeze(outputBatch, dim=1).to(device)
        return out

        
        # 保存特征
        # np.save(outputFile, visual_features)

def apply_mask_to_frame(roi, test_ds_rate, mask_type, original_frame, roiSize, partition, mask_start, mask_length, frame_idx, 
                       occluder_img=None, alpha_mask=None):
    """
    应用不同类型的mask到帧上
    
    Args:
        roi: 当前帧的ROI区域
        mask_type: 遮挡类型 (0:full_mask, 1:occluded, 2:low resolution)
        original_frame: 原始帧
        roiSize: ROI尺寸
        partition: 数据集分区 (train/val/test)
        mask_start: mask开始帧
        mask_length: mask持续时间
        frame_idx: 当前帧索引
        occluder_img: 遮挡物图像
        alpha_mask: 遮挡物遮罩
    """
    import random
    import torchvision
    from skimage.util import random_noise
    import sys
    
    # 如果不在mask范围内，直接返回原始ROI
    if frame_idx < mask_start or frame_idx >= mask_start + mask_length:
        return roi
    
    if mask_type == 0:  # full_mask (全遮挡)
        return np.zeros_like(roi)
    
    elif mask_type == 1:  # occluded (部分遮挡)
        if occluder_img is None or alpha_mask is None:
            # 如果没有提供遮挡物，使用默认的全遮挡
            return np.zeros_like(roi)
        
        frame_resized = cv.resize(original_frame, (roiSize * 2, roiSize * 2))
        roi_color = frame_resized[
            int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
        ]
        
        if partition != "test":
            offset_x = random.uniform(13, 17)
            offset_y = random.uniform(13, 17)
            x = random.uniform(roiSize / 2 - 5, roiSize / 2 + 5)
            y = random.uniform(roiSize / 2 - 5, roiSize / 2 + 5)
        else:
            offset_x = 15
            offset_y = 15
            x = roiSize / 2
            y = roiSize / 2
        
        # 应用遮挡
        roi_color = overlay_image_alpha(
            roi_color,
            occluder_img,
            int(x - offset_x),
            int(y - offset_y),
            alpha_mask,
        )
        roi_masked = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
        return roi_masked / 255
    
    elif mask_type == 2:  # low resolution (低分辨率)
        frame_resized = cv.resize(original_frame, (roiSize * 2, roiSize * 2))
        roi_color = frame_resized[
            int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
            int(roiSize - (roiSize / 2)) : int(roiSize + (roiSize / 2)),
        ]
        
        if partition != "test":
            if random.random() < 0.5:
                # 添加高斯噪声
                var = random.uniform(0.02, 0.2)
                roi_color = random_noise(
                    roi_color, mode="gaussian", mean=0, var=var, clip=True
                ) * 255
                roi_color = np.uint8(roi_color)
            else:
                # 高斯模糊
                blur = torchvision.transforms.GaussianBlur(
                    kernel_size=(13, 13), sigma=(4, 8)
                )
                roi_tensor = torch.tensor(roi_color).unsqueeze(0).permute(0, 3, 1, 2)
                roi_blurred = blur(roi_tensor).permute(0, 2, 3, 1).squeeze(0).numpy()
                roi_color = roi_blurred
        else:
            # 测试时使用固定的低分辨率处理
            times = test_ds_rate
            roi_color = cv.resize(roi_color, (roiSize // times, roiSize // times))
            roi_color = cv.resize(roi_color, (roiSize, roiSize))
        
        roi_masked = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)
        return roi_masked / 255
    
    else:
        sys.exit("error: unknown mask type", mask_type)
    
 
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix_wav_scp", type=str, default = None)
    parser.add_argument("--ref_wav_scp", type=str, default = None)
    parser.add_argument("--visual_sync_scp", type=str, default = None)

     
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
    # visual_sync_ids, visual_sync_paths = get_source_list(args.visual_sync_scp, ret_name=True)

    scp_list = [] # [ [mix_wav, ref_wav, ref_codec], [...] ]
    for id in mix_wav_ids:
        mix_wav_path = mix_wav_paths[mix_wav_ids.index(id)]
        ref_wav_path = ref_wav_paths[ref_wav_ids.index(id)]
        scp_list.append([mix_wav_path, ref_wav_path])   
     

    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # load model
    logger.info("loading model")
    tse = TSExtraction(args, args.model_ckpt, device, logger)
    # mel spec
    mel_spec = MelSpec(**args.mel_config)
 
    scp = scp_list[rank::args.num_proc]
    # Inference
    total_rtf = 0.0
    with torch.no_grad(), tqdm.tqdm(scp, desc=f"[inferencing...rank {rank}]") as pbar:
        for paths in pbar:
            mix_wav_path, ref_wav_path = paths
            if 'all_concat' in mix_wav_path:
                lrs3_mp4_base_dir = '/mnt/users/hccl.local/wwu/AVTSE_Momentum/lrs3_very_long_testset/mp4/'
     
                video1_path = lrs3_mp4_base_dir + mix_wav_path.split('/mix/test_test_')[1].split('_0_')[0].replace('_all','/all') + '.mp4'
            else:
                lrs3_mp4_base_dir = '/mnt/Corpus-Upload/lrs3/mp4/'
                video1_path = lrs3_mp4_base_dir + '/test/' + mix_wav_path.split('/mix/test_test_')[1].split('_0_')[0].replace('_','/') + '.mp4'

            # 2. Save audio
            base_name = Path(mix_wav_path).stem + ".wav"
            save_path = args.output_dir / base_name
            #     continue

             
            # 0. Mix Mel -> [1, T,]
            mix_wav, sr = torchaudio.load(mix_wav_path)  # [1,T]
            if limit_infer_length == True:
                mix_wav = mix_wav[:, :int(args.max_infer_length * 16000)]
             
            mask = torch.tensor([mix_wav.size(1)], dtype=torch.long)
            mix_mel, _ = mel_spec.mel(mix_wav, mask)
            mix_mel = mix_mel.to(device)
           

            # 1. Ref Mel -> [1,T,D]
            ref_wav, sr = torchaudio.load(ref_wav_path)  # [1,T]
            if args.max_aux_ds is not None:
                ref_wav = ref_wav[:, -int(args.max_aux_ds * 16000):]
       
            mask = torch.tensor([ref_wav.size(1)], dtype=torch.long)
            ref_mel, _ = mel_spec.mel(ref_wav, mask)
            ref_mel = ref_mel.to(device)
            

                    # 初始化视觉前端模块
            vf = VisualFrontend()
            vf.load_state_dict(torch.load('/mnt/users/hccl.local/wwu/MuSE/pretrain_networks/visual_frontend.pt', 
                                        map_location=device))
            vf.to(device)

            # 设置参数
            params = {
                "roiSize": 112, 
                "normMean": 0.4161, 
                "normStd": 0.1688, 
                "vf": vf,
                "device": device,
                "obj_dir": 1,
                "obj_mask_dir": 1
            }

            test_ds_rate = 25
            mask_start = 99999 
            mask_len = 99999
            mask_type = -1

            visual_sync =  process_video_with_mask(video1_path, test_ds_rate, params, mask_type, mask_start, mask_len, partition='test')
 

            # 1. Inference
            start = time.time()
            if visual_sync.dim() ==2:
                visual_sync = visual_sync.unsqueeze(0).to(device)
            visual_sync = visual_sync[:, :int(args.max_infer_length * 25)]

            start = time.time()
            continual = None #[T,Nq]

            output_tuple = tse(mix_mel, ref_mel, visual_sync, enroll_setting = args.enroll_setting, enroll_length = args.max_aux_ds, enroll_stage = args.enroll_stage, tgt_scp_path=None) 

            if output_tuple[0] == 0:
                print("none!!!")
                continue
            output = output_tuple[0]["gen"].squeeze()  # [T]
            # 1. Inference
             
            # # import pdb;pdb.set_trace()
             
            #     print("none!!!")
            #     continue
             
            rtf = (time.time() - start) / (len(output) / sr)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # 2. Save audio
            base_name = Path(mix_wav_path).stem + ".wav"
            save_path = args.output_dir / base_name
            sf.write( save_path, normalize(output.cpu().numpy(), mix_wav.numpy().squeeze()),samplerate=sr, )
    logger.info(
        f"Finished generation of {len(scp)} utterances (RTF = {total_rtf / len(scp):.03f})."
    )

def normalize(output: np.ndarray, mixture: np.ndarray):
    norm = np.linalg.norm(mixture, np.inf)
    return output * norm / np.max(np.abs(output))

if __name__ == "__main__":
    args = parse_args()
    main(args)
