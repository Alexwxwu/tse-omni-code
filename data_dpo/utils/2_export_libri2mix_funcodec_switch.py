from funcodec.bin.codec_inference import Speech2Token
import torch
import librosa
import tqdm
import numpy as np
import os
import torch.multiprocessing as mp
import argparse
from pathlib import Path

SEED = 1234

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_file', type=str, required=True, 
                   help='Input file with format: train,pretrain,UoBUXOOdLXY/00004,0,...')
    p.add_argument('--switch_audio_dir', type=str, required=True, 
                   help='Input file with format: train,pretrain,UoBUXOOdLXY/00004,0,...')
    p.add_argument('--partition', type=str, required=True, 
                   help='Input file with format: train,pretrain,UoBUXOOdLXY/00004,0,...')
     
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--output_dir', type=str, required=True, 
                   help='Output directory for codec files')
    p.add_argument('--normalize', default=True, action="store_true")
    p.add_argument('--num_proc', type=int, default=4)
    p.add_argument('--gpus', nargs="+", default=["cuda:0"])
    return p.parse_args()

def extract_audio_paths(input_file,switch_audio_dir, partition):
    """从输入文件中提取所有音频路径"""
    import pdb;pdb.set_trace()
    audio_paths = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            # if not line:
            #     continue
            
            # 解析每行，提取所有可能的音频路径
            # parts = line.split(',')
      
            audio_path = switch_audio_dir + partition + '/s2/' + line.replace(',','_').replace('/','_').strip() + '.wav'
            audio_paths.append(audio_path)
    
    return list(set(audio_paths))  # 去重

def process_audio(args, audio_paths):
    """处理所有音频文件"""
    # 初始化模型
    model = Speech2Token(config_file=args.config, model_file=args.model, device='cuda')
    model.eval()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with torch.no_grad():
        # import pdb;pdb.set_trace()
        for audio_path in tqdm.tqdm(audio_paths, desc="Processing audio files"):
            try:
        
                full_audio_path = os.path.join(args.switch_audio_dir, audio_path )
         
                
                # 加载音频
                audio, sr = librosa.load(full_audio_path, sr=16000)
                assert sr == 16000
                
                if args.normalize:
                    audio = audio / np.max(np.abs(audio))
                
                audio = torch.from_numpy(audio).cuda()
                audio = audio.unsqueeze(0).unsqueeze(0)  # [1,1,T]
                
                # 提取codec
                codes = model(audio, run_mod="encode")[0][0].permute(1,2,0).squeeze(0)  # [T, n_q]
                
                # 生成输出路径：output_dir + '_' + 音频路径
                output_filename = f"{audio_path.split('/s2/')[1].split('.wav')[0]}.npy"
                output_path = os.path.join(args.output_dir, output_filename)
                # import pdb;pdb.set_trace()
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 保存codec
                np.save(output_path, codes.cpu().numpy())
                print(f"Saved codec to: {output_path}")
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

def main():
    args = parse_args()
    
    # 提取音频路径
    audio_paths = extract_audio_paths(args.input_file, args.switch_audio_dir, args.partition)
    print(f"Found {len(audio_paths)} unique audio paths")
    
    # 单进程处理所有音频
    process_audio(args, audio_paths)
    
    print("Codec extraction completed!")

if __name__ == "__main__":
    main()