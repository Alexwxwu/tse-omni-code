"""检查 scp 文件中的音频文件是否有效，必要时用 ffmpeg 从视频提取音频。

用法:
    python check_audio.py --scp X
    python check_audio.py --scp X --video-suffix .mp4
"""
import argparse
import os
import subprocess

import soundfile as sf
import torchaudio


def check_audio_file(filepath):
    """检查音频文件是否有效，返回 (是否有效, 说明)。"""
    if not os.path.isfile(filepath):
        return False, "文件不存在"
    if os.path.getsize(filepath) == 0:
        return False, "文件为空"
    try:
        info = sf.info(filepath)
        if info.frames == 0:
            return False, "音频为空（无音频帧）"
    except RuntimeError as e:
        return False, f"格式错误：{e}"
    return True, "文件正常"


def convert_video_to_audio(video_path, audio_path):
    """使用 ffmpeg 从视频提取音频。"""
    command = f"ffmpeg -y -i {video_path} -ac 1 -ar 16000 {audio_path}"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"ffmpeg转换失败: {result.stderr.decode()}")
        return False
    print(f"视频转换为音频成功：{audio_path}")
    return True


def load_audio_with_check(audio_path, video_path):
    """检查并加载音频，必要时先从视频转换。"""
    valid, msg = check_audio_file(audio_path)
    if not valid:
        print(f"检测未通过：{msg},尝试用ffmpeg转换视频为音频。")
        if not convert_video_to_audio(video_path, audio_path):
            raise RuntimeError(f"无法转换视频：{video_path}")
        valid, msg = check_audio_file(audio_path)
        if not valid:
            raise RuntimeError(f"转换后音频检测失败：{msg}")
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate


def main():
    parser = argparse.ArgumentParser(description="检查 scp 中的音频文件")
    parser.add_argument("--scp", required=True)
    parser.add_argument("--video-suffix", default=".mp4", help="视频文件后缀")
    args = parser.parse_args()

    with open(args.scp) as f:
        lines = f.readlines()
    for count, line in enumerate(lines):
        audio_path = line.split(" ")[1].split("\n")[0]
        video_path = (
            audio_path.split("/wav")[0]
            + "/mp4"
            + audio_path.split("/wav")[1].replace(".wav", args.video_suffix)
        )
        try:
            waveform, sr = load_audio_with_check(audio_path, video_path)
            print(f"[{count}] 加载成功，采样率：{sr}")
        except RuntimeError as e:
            print(f"[{count}] 错误：{e}")


if __name__ == "__main__":
    main()
