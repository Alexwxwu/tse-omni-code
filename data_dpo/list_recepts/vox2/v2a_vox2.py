"""把 VoxCeleb2 视频批量转换为音频。

用法:
    python v2a_vox2.py --all-list X --video-dir Y --audio-dir Z
"""
import argparse
import os

from scp_utils import read_lines, ensure_audio_from_video


def main():
    parser = argparse.ArgumentParser(description="把 VoxCeleb2 视频批量转换为音频")
    parser.add_argument("--all-list", required=True, help="file.list，每行形如 spk/xxx/yyy")
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--audio-dir", required=True)
    args = parser.parse_args()

    all_lines = read_lines(args.all_list)
    for aux_line in all_lines:
        aux_item = aux_line.split("/")
        if aux_item[1] == "dev":
            sub = os.path.join("train", *aux_item[2:])
        else:
            sub = os.path.join("test", *aux_item[2:])
        video_path = os.path.join(args.video_dir, sub) + ".mp4"
        audio_path = os.path.join(args.audio_dir, sub) + ".wav"
        ensure_audio_from_video(audio_path, video_path)
    print("转换完成")


if __name__ == "__main__":
    main()
