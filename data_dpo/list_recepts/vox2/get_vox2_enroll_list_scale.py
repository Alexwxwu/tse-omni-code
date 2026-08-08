"""生成 VoxCeleb2 的 aux_s1（注册）scp 列表，必要时用 ffmpeg 从视频提取音频。

用法:
    python get_vox2_enroll_list_621.py --split train_scale
    python get_vox2_enroll_list_621.py --split test_scale
"""
import argparse
import os

from paths import VOX2
from scp_utils import read_lines, read_csv_rows, write_scp, scp_line, ensure_audio_from_video

# 各 split 对应的输入/输出路径
SPLITS = {
    "train_scale": {
        "mix_csv": "mix_csv_train_scale",
        "file_list": "file_list",
        "video_dir": "aux_video_dir",
        "audio_dir": "aux_audio_dir",
        "aux_s1_scp": "aux_s1_scp_train_scale",
    },
    "test_scale": {
        "mix_csv": "mix_csv_test_scale",
        "file_list": "file_list",
        "video_dir": "aux_video_dir",
        "audio_dir": "aux_audio_dir",
        "aux_s1_scp": "aux_s1_scp_test_scale",
    },
}


def build_lines(mix_csv, all_list, video_dir, audio_dir):
    """在 file.list 中为每个 mixture 行匹配辅助说话人，并确保其音频存在。"""
    rows = read_csv_rows(mix_csv)
    all_lines = read_lines(all_list)
    out = []
    for items in rows:
        aux_id = items[2]
        for aux_line in all_lines:
            aux_item = aux_line.split("/")
            if aux_item[2] == aux_id:
                if aux_item[1] == "dev":
                    sub = os.path.join("dev", *aux_item[2:])
                else:
                    sub = os.path.join("test", *aux_item[2:])
                video_path = os.path.join(video_dir, sub) + ".mp4"
                audio_path = os.path.join(audio_dir, sub) + ".wav"
                ensure_audio_from_video(audio_path, video_path)
                utt_id = "_".join(items[2:4] + items[6:8]).replace("/", "_")
                out.append(scp_line(utt_id, audio_path))
                break
    return out


def main():
    parser = argparse.ArgumentParser(description="生成 VoxCeleb2 aux_s1 注册 scp 列表")
    parser.add_argument("--split", choices=list(SPLITS), default="test_scale")
    args = parser.parse_args()

    cfg = SPLITS[args.split]
    lines = build_lines(
        VOX2[cfg["mix_csv"]],
        VOX2[cfg["file_list"]],
        VOX2[cfg["video_dir"]],
        VOX2[cfg["audio_dir"]],
    )
    write_scp(VOX2[cfg["aux_s1_scp"]], lines)
    print(f"已写入 {len(lines)} 行 -> {VOX2[cfg['aux_s1_scp']]}")


if __name__ == "__main__":
    main()
