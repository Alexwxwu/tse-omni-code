"""生成 VoxCeleb2（legacy 路径）的 aux_s1 注册 scp 列表。

与 get_vox2_enroll_list_621.py 逻辑相同，但使用旧集群路径，
且 dev 目录映射为 train。支持 --skip 跳过前 N 行。

用法:
    python get_vox2_enroll_list.py --skip 16495
"""
import argparse
import os

from paths import VOX2
from scp_utils import read_lines, read_csv_rows, write_scp, scp_line, ensure_audio_from_video


def build_lines(mix_csv, all_list, video_dir, audio_dir, skip=0):
    """在 file.list 中为每个 mixture 行匹配辅助说话人，并确保其音频存在。"""
    rows = read_csv_rows(mix_csv)
    all_lines = read_lines(all_list)
    out = []
    for idx, items in enumerate(rows):
        if idx < skip:
            continue
        aux_id = items[2]
        for aux_line in all_lines:
            aux_item = aux_line.split("/")
            if aux_item[2] == aux_id:
                if aux_item[1] == "dev":
                    sub = os.path.join("train", *aux_item[2:])
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
    parser = argparse.ArgumentParser(description="生成 VoxCeleb2 (legacy) aux_s1 注册 scp 列表")
    parser.add_argument("--skip", type=int, default=0, help="跳过前 N 行")
    args = parser.parse_args()

    lines = build_lines(
        VOX2["mix_csv_train"],
        VOX2["file_list_legacy"],
        VOX2["aux_video_dir_legacy"],
        VOX2["aux_audio_dir_legacy"],
        args.skip,
    )
    write_scp(VOX2["aux_s1_scp_train"], lines)
    print(f"已写入 {len(lines)} 行 -> {VOX2['aux_s1_scp_train']}")


if __name__ == "__main__":
    main()
