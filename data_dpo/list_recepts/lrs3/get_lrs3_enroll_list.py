"""生成 LRS3 的 aux_s1（注册）scp 列表，必要时用 ffmpeg 从视频提取音频。

用法:
    python get_lrs3_enroll_list_621.py
"""
import argparse
import os

from paths import LRS3
from scp_utils import read_lines, read_csv_rows, write_scp, scp_line, ensure_audio_from_video


def build_lines(mix_csv, all_list, video_dir, audio_dir):
    """在 all_list 中为每个 mixture 行匹配辅助说话人，并确保其音频存在。"""
    rows = read_csv_rows(mix_csv)
    all_lines = read_lines(all_list)
    out = []
    for items in rows:
        aux_id = items[2].split("/")[0]
        for aux_line in all_lines:
            aux_item = aux_line.split("/")
            aux_item[-1] = aux_item[-1].split(".wav")[0]
            if aux_item[-2] == aux_id:
                if aux_item[-3] == "pretrain":
                    sub = os.path.join("pretrain", *aux_item[-2:])
                else:
                    sub = os.path.join("test", *aux_item[-2:])
                video_path = os.path.join(video_dir, sub) + ".mp4"
                audio_path = os.path.join(audio_dir, sub) + ".wav"
                ensure_audio_from_video(audio_path, video_path)
                utt_id = "_".join(items[2:3] + items[5:6]).replace("/", "_")
                out.append(scp_line(utt_id, audio_path))
                break
    return out


def main():
    parser = argparse.ArgumentParser(description="生成 LRS3 aux_s1 注册 scp 列表")
    parser.add_argument("--output", default=LRS3["aux_s1_scp_test_verylong"])
    args = parser.parse_args()

    lines = build_lines(
        LRS3["mix_csv_test_verylong"],
        LRS3["all_list"],
        LRS3["aux_video_dir"],
        LRS3["aux_audio_dir"],
    )
    write_scp(args.output, lines)
    print(f"已写入 {len(lines)} 行 -> {args.output}")


if __name__ == "__main__":
    main()
