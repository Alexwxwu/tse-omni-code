"""生成 LibriMix (wesep) train-100 的 s1 / mix_clean / aux_s1 scp 列表。

用法:
    python get_librimix_wesep_list_train-100.py --output mix_clean
    python get_librimix_wesep_list_train-100.py --output all
"""
import argparse
import json
import random

from paths import LIBRIMIX
from scp_utils import read_lines, write_scp, scp_line


def build_lines(wav_raw_scp, enroll_json, output):
    """根据 wav.scp 和 spk2enroll.json 构造 scp 行。

    wav.scp 每行: mix_utt_id mix_wav_path tgt_wav_path
    spk2enroll.json: {spk_id: [[_, enroll_path], ...]}
    """
    lines = read_lines(wav_raw_scp)
    with open(enroll_json, encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for line in lines:
        items = line.split(" ")
        mix_utt_id = items[0]
        mix_wav_path = items[1]
        tgt_wav_path = items[2]
        tgt_spk_id = items[0].split("-")[0]
        enroll_path = random.choice(data[tgt_spk_id])[1]

        if output in ("s1", "all"):
            out.append(scp_line(mix_utt_id, tgt_wav_path))
        if output in ("mix_clean", "all"):
            out.append(scp_line(mix_utt_id, mix_wav_path))
        if output in ("aux_s1", "all"):
            out.append(scp_line(mix_utt_id, enroll_path))
    return out


def main():
    parser = argparse.ArgumentParser(description="生成 LibriMix wesep train-100 scp 列表")
    parser.add_argument(
        "--output",
        choices=["s1", "mix_clean", "aux_s1", "all"],
        default="mix_clean",
        help="要生成的 scp 类型（默认 mix_clean）",
    )
    args = parser.parse_args()

    out_path = {
        "s1": LIBRIMIX["s1_scp_train100"],
        "mix_clean": LIBRIMIX["mix_clean_scp_train100"],
        "aux_s1": LIBRIMIX["aux_s1_scp_train100"],
    }[args.output]

    lines = build_lines(
        LIBRIMIX["wav_raw_scp_train100"],
        LIBRIMIX["enroll_json_train100"],
        args.output,
    )
    write_scp(out_path, lines)
    print(f"已写入 {len(lines)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
