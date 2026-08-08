"""生成 LibriMix (wesep) test 的 s1 / mix_clean / aux_s1 scp 列表。

用法:
    python get_librimix_wesep_list.py --output s1
    python get_librimix_wesep_list.py --output all
"""
import argparse

from paths import LIBRIMIX
from scp_utils import read_lines, write_scp, scp_line


def build_lines(wav_raw_scp, enroll_raw_scp, enroll_s1_wav, output):
    """根据 wav.scp 和 spk1.enroll 构造 scp 行。

    wav.scp 每行: mix_utt_id mix_wav_path tgt_wav_path
    spk1.enroll 每行: spk_id enroll_path
    """
    lines = read_lines(wav_raw_scp)
    enroll_lines = read_lines(enroll_raw_scp)
    out = []
    for idx, line in enumerate(lines):
        items = line.split(" ")
        mix_utt_id = items[0]
        mix_wav_path = items[1]
        tgt_wav_path = items[2]
        enroll_path = enroll_s1_wav + enroll_lines[idx].split(" ")[1]

        if output in ("s1", "all"):
            out.append(scp_line(mix_utt_id, tgt_wav_path))
        if output in ("mix_clean", "all"):
            out.append(scp_line(mix_utt_id, mix_wav_path))
        if output in ("aux_s1", "all"):
            out.append(scp_line(mix_utt_id, enroll_path))
    return out


def main():
    parser = argparse.ArgumentParser(description="生成 LibriMix wesep test scp 列表")
    parser.add_argument(
        "--output",
        choices=["s1", "mix_clean", "aux_s1", "all"],
        default="s1",
        help="要生成的 scp 类型（默认 s1）",
    )
    args = parser.parse_args()

    out_path = {
        "s1": LIBRIMIX["s1_scp_test"],
        "mix_clean": LIBRIMIX["mix_clean_scp_test"],
        "aux_s1": LIBRIMIX["aux_s1_scp_test"],
    }[args.output]

    lines = build_lines(
        LIBRIMIX["wav_raw_scp_test"],
        LIBRIMIX["enroll_raw_scp_test"],
        LIBRIMIX["enroll_s1_wav"],
        args.output,
    )
    write_scp(out_path, lines)
    print(f"已写入 {len(lines)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
