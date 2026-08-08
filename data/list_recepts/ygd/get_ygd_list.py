"""生成 YGD 的 s1 / mix_clean scp 列表。

用法:
    python get_ygd_list_621.py --split train --output s1
    python get_ygd_list_621.py --split train --output mix_clean
"""
import argparse

from paths import YGD
from scp_utils import read_csv_rows, write_scp, scp_line


def build_lines(mix_csv, mix_dir, s1_dir, split, output):
    """根据 mixture csv 构造 s1 / mix_clean scp 行（按 split 过滤）。"""
    rows = read_csv_rows(mix_csv)
    out = []
    for items in rows:
        if items[0] != split:
            continue
        mix_utt_id = items[2] + "+" + items[3] + "$" + items[6] + "+" + items[7]
        mix_utt_id_base_name = "_".join(items).replace("/", "_")
        s1_utt_id_base_name = "/".join(items[2:4])
        if output in ("s1", "all"):
            out.append(scp_line(mix_utt_id, s1_dir + s1_utt_id_base_name + ".wav"))
        if output in ("mix_clean", "all"):
            out.append(scp_line(mix_utt_id, mix_dir + mix_utt_id_base_name + ".wav"))
    return out


def main():
    parser = argparse.ArgumentParser(description="生成 YGD scp 列表")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", choices=["s1", "mix_clean", "all"], default="s1")
    args = parser.parse_args()

    mix_dir = YGD["mix_dir"] + args.split + "/"
    s1_dir = YGD["s1_dir"] + args.split + "/"
    out_path = YGD["s1_scp_train"] if args.output == "s1" else YGD["mix_clean_scp_train"]

    lines = build_lines(YGD["mix_csv"], mix_dir, s1_dir, args.split, args.output)
    write_scp(out_path, lines)
    print(f"已写入 {len(lines)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
