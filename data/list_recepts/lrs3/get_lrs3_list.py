"""生成 LRS3 的 s1 / mix_clean scp 列表。

用法:
    python get_lrs3_list_621.py --output s1
    python get_lrs3_list_621.py --output mix_clean
"""
import argparse

from paths import LRS3
from scp_utils import read_csv_rows, write_scp, scp_line, join_id


def build_lines(mix_csv, mix_dir, s1_dir, output):
    """根据 mixture csv 构造 s1 / mix_clean scp 行。"""
    rows = read_csv_rows(mix_csv)
    out = []
    for items in rows:
        mix_utt_id = join_id(
            items[2].split("/")[0], items[2].split("/")[1],
            items[5].split("/")[0], items[5].split("/")[1],
        )
        path = "_".join(items).replace("/", "_")
        if output in ("s1", "all"):
            out.append(scp_line(mix_utt_id, s1_dir + path + ".wav"))
        if output in ("mix_clean", "all"):
            out.append(scp_line(mix_utt_id, mix_dir + path + ".wav"))
    return out


def main():
    parser = argparse.ArgumentParser(description="生成 LRS3 scp 列表")
    parser.add_argument(
        "--output",
        choices=["s1", "mix_clean", "all"],
        default="s1",
        help="要生成的 scp 类型（默认 s1）",
    )
    args = parser.parse_args()

    out_path = {
        "s1": LRS3["s1_scp_test_verylong"],
        "mix_clean": LRS3["mix_clean_scp_test_verylong"],
    }[args.output]

    lines = build_lines(
        LRS3["mix_csv_test_verylong"],
        LRS3["mix_dir_test_verylong"],
        LRS3["s1_dir_test_verylong"],
        args.output,
    )
    write_scp(out_path, lines)
    print(f"已写入 {len(lines)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
