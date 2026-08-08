"""把 YGD 的 all.scp 路径替换为 VSR（手势）特征路径。

用法:
    python replace_path_ygd_vsr_feat.py --kind test
    python replace_path_ygd_vsr_feat.py --input X --output Y --feat-dir Z
"""
import argparse

from paths import YGD
from scp_utils import read_csv_rows, write_csv


def target_subpath(target_id):
    """根据 target_id 构造手势特征子路径（'$' 前部分 '+' 替换为 '/'）。"""
    return target_id.split("$")[0].replace("+", "/")


def main():
    parser = argparse.ArgumentParser(description="替换 YGD 手势特征路径")
    parser.add_argument("--input", default=YGD["codec_all_scp_test"])
    parser.add_argument("--output", default=YGD["vsr_feat_scp_test"])
    parser.add_argument("--feat-dir", default=YGD["vsr_feat_dir_test"])
    args = parser.parse_args()

    rows = read_csv_rows(args.input, delimiter=";")
    out = []
    for row in rows:
        target_id = row[0].split(" ")[0]
        out.append([target_id, args.feat_dir + target_subpath(target_id) + ".npy"])

    write_csv(args.output, out, delimiter=" ")
    print(f"已写入 {len(out)} 行 -> {args.output}")


if __name__ == "__main__":
    main()
