"""把 LRS3 的 all.scp 路径替换为 VSR 连续特征路径。

用法:
    python replace_path_lrs3_vsr_feat.py --kind verylong
    python replace_path_lrs3_vsr_feat.py --input X --output Y --feat-dir Z --kind test
"""
import argparse

from paths import LRS3
from scp_utils import read_csv_rows, write_csv


def target_subpath(target_id, kind):
    """根据 target_id 和数据集类型构造 VSR 特征子路径。"""
    if kind == "verylong":
        return target_id[:11] + "/" + target_id[12:22]
    return target_id[:11] + "/" + target_id[12:12 + 5]


def main():
    parser = argparse.ArgumentParser(description="替换 LRS3 VSR 特征路径")
    parser.add_argument("--input", default=LRS3["codec_all_scp_test_verylong"])
    parser.add_argument("--output", default=LRS3["vsr_feat_scp_test_verylong"])
    parser.add_argument("--feat-dir", default=LRS3["vsr_feat_dir_verylong"])
    parser.add_argument("--kind", choices=["test", "train", "verylong"], default="verylong")
    args = parser.parse_args()

    rows = read_csv_rows(args.input, delimiter=";")
    out = []
    for row in rows:
        target_id = row[0].split(" ")[0]
        out.append([target_id, args.feat_dir + target_subpath(target_id, args.kind) + ".npy"])

    write_csv(args.output, out, delimiter=" ")
    print(f"已写入 {len(out)} 行 -> {args.output}")


if __name__ == "__main__":
    main()
