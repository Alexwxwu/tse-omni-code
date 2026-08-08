"""把 LRS3 的 all.scp 路径替换为 FunCodec codec token 路径。

用法:
    python replace_path_lrs3_codec_621.py
    python replace_path_lrs3_codec_621.py --input X --output Y --feat-dir Z
"""
import argparse

from paths import LRS3
from scp_utils import read_csv_rows, write_csv


def target_subpath(target_id):
    """根据 target_id 构造 codec token 子路径。"""
    return target_id[:11] + "/" + target_id[12:12 + 5]


def main():
    parser = argparse.ArgumentParser(description="替换 LRS3 codec token 路径")
    parser.add_argument("--input", default=LRS3["codec_all_scp_test"])
    parser.add_argument("--output", default=LRS3["codec_token_scp_test_avhubert"])
    parser.add_argument("--feat-dir", default=LRS3["codec_token_dir_test_avhubert"])
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
