"""把 VoxCeleb2 的 all.scp 路径替换为 AV-HuBERT 连续特征路径。

用法:
    python replace_path_vox2_avhubert_continous_feat.py
    python replace_path_vox2_avhubert_continous_feat.py --input X --output Y --feat-dir Z
"""
import argparse

from paths import VOX2
from scp_utils import read_csv_rows, write_csv


def target_subpath(target_id):
    """根据 target_id 构造 AV-HuBERT 特征子路径。"""
    return target_id[:7] + "/" + target_id[8:8 + 11] + "/" + target_id[20:20 + 5]


def main():
    parser = argparse.ArgumentParser(description="替换 VoxCeleb2 AV-HuBERT 特征路径")
    parser.add_argument("--input", default=VOX2["codec_all_scp_train"])
    parser.add_argument("--output", default=VOX2["avhubert_feat_scp_train"])
    parser.add_argument("--feat-dir", default=VOX2["avhubert_feat_dir_train"])
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
