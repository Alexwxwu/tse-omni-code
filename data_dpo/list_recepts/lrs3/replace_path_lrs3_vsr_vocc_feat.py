"""把 LRS3 遮挡（vocc）scp 路径替换为遮挡 VSR 特征路径。

用法:
    python replace_path_lrs3_vsr_vocc_feat.py
    python replace_path_lrs3_vsr_vocc_feat.py --input X --output Y --feat-dir Z
"""
import argparse

from paths import LRS3
from scp_utils import read_csv_rows, write_csv


def target_subpath(target_id):
    """根据 target_id 构造遮挡 VSR 特征子路径。"""
    return target_id[:11] + "/" + target_id[12:12 + 5]


def main():
    parser = argparse.ArgumentParser(description="替换 LRS3 遮挡 VSR 特征路径")
    parser.add_argument("--input", default=LRS3["mix_clean_vocc_scp_test"])
    parser.add_argument("--output", default=LRS3["vsr_feat_vocc_scp_test"])
    parser.add_argument("--feat-dir", default=LRS3["vsr_feat_dir_vocc_test"])
    args = parser.parse_args()

    rows = read_csv_rows(args.input, delimiter=";")
    out = []
    for row in rows:
        target_id = row[0].split(" ")[0]
        target_path = row[0].split(" ")[1]
        occ_info = target_path.split("_")[-6:-3]
        sub = target_subpath(target_id)
        new_path = (
            args.feat_dir + sub
            + "_mask" + occ_info[-1]
            + "_start" + occ_info[0]
            + "_len" + occ_info[1] + ".npy"
        )
        out.append([target_id, new_path])

    write_csv(args.output, out, delimiter=" ")
    print(f"已写入 {len(out)} 行 -> {args.output}")


if __name__ == "__main__":
    main()
