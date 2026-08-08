"""生成 VoxCeleb2 遮挡（vocc）变体的 s1 / mix_clean scp 列表。

用法:
    python get_vox2_vocc_list_621.py --split train --output mix_clean
    python get_vox2_vocc_list_621.py --split test_scale --output mix_clean
"""
import argparse

from paths import VOX2
from scp_utils import read_csv_rows, write_scp, scp_line, join_id

# 各 split 对应的输入/输出路径
SPLITS = {
    "train": {
        "mix_csv": "mix_csv_vocc_train",
        "mix_dir": "mix_dir_train_vocc",
        "s1_dir": "s1_dir_train_vocc",
        "s1_scp": "s1_scp_train",
        "mix_clean_scp": "mix_clean_vocc_scp_train",
    },
    "test_scale": {
        "mix_csv": "mix_csv_vocc_test_scale",
        "mix_dir": "mix_dir_test_vocc",
        "s1_dir": "s1_dir_test_vocc",
        "s1_scp": "s1_scp_test_scale",
        "mix_clean_scp": "mix_clean_vocc_scp_test_scale",
    },
}


def build_lines(mix_csv, mix_dir, s1_dir, output):
    """根据遮挡 mixture csv 构造 s1 / mix_clean scp 行（含 dev→train 归一化）。"""
    rows = read_csv_rows(mix_csv)
    out = []
    for items in rows:
        if items[1] == "dev":
            items[1] = "train"
        if items[5] == "dev":
            items[5] = "train"

        mix_utt_id = join_id(
            items[2], items[3].split("/")[0], items[3].split("/")[1],
            items[6], items[7].split("/")[0], items[7].split("/")[1],
        )
        path = "_".join(items).replace("/", "_")
        if output in ("s1", "all"):
            out.append(scp_line(mix_utt_id, s1_dir + path + ".wav"))
        if output in ("mix_clean", "all"):
            out.append(scp_line(mix_utt_id, mix_dir + path + ".wav"))
    return out


def main():
    parser = argparse.ArgumentParser(description="生成 VoxCeleb2 遮挡 scp 列表")
    parser.add_argument("--split", choices=list(SPLITS), default="test_scale")
    parser.add_argument("--output", choices=["s1", "mix_clean", "all"], default="mix_clean")
    args = parser.parse_args()

    cfg = SPLITS[args.split]
    out_path = VOX2[cfg["s1_scp"] if args.output == "s1" else cfg["mix_clean_scp"]]

    lines = build_lines(
        VOX2[cfg["mix_csv"]],
        VOX2[cfg["mix_dir"]],
        VOX2[cfg["s1_dir"]],
        args.output,
    )
    write_scp(out_path, lines)
    print(f"已写入 {len(lines)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
