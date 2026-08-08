"""从 TSE 输出目录提取 wav 并生成 scp 文件。

用法:
    python 1_extract_infered_wav_scp.py --tse_output_dir X --output_scp Y
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="从 TSE 输出目录提取 wav 生成 scp")
    parser.add_argument("--tse_output_dir", required=True, help="TSE 输出 wav 目录")
    parser.add_argument("--output_scp", required=True, help="输出 scp 文件路径")
    args = parser.parse_args()

    with open(args.output_scp, "w") as scp_file:
        for filename in os.listdir(args.tse_output_dir):
            if not filename.endswith(".wav"):
                continue
            if "vox2" in args.output_scp:
                filebase_name = filename.split(".wav")[0]
                id1_start = 12
                id1_length = 26
                id2_start = 12 + 34
                id2_length = 25
                id1 = filebase_name[id1_start:id1_start + id1_length]
                id2 = filebase_name[id2_start:id2_start + id2_length]
                base_name = id1 + id2
            else:
                base_name = filename.split(".wav")[0]
            scp_line = f"{base_name} {os.path.join(args.tse_output_dir, filename)}\n"
            scp_file.write(scp_line)


if __name__ == "__main__":
    main()
