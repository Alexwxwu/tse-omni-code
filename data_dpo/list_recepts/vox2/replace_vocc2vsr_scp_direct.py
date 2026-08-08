"""把遮挡 mixture csv 直接转换为遮挡 VSR 特征 scp。

用法:
    python replace_vocc2vsr_scp_direct.py --input X --output Y --feat-dir Z
"""
import argparse


def convert_line_to_output(line, feat_dir):
    """把一行遮挡 mixture csv 转换为 'utt_id path'。"""
    parts = line.strip().split(",")
    spk1_path = parts[2]  # 例如 'V1yW5IsnSjo/00009'
    spk2_path = parts[5]  # 例如 'RiM5aSvaNkg/00002'

    combined_id = f"{spk1_path.replace('/', '_')}_{spk2_path.replace('/', '_')}"

    mask_type = parts[-4]  # masktype
    mask_start = parts[-6]  # start
    mask_len = parts[-5]    # len

    mask_filename = f"mask{mask_type}_start{mask_start}_len{mask_len}.npy"
    full_path = f"{feat_dir}{spk1_path}_{mask_filename}"
    return f"{combined_id} {full_path}"


def process_file(input_file, output_file, feat_dir):
    with open(input_file) as infile, open(output_file, "w") as outfile:
        for line in infile:
            if line.strip():
                outfile.write(convert_line_to_output(line, feat_dir) + "\n")


def main():
    parser = argparse.ArgumentParser(description="把遮挡 mixture csv 转换为遮挡 VSR 特征 scp")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feat-dir", required=True, help="特征目录，需以 '/' 结尾")
    args = parser.parse_args()

    process_file(args.input, args.output, args.feat_dir)
    print(f"转换完成 -> {args.output}")


if __name__ == "__main__":
    main()
