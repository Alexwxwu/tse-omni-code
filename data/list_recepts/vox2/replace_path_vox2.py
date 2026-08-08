"""批量替换 scp 文件中的路径前缀。

用法:
    python replace_path_vox2.py --input X --old-prefix A --new-prefix B
    python replace_path_vox2.py --input X --old-prefix A --new-prefix B --output Y
"""
import argparse

from scp_utils import read_lines, write_scp


def main():
    parser = argparse.ArgumentParser(description="批量替换 scp 文件中的路径前缀")
    parser.add_argument("--input", required=True)
    parser.add_argument("--old-prefix", required=True)
    parser.add_argument("--new-prefix", required=True)
    parser.add_argument("--output", help="输出文件，默认覆盖原文件")
    args = parser.parse_args()

    lines = read_lines(args.input)
    new_lines = [line.replace(args.old_prefix, args.new_prefix) for line in lines]
    out_path = args.output or args.input
    write_scp(out_path, new_lines)
    print(f"已处理 {len(new_lines)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
