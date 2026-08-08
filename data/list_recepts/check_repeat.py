"""检查 scp 文件中是否有重复的路径。

用法:
    python check_repeat.py --input X
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="检查 scp 文件中重复的路径")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    seen = set()
    dup = 0
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            path = line.split(" ")[1]
            if path in seen:
                dup += 1
            else:
                seen.add(path)
    print(f"重复路径数: {dup}")


if __name__ == "__main__":
    main()
