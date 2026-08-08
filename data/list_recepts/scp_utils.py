"""list_recepts 公共工具函数。

提供 scp/csv 读写、路径拼接、ID 构造、ffmpeg 转换等公共逻辑，
供各数据集脚本复用，避免重复的 adhoc 代码。
"""
import os
import csv


# ---------------------------------------------------------------------------
# 文件读写
# ---------------------------------------------------------------------------
def read_lines(path):
    """读取文本文件，返回去除换行符的行列表。"""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_csv_rows(path, delimiter=","):
    """读取 CSV 文件，返回按分隔符切分的行列表（跳过空行）。"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                rows.append(line.split(delimiter))
    return rows


def write_scp(path, lines):
    """把形如 'utt_id path' 的行列表写入 scp 文件。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        for line in lines:
            f.write(line + "\n")


def write_csv(path, rows, delimiter=","):
    """把行列表写入 CSV 文件。"""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------
def ensure_dir(path):
    """确保文件所在目录存在，返回原路径。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def ensure_audio_from_video(audio_path, video_path, force=False):
    """若音频不存在，则用 ffmpeg 从视频提取音频，返回音频路径。"""
    if os.path.isfile(audio_path) and not force:
        return audio_path
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    ret = os.system(f"ffmpeg -y -i {video_path} {audio_path}")
    if ret != 0:
        raise RuntimeError(f"ffmpeg 转换失败: {video_path}")
    return audio_path


# ---------------------------------------------------------------------------
# ID / 路径构造
# ---------------------------------------------------------------------------
def join_id(*parts):
    """用下划线连接多个片段，并把 '/' 替换为 '_'，用于构造 utt_id。"""
    return "_".join(str(p) for p in parts).replace("/", "_")


def scp_line(utt_id, path):
    """构造一行 'utt_id path'。"""
    return f"{utt_id} {path}"
