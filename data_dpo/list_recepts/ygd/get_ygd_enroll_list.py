"""生成 YGD 的 aux_s1（注册）scp 列表。

对每个 mixture 行，在辅助说话人目录中随机选一个非目标 wav 作为注册音频。

用法:
    python get_ygd_enroll_list_621.py --split train
"""
import argparse
import os
import random

from paths import YGD
from scp_utils import read_csv_rows, write_scp, scp_line


def build_lines(mix_csv, audio_dir, split):
    """为每个 mixture 行随机挑选一个辅助说话人 wav。"""
    rows = read_csv_rows(mix_csv)
    out = []
    for items in rows:
        if items[0] != split:
            continue
        aux_id = items[2]
        chapter_id = items[3]
        aux_id_dir = audio_dir + split + "/" + aux_id + "/"
        all_wavs = [f for f in os.listdir(aux_id_dir) if f.endswith(".wav")]
        target_name = f"{chapter_id}.wav"
        candidate_wavs = [f for f in all_wavs if f != target_name]
        if not candidate_wavs:
            candidate_wavs = [target_name]
        chosen_wav = random.choice(candidate_wavs)
        chosen_wav_path = os.path.join(aux_id_dir, chosen_wav)
        utt_id = items[2] + "+" + items[3] + "$" + items[6] + "+" + items[7]
        out.append(scp_line(utt_id, chosen_wav_path))
    return out


def main():
    parser = argparse.ArgumentParser(description="生成 YGD aux_s1 注册 scp 列表")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    lines = build_lines(YGD["mix_csv"], YGD["aux_audio_dir"], args.split)
    write_scp(YGD["aux_s1_scp_train"], lines)
    print(f"已写入 {len(lines)} 行 -> {YGD['aux_s1_scp_train']}")


if __name__ == "__main__":
    main()
