"""集中管理 list_recepts 中所有脚本用到的路径。

原始脚本把机器/用户相关的绝对路径硬编码在各自顶部，换机器就全部失效。
这里统一集中管理，换机器时只需修改本文件顶部的根路径（或设置环境变量）。

所有根路径均支持环境变量覆盖，例如：
    LAURA_USER_ROOT=/new/user/root python get_vox2_list_621.py
"""
import os


def _env(name, default):
    """读取环境变量，未设置时返回默认值。"""
    return os.environ.get(name, default)


# ===========================================================================
# 根路径（换机器时优先修改这里，或设置对应环境变量）
# ===========================================================================
# 用户主目录（原始脚本中 /mnt/users/hccl.local/wwu）
USER_ROOT = _env("LAURA_USER_ROOT", "/mnt/users/hccl.local/wwu")

# 输出根目录（scp/dump 输出位置，原始脚本中 .../lauraTSE_code_refact/data_dpo）
OUTPUT_ROOT = _env("LAURA_OUTPUT_ROOT", f"{USER_ROOT}/lauraTSE_code_refact/data_dpo")

# 共享语料根目录（原始脚本中 /mnt/Corpus-Upload）
CORPUS_ROOT = _env("LAURA_CORPUS_ROOT", "/mnt/Corpus-Upload")

# 旧集群数据根目录（原始脚本中 /home/export/base/sc100138/sc100138/online1）
LEGACY_ROOT = _env("LAURA_LEGACY_ROOT", "/home/export/base/sc100138/sc100138/online1")

# 旧 wesep 数据根目录（原始脚本中 /home/export/base/sc100135/sc100135/online1）
WESEP_ROOT = _env("LAURA_WESEP_ROOT", "/home/export/base/sc100135/sc100135/online1")

# 各数据集数据根目录
VSR_LOW_ROOT = _env("LAURA_VSR_LOW_ROOT", f"{USER_ROOT}/vsr-low")
YGD_ROOT = _env("LAURA_YGD_ROOT", f"{USER_ROOT}/YGD-mix-data/YGD")
AVTSE_ROOT = _env("LAURA_AVTSE_ROOT", f"{USER_ROOT}/AVTSE_Momentum")


# ===========================================================================
# 输出 scp 目录辅助函数
# ===========================================================================
def _dump(*parts):
    """构造 dump/wavs 下的输出 scp 路径。"""
    return os.path.join(OUTPUT_ROOT, "dump", "wavs", *parts)


def _funcodec(*parts):
    """构造 funcodec 下的输出 scp 路径。"""
    return os.path.join(OUTPUT_ROOT, "funcodec", *parts)


# ===========================================================================
# VoxCeleb2 (vox2)
# ===========================================================================
VOX2 = {
    # 混合列表 csv
    "mix_csv_train": f"{LEGACY_ROOT}/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/mixture_data_list_2mix_en_train.csv",
    "mix_csv_train_scale": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix-scale/mixture_data_list_2mix_en_train.csv",
    "mix_csv_test_scale": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix-scale/mixture_data_list_2mix_en_test.csv",
    "mix_csv_vocc_train": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix/mixture_data_list_2mix_en_with_occludded_train.csv",
    "mix_csv_vocc_test_scale": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix/mixture_data_list_2mix_en_with_occludded_test_scale.csv",
    # 混合/目标音频目录
    "mix_dir_train": f"{LEGACY_ROOT}/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/train/mix/",
    "s1_dir_train": f"{LEGACY_ROOT}/vsr-low-tse-dataset/vsr-low/English/Voxceleb2-En-mix/train/s1/",
    "mix_dir_train_scale": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix-scale/train/mix/",
    "s1_dir_train_scale": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix-scale/train/s1/",
    "mix_dir_test_scale": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix-scale/test/mix/",
    "s1_dir_test_scale": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix-scale/test/s1/",
    "mix_dir_train_vocc": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix/train/mix/",
    "s1_dir_train_vocc": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix/train/s1/",
    "mix_dir_test_vocc": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix/test/mix/",
    "s1_dir_test_vocc": f"{VSR_LOW_ROOT}/English/Voxceleb2-En-mix/test/s1/",
    # 辅助说话人音频/视频目录
    "aux_audio_dir_legacy": f"{LEGACY_ROOT}/voxceleb2/wav/",
    "aux_audio_dir": f"{CORPUS_ROOT}/Voxceleb2/origin/audio_clean/",
    "aux_video_dir_legacy": f"{LEGACY_ROOT}/voxceleb2/mp4/",
    "aux_video_dir": f"{CORPUS_ROOT}/Voxceleb2/origin/video/",
    # file.list
    "file_list_legacy": f"{LEGACY_ROOT}/vsr-low-tse-dataset/vsr-low/file.list",
    "file_list": f"{VSR_LOW_ROOT}/file.list",
    # 输出 scp
    "s1_scp_train": _dump("vox2_en_2spk_train_list", "s1.scp"),
    "mix_clean_scp_train": _dump("vox2_en_2spk_train_list", "mix_clean.scp"),
    "aux_s1_scp_train": _dump("vox2_en_2spk_train_list", "aux_s1.scp"),
    "s1_scp_train_scale": _dump("vox2_en_2spk_train_list-scale", "s1.scp"),
    "mix_clean_scp_train_scale": _dump("vox2_en_2spk_train_list-scale", "mix_clean.scp"),
    "aux_s1_scp_train_scale": _dump("vox2_en_2spk_train_list-scale", "aux_s1.scp"),
    "s1_scp_test_scale": _dump("vox2_en_2spk_test_list-scale", "s1.scp"),
    "mix_clean_scp_test_scale": _dump("vox2_en_2spk_test_list-scale", "mix_clean.scp"),
    "aux_s1_scp_test_scale": _dump("vox2_en_2spk_test_list-scale", "aux_s1.scp"),
    "mix_clean_vocc_scp_train": _dump("vox2_en_2spk_train_list", "mix_clean_vocc.scp"),
    "mix_clean_vocc_scp_test_scale": _dump("vox2_en_2spk_test_list-scale", "mix_clean_vocc.scp"),
    # funcodec
    "codec_all_scp_test": _funcodec("vox2_en", "test", "all.scp"),
    "codec_all_scp_train_scale": _funcodec("vox2_en", "train_scale", "all.scp"),
    "codec_all_scp_test_scale": _funcodec("vox2_en", "test_scale", "all.scp"),
    "codec_all_scp_train": _funcodec("vox2_en", "train", "all.scp"),
    "codec_token_dir_train_avhubert": _funcodec("vox2_en", "train_avhubert_token", "dev"),
    "codec_token_scp_train_avhubert": _funcodec("vox2_en", "train_avhubert_token", "all_visual_code.scp"),
    # VSR 特征
    "vsr_feat_dir_dev": f"{USER_ROOT}/Voxceleb2/muse/lip/dev/",
    "vsr_feat_dir_test": f"{USER_ROOT}/Voxceleb2/muse/lip/test/",
    "vsr_feat_dir_vocc_dev": f"{USER_ROOT}/Voxceleb2/muse/lip_occ_3type_en/dev/",
    "vsr_feat_dir_vocc_test": f"{USER_ROOT}/Voxceleb2/muse/lip_occ_3type_en/test/",
    "vsr_feat_scp_test": _funcodec("vox2_en", "train_vsr_continous_feat", "test_all_vsr_feat.scp"),
    "vsr_feat_scp_train_scale": _funcodec("vox2_en", "train_vsr_continous_feat_scale", "all_vsr_feat_scale.scp"),
    "vsr_feat_scp_test_scale": _funcodec("vox2_en", "train_vsr_continous_feat_scale", "test_all_vsr_feat_scale.scp"),
    "vsr_feat_vocc_scp_train": _funcodec("vox2_en", "train_vsr_continous_feat", "train_all_vsr_feat_vocc.scp"),
    "vsr_feat_vocc_scp_test": _funcodec("vox2_en", "train_vsr_continous_feat", "test_all_vsr_feat_vocc.scp"),
    "vsr_feat_vocc_scp_test_scale": _funcodec("vox2_en", "train_vsr_continous_feat_scale", "test_all_vsr_feat_vocc.scp"),
    "avhubert_feat_dir_train": _funcodec("vox2_en", "train_avhubert_continous_feat", "dev"),
    "avhubert_feat_scp_train": _funcodec("vox2_en", "train_avhubert_continous_feat", "all_vsr_feat_scale.scp"),
}


# ===========================================================================
# LRS3 (lrs3)
# ===========================================================================
LRS3 = {
    # 混合列表 csv
    "mix_csv_test_verylong": f"{AVTSE_ROOT}/lrs3_very_long_testset_scripts/testset2mix.csv",
    "mix_csv_vocc_test": f"{VSR_LOW_ROOT}/data_preparation_LRS3/mixture_data_list_2mix_with_occludded_lrs3_test_max4s.csv",
    # 混合/目标音频目录
    "mix_dir_test_verylong": f"{AVTSE_ROOT}/lrs3_very_long_testset/test/mix/",
    "s1_dir_test_verylong": f"{AVTSE_ROOT}/lrs3_very_long_testset/test/s1/",
    "mix_dir_test_vocc": f"{CORPUS_ROOT}/lrs3/mixture/test/mix/",
    "s1_dir_test_vocc": f"{CORPUS_ROOT}/lrs3/mixture/test/s1/",
    # 辅助说话人音频/视频目录
    "aux_audio_dir": f"{CORPUS_ROOT}/lrs3/wav/",
    "aux_video_dir": f"{CORPUS_ROOT}/lrs3/mp4/",
    "all_list": f"{VSR_LOW_ROOT}/data_preparation_LRS3/lrs3_wav_files.csv",
    # 输出 scp
    "s1_scp_test_verylong": _dump("lrs3_en_2spk_test_verylong_list", "s1.scp"),
    "mix_clean_scp_test_verylong": _dump("lrs3_en_2spk_test_verylong_list", "mix_clean.scp"),
    "aux_s1_scp_test_verylong": _dump("lrs3_en_2spk_test_verylong_list", "aux_s1.scp"),
    "s1_scp_test": _dump("lrs3_en_2spk_test_list", "s1.scp"),
    "mix_clean_scp_test": _dump("lrs3_en_2spk_test_list", "mix_clean.scp"),
    "mix_clean_vocc_scp_test": _dump("lrs3_en_2spk_test_list", "mix_clean_vocc.scp"),
    "mix_clean_vocc_scp_train": _dump("lrs3_en_2spk_train_list", "mix_clean_vocc.scp"),
    # funcodec
    "codec_all_scp_test": _funcodec("lrs3_en", "test", "all.scp"),
    "codec_all_scp_train": _funcodec("lrs3_en", "train", "all.scp"),
    "codec_all_scp_test_verylong": _funcodec("lrs3_en_2spk_test_verylong", "test", "all.scp"),
    "codec_token_dir_test_avhubert": _funcodec("lrs3_en", "test_avhubert_token", "test"),
    "codec_token_scp_test_avhubert": _funcodec("lrs3_en", "test_avhubert_token", "test_all_visual_code.scp"),
    # VSR 特征
    "vsr_feat_dir_test": f"{CORPUS_ROOT}/lrs3/lip/test/",
    "vsr_feat_dir_pretrain": f"{CORPUS_ROOT}/lrs3/lip/pretrain/",
    "vsr_feat_dir_verylong": f"{AVTSE_ROOT}/lrs3_very_long_testset/mp4/",
    "vsr_feat_dir_vocc_pretrain": f"{CORPUS_ROOT}/lrs3/lip_occ_max4s_2sclean/pretrain/",
    "vsr_feat_dir_vocc_test": f"{CORPUS_ROOT}/lrs3/lip_occ_max4s_2sclean/test/",
    "vsr_feat_scp_test": _funcodec("lrs3_en", "test_vsr_continous_feat", "test_all_vsr_feat.scp"),
    "vsr_feat_scp_train": _funcodec("lrs3_en", "train_vsr_continous_feat", "train_all_vsr_feat.scp"),
    "vsr_feat_scp_test_verylong": _funcodec("lrs3_en_2spk_test_verylong", "test_vsr_continous_feat", "test_all_vsr_feat.scp"),
    "vsr_feat_vocc_scp_train": _funcodec("lrs3_en", "train_vsr_continous_feat", "train_all_vsr_feat_vocc.scp"),
    "vsr_feat_vocc_scp_test": _funcodec("lrs3_en", "test_vsr_continous_feat", "test_all_vsr_feat_vocc.scp"),
    "vsr_feat_vocc_scp_test_max4s": _funcodec("lrs3_en", "test_vsr_continous_feat", "test_all_vsr_feat_vocc_max4s.scp"),
    "avhubert_feat_dir_train": _funcodec("lrs3_en", "train_avhubert_continous_feat", "pretrain"),
    "avhubert_feat_dir_test": _funcodec("lrs3_en", "test_avhubert_continous_feat", "test"),
    "avhubert_feat_scp_train": _funcodec("lrs3_en", "train_avhubert_continous_feat", "train_all_vsr_feat.scp"),
    "avhubert_feat_scp_test": _funcodec("lrs3_en", "test_avhubert_continous_feat", "test_all_vsr_feat.scp"),
}


# ===========================================================================
# YGD (ygd)
# ===========================================================================
YGD = {
    # 混合列表 csv
    "mix_csv": f"{YGD_ROOT}/audio_mixture/2_mix_min/mixture_data_list_2mix.csv",
    # 混合/目标音频目录
    "mix_dir": f"{YGD_ROOT}/audio_mixture/2_mix_min/",
    "s1_dir": f"{YGD_ROOT}/audio_clean/",
    # 辅助说话人音频/视觉目录
    "aux_audio_dir": f"{YGD_ROOT}/audio_clean/",
    "gesture_dir": f"{YGD_ROOT}/visual/gesture_embedding/",
    # 输出 scp
    "s1_scp_train": _dump("ygd_en_2spk_train_list", "s1.scp"),
    "mix_clean_scp_train": _dump("ygd_en_2spk_train_list", "mix_clean.scp"),
    "aux_s1_scp_train": _dump("ygd_en_2spk_train_list", "aux_s1.scp"),
    # funcodec
    "codec_all_scp_test": _funcodec("ygd_en", "test", "all.scp"),
    "codec_all_scp_train": _funcodec("ygd_en", "train", "all.scp"),
    # VSR 特征
    "vsr_feat_dir_test": f"{YGD_ROOT}/visual/gesture_embedding/test/",
    "vsr_feat_dir_train": f"{YGD_ROOT}/visual/gesture_embedding/train/",
    "vsr_feat_scp_test": _funcodec("ygd_en", "test_vsr_continous_feat", "test_all_vsr_feat.scp"),
    "vsr_feat_scp_train": _funcodec("ygd_en", "train_vsr_continous_feat", "train_all_vsr_feat.scp"),
}


# ===========================================================================
# LibriMix (librimix / wesep)
# ===========================================================================
LIBRIMIX = {
    # 输入
    "wav_raw_scp_test": f"{WESEP_ROOT}/wesep/wesep/examples/librimix/tse/v2/data/clean/test/wav.scp",
    "enroll_raw_scp_test": f"{WESEP_ROOT}/wesep/wesep/examples/librimix/tse/v2/data/clean/test/spk1.enroll",
    "enroll_json_train100": f"{WESEP_ROOT}/wesep/wesep/examples/librimix/tse/v2/data/clean/train-100/spk2enroll.json",
    "wav_raw_scp_train100": f"{WESEP_ROOT}/wesep/wesep/examples/librimix/tse/v2/data/clean/train-100/wav.scp",
    "enroll_s1_wav": f"{WESEP_ROOT}/data/Libri2Mix/wav16k/min/test/",
    # 输出 scp
    "aux_s1_scp_test": _dump("librimix_wesep_test_list", "aux_s1.scp"),
    "mix_clean_scp_test": _dump("librimix_wesep_test_list", "mix_clean.scp"),
    "s1_scp_test": _dump("librimix_wesep_test_list", "s1.scp"),
    "aux_s1_scp_train100": _dump("librimix_wesep_train100_list", "aux_s1_scp_wesep.csv"),
    "mix_clean_scp_train100": _dump("librimix_wesep_train100_list", "mix_clean_scp_wesep.csv"),
    "s1_scp_train100": _dump("librimix_wesep_train100_list", "s1.scp"),
}
