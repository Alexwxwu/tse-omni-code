#!/bin/bash
# Standalone AV-HuBERT token + emb extraction.
#   sh script : run_extract.sh
#   py file   : extract_avhubert.py
#
# Reads a scp list and, per utterance, saves:
#   discrete tokens -> <token_save_dir>/dev/<spk>/<vid>/<utt>.npy
#   continuous emb  -> <emb_save_dir>/dev/<spk>/<vid>/<utt>.npy

set -e
cd "$(dirname "$0")"

# ------------------------- edit these ------------------------- #
gpu_id='4'

SCP='/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_list/mix_clean.scp'

VISUAL_DIR='/mnt/users/hccl.local/wwu/Voxceleb2/origin/video/'
AUDIO_DIR='/mnt/users/hccl.local/wwu/Voxceleb2/muse/audio_clean/'
PARTITION='dev'

TOKEN_SAVE_DIR='/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/standalone_avhubert_token/'
EMB_SAVE_DIR='/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en/standalone_avhubert_continous_feat/'

# mAV-HuBERT "unit" ckpt (provides task/dataset) and vanilla AV-HuBERT ckpt (runs forward)
AV2UNIT_PATH='/mnt/users/hccl.local/wwu/av2av/mavhubert_large_noise.pt'
AVHUBERT_VANILLA_PATH='/mnt/users/hccl.local/wwu/lauraTSE_code_refact/large_vox_iter5.pt'

# fairseq source dir (the folder that CONTAINS the `fairseq` package)
FAIRSEQ_PATH='/mnt/users/hccl.local/wwu/av2av/fairseq'

# modalities fed to the model: "video" (matches original pipeline) or "audio,video"
MODALITIES='video'
# --------------------------------------------------------------- #

CUDA_VISIBLE_DEVICES="$gpu_id" \
python extract_avhubert.py \
  --scp "$SCP" \
  --visual-dir "$VISUAL_DIR" \
  --audio-dir "$AUDIO_DIR" \
  --partition "$PARTITION" \
  --token-save-dir "$TOKEN_SAVE_DIR" \
  --emb-save-dir "$EMB_SAVE_DIR" \
  --av2unit-path "$AV2UNIT_PATH" \
  --avhubertvanilla-path "$AVHUBERT_VANILLA_PATH" \
  --fairseq-path "$FAIRSEQ_PATH" \
  --modalities "$MODALITIES"
  # add "--overwrite" to force recompute, "--limit N" for a quick test, "--cpu" to run on CPU
