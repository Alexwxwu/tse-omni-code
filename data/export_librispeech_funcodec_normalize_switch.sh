#!/bin/bash

set -e
set -u
set -o pipefail

# Libri2Mix Clean Data
###########
# out dir #
###########
out_dir=dump/funcodec/lrs3_en_switch

########
# Data #
########
scp_list=(
  "dump/wavs/lrs3_en_2spk_test_list/s1.scp"
)
type=("test")

#######
# DDP #
#######
num_proc=1
gpus="cuda:0"

#########
# Model #
#########
codec_model_file=../codec_config/model.pth
codec_config_file=../codec_config/config.yaml

# switch 试验的输入（请按需修改为实际路径）
input_file=../switch_memo_testset/mixture_data_list_3mix_switch_6s_lrs3_test.csv
switch_audio_dir=../switch_memo_testset/mixture_switch_long_lrs3/
partition=test

python utils/2_export_libri2mix_funcodec_switch.py \
  --input_file $input_file \
  --config $codec_config_file --model $codec_model_file --output_dir $out_dir/$type \
  --num_proc $num_proc --gpus $gpus \
  --partition $partition --switch_audio_dir $switch_audio_dir \
  --normalize ## Normalize input
