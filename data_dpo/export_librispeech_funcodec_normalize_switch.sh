#!/bin/bash

set -e
set -u
set -o pipefail

# Libri2Mix Clean Data
###########
# out dir #
###########
out_dir=dump/funcodec/librispeech
out_dir=dump/funcodec/libri2mix
out_dir=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/vox2_en
out_dir=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/lrs3_en_switch

########
# Data #
########
# scp_list=("dump/wavs/list/librispeech_train/train_100_360_clean.scp")
# type=("train")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/list/libri2mix_test/s1.scp" 
#         )
# type=("test_norm")

scp_list=(
          "/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list-scale/s1.scp" 
        )
type=("train_scale")

scp_list=(
          "/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_test_list-scale/s1.scp" 
        )
type=("test_scale")

scp_list=(
          "/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/lrs3_en_2spk_test_list/s1.scp" 
        )
type=("test")

scp_list=(
          "/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/lrs3_en_2spk_train_list/s1.scp" 
        )
type=("train")

#######
# DDP #
#######
# num_proc=8
# gpus="cuda:0 cuda:1 cuda:2 cuda:3"
num_proc=1
gpus="cuda:0"

#########
# Model #
#########
 
codec_model_file=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/codec_config/model.pth
codec_config_file=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/codec_config/config.yaml



input_file=/mnt/users/hccl.local/wwu/switch_memo_testset/mixture_data_list_3mix_switch_6s_lrs3_test.csv
switch_audio_dir=/mnt/users/hccl.local/wwu/switch_memo_testset/mixture_switch_long_lrs3/
partition=test
python /mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/utils/2_export_libri2mix_funcodec_switch.py \
  --input_file $input_file \
  --config $codec_config_file --model $codec_model_file --output $out_dir/$type \
  --num_proc $num_proc --gpus $gpus \
  --partition $partition --switch_audio_dir $switch_audio_dir
  --normalize ## Normalize input

