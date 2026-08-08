#!/bin/bash

# set -e
# set -u
# set -o pipefail

# Libri2Mix Clean Data
###########
# out dir #
###########
# out_dir=dump/funcodec/libri2mix
out_dir=dump/funcodec/vox2_en
out_dir=dump/funcodec/same_semantic

########
# Data #
########
# scp_list=("dump/wavs/list/libri2mix_dev/s1.scp" \
#           "dump/wavs/list/libri2mix_test/s1.scp" \
#           "dump/wavs/list/libri2mix_train/s1.scp" )
# type=("dev" "test" "train")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/_rej_list/wavs.scp" 
#         )
# type=("test_rej")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/librimix_wesep_test_list/s1_wesep.scp" 
#         )
# type=("test_wesep")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_list/s1.scp" 
#         )
# type=("test")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/librimix_wesep_train100_list/s1.scp" 
#         )
# type=("train_wesep")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/librimix_wesep_train100_list_rej/wavs.scp" 
#         )
# type=("train_wesep_rej")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_list_rej/wavs.scp" 
#         )
# type=("test_rej")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list/s1.scp" 
#         )
# type=("train")

# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/dump/wavs/vox2_en_2spk_train_list_rej/wavs.scp" 
#         )
# type=("train_rej")


# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/CosyVoice/DATA_same_semantics_with_prompt_8second/scp/test_s1.scp" 
#         )
# type=("test_semantic_same")


# scp_list=(
#           "/home/export/base/sc100138/sc100138/online1/CosyVoice/DATA_same_semantics_with_prompt_8second/scp/train_s1_rej.scp" 
#         )
# type=("train_semantic_same_rej")

scp_list=(
          "/home/export/base/sc100138/sc100138/online1/CosyVoice/DATA_same_semantics_with_prompt_8second/scp/test_s1.scp" 
        )
type=("test_semantic_same")



#######
# DDP #
#######
num_proc=1

gpus="cuda:0"

#########
# Model #
#########

codec_model_file=/home/export/base/sc100138/sc100138/online1/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth
codec_config_file=/home/export/base/sc100138/sc100138/online1/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml

. utils/parse_options.sh

# Iterate using indices
for ((i=0; i<${#scp_list[@]}; i++)); do
    type=${type[$i]}
    echo "Processing $type"
    scp_file=${scp_list[$i]}
    python utils/2_export_libri2mix_funcodec.py --scp_file $scp_file \
      --config $codec_config_file --model $codec_model_file --output $out_dir/$type \
      --num_proc $num_proc --gpus $gpus \
      # >>/home/export/base/sc100138/sc100138/online1/lauraTSE_code_refact/data_dpo/utils/codectrain100rej.txt
    
done

echo "everything done"



