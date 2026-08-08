#!/bin/bash

set -e
set -u
set -o pipefail

# Libri2Mix Clean Data
###########
# out dir #
###########
out_dir=dump/funcodec/ygd_en

########
# Data #
########
scp_list=(
  "dump/wavs/ygd_en_2spk_test_list/s1.scp"
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

. utils/parse_options.sh

# Iterate using indices
for ((i=0; i<${#scp_list[@]}; i++)); do
    type=${type[$i]}
    echo "Processing $type"
    scp_file=${scp_list[$i]}
    python utils/recon_funcodec_wav.py --scp_file $scp_file \
      --config $codec_config_file --model $codec_model_file --output $out_dir/$type \
      --num_proc $num_proc --gpus $gpus \
      --normalize ## Normalize input
done

echo "everything done"
