#!/bin/bash
#
# Unified training recipe for LauraTSE (model_var).
#
# Usage:
#   bash train.sh --mode <mode> [--gpu_id '0,1' --batch_size 8 --model_name laura_tse ...]
#
# Modes (dataset / scenario):
#   vox2            vox2 EN 2-spk mix, VSR-frontend features
#   vox2_vocc       vox2 with visual occlusion (vocc) lists
#   avhubert_vox2   vox2 with AV-HuBERT continuous visual features
#   lrs3            lrs3 EN 2-spk mix (qwen backbone, --Laura 0)
#   lrs3_vocc       lrs3 with visual occlusion lists
#   lrs3_avhubert   lrs3 with AV-HuBERT visual features
#   lrs3_switch     lrs3 switch dataset (csv-based, missing-cue robustness)
#   scale           vox2-scale data, qwen backbone (--Laura 0)
#   trimodal_vox2   vox2-scale trimodal (audio+visual+transcript)
#   trimodal_ygd    ygd trimodal
#   ygd             ygd EN 2-spk mix (omni)
#
# The model variant is selected through src/model_registry.py
# (optional --model_name, otherwise the registry default).

# Base path of the code repo (all $ROOT paths resolve against this)
ROOT=/mnt/users/hccl.local/wwu/lauraTSE_code_refact

conda activate FAcodec
export PATH=$PWD:$PATH
export PYTHONPATH=$PWD:$(dirname $(dirname $PWD)):$(dirname $PWD):$(dirname $PWD)/Model:$PYTHONPATH

###########
# Setting #
###########

mode=vox2
model_name=                       # registry name, empty = registry default
data_mode=normal                  # dataset class in dataload.py (set per mode below)
text_direc=                       # transcript dir (scale_trimodal / gesture_trimodal)

# common hyperparameters (per-mode defaults applied below, all overridable)
config_path=$ROOT/model_config/laura_tse_librispeech_dm_e_100_config.yaml
continue_from=$ROOT/model_config/laura_tse_librispeech_dm_e_100.pth
log_name=
gpu_id='0'
master_port=1209
epochs=600
max_length=4
max_enroll_length=2
accu_grad=0
batch_size=8
num_workers=2
use_tensorboard=0
lr=1e-3
use_visual_aux=0
Laura=1
lip_setting=0
enroll_second=2

# data lists (scp family; unused by lrs3_switch)
mix_lst_train_path=
aux_train_list=
codec_lst_train_path=
vsr_sync_lst_train_path=
vsr_sync_lst_path=
mix_lst_path=
aux_list=
codec_lst_path=
visual_codec_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train_avhubert_token/all_visual_code.scp
visual_dir='/mnt/users/hccl.local/wwu/Voxceleb2/origin/video/'
vsr_feat_dir=

# switch-family data (only used by lrs3_switch)
mix_csv_path='/mnt/users/hccl.local/wwu/switch_memo_testset/mixture_data_list_3mix_switch_6s_lrs3_train_and_test.csv'
codec_dir=$ROOT/data_dpo/funcodec/lrs3_en_switch/
switch_audio_dir='/mnt/users/hccl.local/wwu/switch_memo_testset/mixture_switch_long_lrs3/'

. ../../utils/parse_options.sh

#######################
# Per-mode dispatch  #
#######################

case $mode in
  vox2)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/log_av_enr_2s/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list/mix_clean.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/all_vsr_feat.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/test_all_vsr_feat.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_list/mix_clean.scp
    aux_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/vox2_en/test/all.scp
    vsr_feat_dir=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat
    gpu_id='1,3,4,6'; master_port=1209; batch_size=32; lr=1e-3
    data_mode=normal
    ;;
  vox2_vocc)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/log_av_enr_2s_vocc/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list/mix_clean_vocc.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/train_all_vsr_feat_vocc.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/test_all_vsr_feat_vocc.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_list/mix_clean_vocc.scp
    aux_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/vox2_en/test/all.scp
    vsr_feat_dir='/mnt/users/hccl.local/wwu/Voxceleb2/muse/lip_occ_3type_en/'
    gpu_id='4,5,6,7'; master_port=1215; batch_size=32; lr=1e-3
    data_mode=vocc_vox2
    ;;
  avhubert_vox2)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/log_av_enr_2s_avhubert/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list/mix_clean.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train_avhubert_continous_feat/all_vsr_feat.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/vox2_en/train_avhubert_continous_feat/test_all_vsr_feat.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_list/mix_clean.scp
    aux_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/vox2_en/test/all.scp
    vsr_feat_dir=$ROOT/data_dpo/funcodec/vox2_en/train_avhubert_continous_feat
    gpu_id='0,1,2'; master_port=1117; batch_size=20; lr=1e-3
    data_mode=normal
    ;;
  lrs3)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/lrs3_log/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_train_list/mix_clean.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_train_list/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/lrs3_en/train/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/lrs3_en/train_vsr_continous_feat/all_vsr_feat.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/lrs3_en/test_vsr_continous_feat/test_all_vsr_feat.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_test_list/mix_clean.scp
    aux_list=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_test_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/lrs3_en/test/all.scp
    vsr_feat_dir=$ROOT/data_dpo/funcodec/lrs3_en/train_vsr_continous_feat/
    gpu_id='0,1,2,3'; master_port=1215; batch_size=8; lr=1e-4; Laura=0
    data_mode=normal
    ;;
  lrs3_vocc)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/log_lrs3_vocc/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_train_list/mix_clean_vocc.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_train_list/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/lrs3_en/train/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/lrs3_en/train_vsr_continous_feat/train_all_vsr_feat_vocc.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/lrs3_en/test_vsr_continous_feat/test_all_vsr_feat_vocc.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_test_list/mix_clean_vocc.scp
    aux_list=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_test_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/lrs3_en/test/all.scp
    vsr_feat_dir=$ROOT/data_dpo/funcodec/lrs3_en/train_vsr_continous_feat
    gpu_id='1,3,4,6'; master_port=1203; batch_size=32; lr=1e-3
    data_mode=vocc_lrs3
    ;;
  lrs3_avhubert)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/log_lrs3_avhubert/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_train_list/mix_clean.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_train_list/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/lrs3_en/train/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/lrs3_en/train_avhubert_continous_feat/train_all_vsr_feat.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/lrs3_en/test_avhubert_continous_feat/test_all_vsr_feat.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_test_list/mix_clean.scp
    aux_list=$ROOT/data_dpo/dump/wavs/lrs3_en_2spk_test_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/lrs3_en/test/all.scp
    vsr_feat_dir=$ROOT/data_dpo/funcodec/lrs3_en/train_vsr_continous_feat
    gpu_id='4,5'; master_port=1207; batch_size=32; lr=1e-3
    data_mode=normal
    ;;
  lrs3_switch)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/lrs3_log_after_ijcai/switch/
    visual_dir='/mnt/Corpus-Upload/lrs3/mp4/'
    gpu_id='4,5,6,7'; master_port=1210; batch_size=16; lr=1e-3
    max_length=6; num_workers=4; lip_setting=1
    data_mode=switch
    ;;
  scale)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/vox2_log_after_ijcai/scale/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list-scale/mix_clean.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list-scale/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train_scale/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat_scale/all_vsr_feat.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat/test_all_vsr_feat.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_list/mix_clean.scp
    aux_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/vox2_en/test/all.scp
    vsr_feat_dir=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat
    gpu_id='1,4,5,6,7'; master_port=1203; batch_size=20; lr=1e-4
    Laura=0; lip_setting=1; enroll_second=2
    data_mode=scale
    ;;
  trimodal_vox2)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/vox2_log_after_ijcai/trimodal/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list-scale/mix_clean.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_train_list-scale/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train_scale/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat_scale/all_vsr_feat.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat_scale/test_all_vsr_feat_scale.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_test_list-scale/mix_clean.scp
    aux_list=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_test_list-scale/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/vox2_en/test_scale/all.scp
    vsr_feat_dir=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat
    gpu_id='1'; master_port=1207; batch_size=4; lr=1e-4; num_workers=0
    Laura=1; model_name=${model_name:-laura_tse}
    data_mode=scale_trimodal
    [ -z "$text_direc" ] && text_direc='/mnt/users/hccl.local/wwu/vsr-low/English/Voxceleb2-En-mix-scale/audio_clean_text_en/'
    ;;
  trimodal_ygd)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/ygd_log_after_ijcai/trimodal/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/ygd_en_2spk_train_list/mix_clean.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/ygd_en_2spk_train_list/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/ygd_en/train/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/ygd_en/train_vsr_continous_feat/train_all_vsr_feat.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/ygd_en/test_vsr_continous_feat/test_all_vsr_feat.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/ygd_en_2spk_test_list/mix_clean.scp
    aux_list=$ROOT/data_dpo/dump/wavs/ygd_en_2spk_test_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/ygd_en/test/all.scp
    vsr_feat_dir='/mnt/users/hccl.local/wwu/YGD-mix-data/YGD/visual/gesture_embedding/'
    gpu_id='4,5,6,7'; master_port=1213; batch_size=24; lr=1e-4
    Laura=1; model_name=${model_name:-laura_tse}
    data_mode=scale_trimodal
    [ -z "$text_direc" ] && text_direc='/mnt/users/hccl.local/wwu/YGD-mix-data/YGD/audio_clean_text/'
    ;;
  ygd)
    [ -z "$log_name" ] && log_name=$ROOT/src/model_var/ygd_log_after_ijcai/omni/
    mix_lst_train_path=$ROOT/data_dpo/dump/wavs/ygd_en_2spk_train_list/mix_clean.scp
    aux_train_list=$ROOT/data_dpo/dump/wavs/ygd_en_2spk_train_list/aux_s1.scp
    codec_lst_train_path=$ROOT/data_dpo/funcodec/ygd_en/train/all.scp
    vsr_sync_lst_train_path=$ROOT/data_dpo/funcodec/ygd_en/train_vsr_continous_feat/train_all_vsr_feat.scp
    vsr_sync_lst_path=$ROOT/data_dpo/funcodec/ygd_en/test_vsr_continous_feat/test_all_vsr_feat.scp
    mix_lst_path=$ROOT/data_dpo/dump/wavs/ygd_en_2spk_test_list/mix_clean.scp
    aux_list=$ROOT/data_dpo/dump/wavs/ygd_en_2spk_test_list/aux_s1.scp
    codec_lst_path=$ROOT/data_dpo/funcodec/ygd_en/test/all.scp
    vsr_feat_dir='/mnt/users/hccl.local/wwu/YGD-mix-data/YGD/visual/gesture_embedding/'
    gpu_id='0'; master_port=1103; batch_size=16; lr=1e-3
    data_mode=normal
    ;;
  *)
    echo "Unknown mode '$mode'. Valid modes: vox2 vox2_vocc avhubert_vox2 lrs3 lrs3_vocc lrs3_avhubert lrs3_switch scale trimodal_vox2 trimodal_ygd ygd"
    exit 1
    ;;
esac

mkdir -p $log_name

echo "[Train] mode=$mode model_name=${model_name:-<registry default>} log_name=$log_name"

num_gpus=$(echo $gpu_id | awk -F ',' '{print NF}')

# optional registry model selection
model_name_arg=""
if [ -n "$model_name" ]; then
  model_name_arg="--model_name $model_name"
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$num_gpus \
--master_port=$master_port \
$ROOT/src/model_var/main.py \
--log_name $log_name \
--config $config_path \
--epochs $epochs \
--max_length $max_length \
--max_enroll_length $max_enroll_length \
--accu_grad $accu_grad \
--batch_size $batch_size \
--num_workers $num_workers \
--use_tensorboard $use_tensorboard \
--lr $lr \
--use_visual_aux $use_visual_aux \
--Laura $Laura \
--lip_setting $lip_setting \
--enroll_second $enroll_second \
--continue_from $continue_from \
--data_mode $data_mode \
--text_direc $text_direc \
$model_name_arg \
$(if [ "$mode" == "lrs3_switch" ]; then
  echo "--mix_csv_path $mix_csv_path --codec_dir $codec_dir --visual_dir $visual_dir --switch_audio_dir $switch_audio_dir"
else
  echo "--mix_lst_train_path $mix_lst_train_path \
        --aux_train_list $aux_train_list \
        --codec_lst_train_path $codec_lst_train_path \
        --vsr_sync_lst_train_path $vsr_sync_lst_train_path \
        --vsr_sync_lst_path $vsr_sync_lst_path \
        --mix_lst_path $mix_lst_path \
        --aux_list $aux_list \
        --codec_lst_path $codec_lst_path \
        --visual_codec_lst_train_path $visual_codec_lst_train_path \
        --visual_dir $visual_dir \
        --vsr_feat_dir $vsr_feat_dir"
fi) \
2>&1 | tee $log_name/train.log
