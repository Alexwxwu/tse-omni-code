#!/bin/bash
#
# Unified inference recipe for all LauraTSE scenarios.
#
# Usage:
#   bash recipes/inference.sh --mode <mode> [--mix_wav_scp ... --model_ckpt ... etc.]
#
# Modes (scenario -> src entry -> default model_name):
#   clean                 audio-only original LauraTSE      -> infer.py                          -> laura_base
#   vc                    basic audio-visual (clean)        -> infer_visual_cue.py               -> laura_front_align_omni
#   fromvideo             audio-visual from video           -> infer_visual_cue_from_video.py    -> laura_front_align_omni
#   streaming_fromvideo   streaming, video-derived visual   -> infer_visual_cue_streaming_from_video.py -> laura_front_align_omni
#   switch                robust to missing visual cue      -> infer_visual_cue_switch_from_video.py    -> laura_front_align_switch
#   vocc                  visual-occlusion                  -> infer_visual_cue_vocc_from_video.py      -> laura_front_align_omni
#   trimodal              audio+visual+transcript (vox2/ygd)-> infer_trimodal.py                 -> laura_front_align_omni_trimodal
#   transcript            transcript-only enrollment        -> infer_trimodal.py                 -> laura_front_align_transcript_only
#
# All model names are defined in src/model_registry.py.

# Base path of the code repo (all $ROOT paths resolve against this)
ROOT=/mnt/users/hccl.local/wwu/lauraTSE_code_refact

###########
# Setting #
###########

mode=vc

# Input wavs
mix_wav_scp=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_test_list-scale/mix_clean.scp
ref_wav_scp=$ROOT/data_dpo/dump/wavs/vox2_en_2spk_test_list-scale/aux_s1.scp
# VSR-frontend continuous visual features (512-dim .npy scp)
visual_sync_scp=$ROOT/data_dpo/funcodec/vox2_en/train_vsr_continous_feat_scale/test_all_vsr_feat_scale.scp
# Transcript dir (only used by trimodal / transcript modes)
text_direc='/mnt/users/hccl.local/wwu/YGD-mix-data/YGD/audio_clean_text/'

# LauraTSE config and ckpt
config_path=$ROOT/model_config/laura_tse_librispeech_dm_e_100_config.yaml
model_ckpt=$ROOT/model_config/laura_tse_librispeech_dm_e_100.pth
# Model variant (see src/model_registry.py). Empty = per-mode default below.
model_name=

# FunCodec ckpt and config
codec_config_file=$ROOT/codec_config/config.yaml
codec_model_file=$ROOT/codec_config/model.pth

# Output dir. Audio output will be <output_dir>/wavs/*.wav.
output_dir=$ROOT/infer_results

# DDP #
num_proc=1
gpus=cuda:0
max_aux_ds=2

# Inference behavior (per-mode defaults are applied below if left unchanged)
enroll_setting='audio_visual_transcript_enroll'
max_infer_length=6
limit_infer_length=True
enroll_stage='both_stage'
Laura=1 #1 is lauratse backbone, 0 is Qwen2.5 backbone

. utils/parse_options.sh

#######################
# Per-mode dispatch  #
#######################

case $mode in
  clean)
    infer_script=$ROOT/src/infer.py
    default_model_name=laura_base
    ;;
  vc)
    infer_script=$ROOT/src/infer_visual_cue.py
    default_model_name=laura_front_align_omni
    ;;
  fromvideo)
    infer_script=$ROOT/src/infer_visual_cue_from_video.py
    default_model_name=laura_front_align_omni
    ;;
  streaming_fromvideo)
    infer_script=$ROOT/src/infer_visual_cue_streaming_from_video.py
    default_model_name=laura_front_align_omni
    ;;
  switch)
    infer_script=$ROOT/src/infer_visual_cue_switch_from_video.py
    default_model_name=laura_front_align_switch
    ;;
  vocc)
    infer_script=$ROOT/src/infer_visual_cue_vocc_from_video.py
    default_model_name=laura_front_align_omni
    ;;
  trimodal)
    infer_script=$ROOT/src/infer_trimodal.py
    default_model_name=laura_front_align_omni_trimodal
    ;;
  transcript)
    infer_script=$ROOT/src/infer_trimodal.py
    default_model_name=laura_front_align_transcript_only
    ;;
  *)
    echo "Unknown mode '$mode'. Valid modes: clean vc fromvideo streaming_fromvideo switch vocc trimodal transcript"
    exit 1
    ;;
esac

[ -z "$model_name" ] && model_name=$default_model_name

# per-mode defaults for enrollment / length settings
case $mode in
  clean) ;;
  vc|fromvideo|streaming_fromvideo)
    [ -z "$enroll_setting" ] && enroll_setting='audio_enroll'
    [ -z "$max_infer_length" ] && max_infer_length=4
    ;;
  switch)
    [ -z "$enroll_setting" ] && enroll_setting='visual_enroll'
    [ -z "$max_infer_length" ] && max_infer_length=6
    ;;
  vocc)
    [ -z "$enroll_setting" ] && enroll_setting='visual_enroll'
    [ -z "$max_infer_length" ] && max_infer_length=10
    ;;
  trimodal)
    [ -z "$enroll_setting" ] && enroll_setting='audio_visual_transcript_enroll'
    [ -z "$max_infer_length" ] && max_infer_length=4
    ;;
  transcript)
    [ -z "$enroll_setting" ] && enroll_setting='transcript_enroll'
    [ -z "$max_infer_length" ] && max_infer_length=4
    ;;
esac

mkdir -p $output_dir

echo "[Inference] mode=$mode model_name=$model_name script=$(basename $infer_script)"

# common arguments
common_args="--mix_wav_scp $mix_wav_scp --ref_wav_scp $ref_wav_scp \
 --config $config_path --model_ckpt $model_ckpt \
 --num_proc $num_proc --gpus $gpus \
 --codec_model_file $codec_model_file --codec_config_file $codec_config_file \
 --model_name $model_name"

if [ "$mode" == "clean" ]; then
  # audio-only original LauraTSE
  python $infer_script $common_args \
   --output_dir "$output_dir/wavs"
elif [ "$mode" == "trimodal" ] || [ "$mode" == "transcript" ]; then
  # trimodal / transcript: extra text_direc, per-enroll-setting output subdir
  python $infer_script $common_args \
   --visual_sync_scp $visual_sync_scp --text_direc $text_direc \
   --output_dir "$output_dir/$enroll_setting/wavs" \
   --max_aux_ds $max_aux_ds \
   --enroll_setting $enroll_setting --max_infer_length $max_infer_length \
   --limit_infer_length $limit_infer_length --enroll_stage $enroll_stage --Laura $Laura
elif [ "$mode" == "vc" ] || [ "$mode" == "switch" ]; then
  # visual-cue / switch entries accept --Laura
  python $infer_script $common_args \
   --visual_sync_scp $visual_sync_scp \
   --output_dir "$output_dir/wavs" \
   --max_aux_ds $max_aux_ds \
   --enroll_setting $enroll_setting --max_infer_length $max_infer_length \
   --limit_infer_length $limit_infer_length --enroll_stage $enroll_stage --Laura $Laura
else
  # fromvideo / streaming_fromvideo / vocc
  python $infer_script $common_args \
   --visual_sync_scp $visual_sync_scp \
   --output_dir "$output_dir/wavs" \
   --max_aux_ds $max_aux_ds \
   --enroll_setting $enroll_setting --max_infer_length $max_infer_length \
   --limit_infer_length $limit_infer_length --enroll_stage $enroll_stage
fi
