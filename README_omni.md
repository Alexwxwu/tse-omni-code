## Data Preprocessing
# audio only:

input: mixture_scp, enroll_scp, gt_codec_scp (for ar teacher forcing)

output: gt_codec 

# audio-visual:

input: mixture_scp, enroll_scp(audio_enroll_scp, visual_enroll_scp(continous vsr feature)), gt_codec_scp (for ar teacher forcing)

output: gt_codec 

# For Vox2: 
Step1 simulate EN mixture speech data, get_vocc_emb
/mnt/users/hccl.local/wwu/vsr-low/data_preparation_vox2_en/run_scale_data.sh

step2: create EN mixture enroll scp:
/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/list_recepts/get_vox2_enroll_list_621.py

step3: Create EN scp for mixture scp; s1 scp, enroll scp:
/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/list_recepts/get_vox2_list_621.py
 
step4: create gt funcodec:
/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/export_librispeech_funcodec_normalize.sh

step5: get vsr feature scp(replace prefix from s1_scp):
/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/list_recepts/replace_path_vox2_vsr_feat.py

# For LRS3:
similar to Vox2 preprocess, get scp files and gt codec (clean tgt speech codec)

# For YGD:
similar to Vox2 preprocess, get scp files and gt codec (clean tgt speech codec)

 

## Inference
/mnt/users/hccl.local/wwu/lauraTSE_code_refact/recipes/inference_visual_cue.sh


# manually set:
model_ckpt=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/src/model_var/log_lrs3_av_enr_2s_target_v_conformer_1_residual/model_dict_best.pt
 
 
mix_wav_scp=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/lrs3_en_2spk_test_list/mix_clean.scp

ref_wav_scp=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/dump/wavs/lrs3_en_2spk_test_list/aux_s1.scp

visual_sync_scp=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/lrs3_en/test_vsr_continous_feat/test_all_vsr_feat.scp

<!-- for avhubert:
visual_sync_scp=/mnt/users/hccl.local/wwu/lauraTSE_code_refact/data_dpo/funcodec/lrs3_en/test_avhubert_continous_feat/test_all_vsr_feat.scp  -->

## model (explicitly selected via src/model_registry.py)
# old: from model_var.laura_model_only_clean_av_unify_v_conformer import LauraTSE (deprecated, removed)
# now: pass --model_name <name> (see src/model_registry.py), e.g.
#      --model_name laura_front_align_omni_trimodal

 
 

