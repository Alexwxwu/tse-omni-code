# FunCodec ckpt and config
# unset CUDA_VISIBLE_DEVICES

codec_model_file=../codec_config/model.pth
codec_config_file=../codec_config/config.yaml

bash export_libri2mix_funcodec.sh --codec_model_file $codec_model_file --codec_config_file $codec_config_file
