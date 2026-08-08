# FunCodec ckpt and config
# unset CUDA_VISIBLE_DEVICES

codec_model_file=/home/export/base/sc100138/sc100138/online1/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/model.pth
codec_config_file=/home/export/base/sc100138/sc100138/online1/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch/config.yaml


bash export_libri2mix_funcodec.sh --codec_model_file $codec_model_file --codec_config_file $codec_config_file