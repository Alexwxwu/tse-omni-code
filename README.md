# TSE-Omni

Built from LauraTSE: a single lightweight auto-regressive LLM that supports
multiple-cue target speaker extraction (TSE) with limited training data.

- **Efficient**: A unified framework that supports both audio-based and
  visual-based TSE within a single autoregressive LLM, eliminating the need for
  modality-specific architectures and reducing deployment complexity.
- **Self-enrollment**: Leverages the LLM's next-token prediction capability to
  naturally integrate historical predicted speech semantic tokens as
  self-enrollment, seamlessly aligning them with synchronous visual semantic
  tokens.
- **Robust**: Supports target speaker switch, visual corruption, and streaming
  inference scenarios.

## Installation

Our experiments are run in `python3.10`.

1. Install [FunCodec](https://github.com/modelscope/FunCodec) package.
2. Install the FunCodec
   [model](https://huggingface.co/alibaba-damo/audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch).
3. `pip install -r requirements.txt`.

## Inference

All scenarios share ONE entry script: `recipes/inference.sh`. Select the
scenario with `--mode`:

```sh
bash recipes/inference.sh --mode <mode> \
 --mix_wav_scp <mix scp> --ref_wav_scp <ref scp> \
 --config_path <model config> --model_ckpt <ckpt> \
 --codec_model_file <funcodec model> --codec_config_file <funcodec yaml> \
 --output_dir <output> --num_proc 4 --gpus "cuda:0 cuda:1 cuda:2 cuda:3"
```

Supported modes:

| `--mode` | Scenario | infer script | default `--model_name` |
| --- | --- | --- | --- |
| `trimodal` | audio + visual + transcript (vox2 / ygd) | `src/infer_trimodal.py` | `laura_tse` |
| `switch` | robust to missing visual cue | `src/infer_visual_cue_switch_from_video.py` | `laura_front_align_switch` |
| `streaming_fromvideo` | chunk/streaming inference (video-derived) | `src/infer_visual_cue_streaming_from_video.py` | `laura_tse` |
| `vocc` | visual-occlusion inference | `src/infer_visual_cue_vocc_from_video.py` | `laura_tse` |
| (`--Laura 0`) | Qwen2 / Llama backbone experiments | - | `qwen2_omni_av` |
| - | Gesture branch | - | `laura_gesture_rnn*`, `laura_front_align_omni_gesture_trimodal` |

### Example: trimodal inference (vox2 / ygd)

```sh
bash recipes/inference.sh --mode trimodal \
  --mix_wav_scp <path>/mix_clean.scp \
  --ref_wav_scp <path>/aux_s1.scp \
  --visual_sync_scp <path>/test_all_vsr_feat.scp \
  --text_direc <path>/audio_clean_text/ \
  --config_path <path>/model_config/laura_tse_librispeech_dm_e_100_config.yaml \
  --model_ckpt <path>/checkpoints/trimodal/omni_vox2_trimodal/model_dict_best.pt \
  --codec_model_file <path>/codec_config/model.pth \
  --codec_config_file <path>/codec_config/config.yaml \
  --output_dir <path>/infer_results \
  --num_proc 4 --gpus "cuda:0 cuda:1 cuda:2 cuda:3"
```

## Training

All scenarios share ONE training script: `src/model_var/train.sh`
(entry `src/model_var/main.py`). Select the dataset/scenario with `--mode`:

```sh
cd src/model_var
bash train.sh --mode <mode> [--gpu_id '0,1' --batch_size 8 --lr 1e-4 \
    --continue_from <ckpt> --log_name <dir> --model_name <name>]
```

### Example: trimodal training (vox2 / ygd)

```sh
cd src/model_var
bash train.sh --mode trimodal_vox2 --gpu_id '0,1' --batch_size 8 --lr 1e-4
bash train.sh --mode trimodal_ygd --gpu_id '0,1' --batch_size 8 --lr 1e-4
```

## Data

The dataset (scp lists and codec features) is available at
[https://huggingface.co/datasets/AlexWu/tseomni](https://huggingface.co/datasets/AlexWu/tseomni).

## Acknowledgements

This project is built upon and thanks the following open-source projects:

- [LauraTSE](https://github.com/Beilong-Tang/lauraTSE_code): the codec and
  data scp format are mainly based on this repository.
- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio): the
  dataloader and training optimization are based on a simplified version of
  this repository, which also provides the segmentation (seg) model checkpoint.
