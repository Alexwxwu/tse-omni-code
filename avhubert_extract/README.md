# avhubert_extract — standalone AV-HuBERT token / emb extraction tool

A minimal runnable version extracted from the `av2av` repo: it reads an scp
list and, for each utterance, extracts both

- **discrete token** (argmax of the first codebook logits, int array)
- **continuous emb** (AV-HuBERT last encoder layer output, float array)

and saves each as `.npy`.

## Directory layout
```
avhubert_extract/
├── extract_avhubert.py   # main script (single entry)
├── run_extract.sh        # one-click launcher (paths are set here)
├── README.md
└── av2unit/              # bundled AV-HuBERT model + task code (copied from av2av)
    ├── task.py
    └── avhubert/         # hubert.py / hubert_dataset.py / hubert_pretraining.py / ...
```
The only external dependency is `fairseq`, specified via `--fairseq-path`
(default points to `av2av/fairseq`); fairseq itself is not copied.

## Run
```bash
cd /mnt/users/hccl.local/wwu/lauraTSE_code_backup/avhubert_extract
bash run_extract.sh
```
First edit `SCP / VISUAL_DIR / AUDIO_DIR / the two model paths / output dir /
gpu_id` at the top of `run_extract.sh`.

## Two scp formats (auto-detected)
1. **Explicit paths**: `<uttid> <video.mp4> <audio.wav>`
2. **uttid only** (or uttid + another wav, e.g. `mix_clean.scp`): rebuild the
   single-speaker path by slicing the uttid
   ```
   id00708_s8d-CyvzvE0_00258_id06114_oiot7gTiKgE_00431
   spk = uttid[0:7]   -> id00708
   vid = uttid[8:19]  -> s8d-CyvzvE0
   utt = uttid[20:25] -> 00258
   video = <visual_dir>/<partition>/<spk>/<vid>/<utt>.mp4
   audio = <audio_dir>/<partition>/<spk>/<vid>/<utt>.wav
   ```

## Output
Mirrors the last 4 levels of the video path:
```
<token_save_dir>/dev/<spk>/<vid>/<utt>.npy   # discrete token (int64)
<emb_save_dir>/dev/<spk>/<vid>/<utt>.npy     # continuous emb (float)
```

## Key arguments
| Argument | Description |
|---|---|
| `--scp` | Input scp list |
| `--av2unit-path` | mAV-HuBERT unit ckpt (provides task/dataset), e.g. `mavhubert_large_noise.pt` |
| `--avhubertvanilla-path` | Vanilla AV-HuBERT ckpt used for the forward pass, e.g. `large_vox_iter5.pt` |
| `--modalities` | Modalities fed to the model: `video` (same as the original pipeline) or `audio,video` |
| `--fairseq-path` | fairseq source dir (the level containing the fairseq package) |
| `--overwrite` | Recompute even if output exists |
| `--limit N` | Only process the first N entries, for quick validation |
| `--cpu` | Force CPU |

## Correspondence with the original av2av code
- Extraction core = `av2unit/task.py::AVHubertUnitPretrainingTask.inference_vanilla()`,
  returning `(pred, x)`, i.e. discrete token and continuous emb.
- Forward pass uses `AVHubertModel.extract_finetune()` (a standalone method that
  does not go through the training branch in `forward()`).
- Default `--modalities video` is equivalent to the video-only behavior of
  `process_avhubert_vanilla_av2unit()` with `audio=None`; for true audio-visual
  joint tokens, use `--modalities audio,video`.
- Supports resuming: skips automatically when both outputs already exist.

## Dependencies
`torch`, `numpy`, `opencv-python (cv2)`, `python_speech_features`, `scipy`,
`fairseq` (source), optional `tqdm`.
