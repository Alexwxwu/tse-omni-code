#!/usr/bin/env python3
"""
Standalone AV-HuBERT token / embedding extractor.

Read a scp list, and for each utterance extract:
  1) discrete tokens  -> int array  (argmax over the first codebook logits)
  2) continuous emb   -> float array (AV-HuBERT last-layer encoder output)

Both are saved as .npy files under two separate output dirs, mirroring the
sub-path of the source video ( .../dev/<spk>/<vid>/<utt>.npy ).

This directory is self-contained: it ships its own `av2unit` package (the
AV-HuBERT model + task code). The only external dependency is `fairseq`,
which is located via --fairseq-path (default: the av2av repo copy).

Usage:
    PYTHONPATH=<fairseq> python extract_avhubert.py --scp <mix_clean.scp> ...
See run_extract.sh for a ready-to-run example.
"""

import os
import sys
import argparse
import numpy as np

# numpy>=1.20 removed np.complex; older fairseq/avhubert code may reference it.
if not hasattr(np, "complex"):
    np.complex = complex

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Path setup: make `fairseq` and the local `av2unit` importable.
# --------------------------------------------------------------------------- #
def setup_paths(fairseq_path: str) -> None:
    here = os.path.dirname(os.path.abspath(__file__))  # contains ./av2unit
    for p in (fairseq_path, here):
        if p and p not in sys.path:
            sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# Model / task loading
# --------------------------------------------------------------------------- #
def load_av2unit_task(model_path: str, modalities: str):
    """Load the mAV-HuBERT 'unit' checkpoint. We keep its *task* object because
    it provides (a) the dataset used for feature loading / collating and
    (b) the `inference_vanilla` routine that yields tokens + embeddings."""
    from fairseq import checkpoint_utils
    import av2unit.task  # noqa: F401  (registers task + model with fairseq)

    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
    task.cfg.modalities = modalities.split(",")
    task.load_dataset()
    return task


def load_vanilla_avhubert(model_path: str, modalities: str, use_cuda: bool):
    """Load the vanilla (pretrained) AV-HuBERT checkpoint that actually runs the
    forward pass to produce the tokens / embeddings."""
    from fairseq import checkpoint_utils
    import av2unit.task  # noqa: F401  (ensure model/task registered)

    models, cfg, _ = checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = models[0]
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# scp parsing -> (uttid, video_path, audio_path)
# --------------------------------------------------------------------------- #
def resolve_paths(line: str, args):
    """Support two scp layouts:

    (A) explicit paths:  `<uttid> <video.mp4> <audio.wav>`
    (B) uttid only (or uttid + extra wav): reconstruct single-speaker paths
        from the uttid, e.g.
        id00708_s8d-CyvzvE0_00258_id06114_oiot7gTiKgE_00431
          spk = uttid[0:7]   -> id00708
          vid = uttid[8:19]  -> s8d-CyvzvE0
          utt = uttid[20:25] -> 00258
        video = <visual_dir>/<partition>/<spk>/<vid>/<utt>.mp4
        audio = <audio_dir>/<partition>/<spk>/<vid>/<utt>.wav
    """
    parts = line.split()
    uttid = parts[0]

    if len(parts) >= 3 and parts[1].endswith(".mp4") and parts[2].endswith(".wav"):
        return uttid, parts[1], parts[2]

    spk = uttid[0:7]
    vid = uttid[8:19]
    utt = uttid[20:25]
    video = os.path.join(args.visual_dir, args.partition, spk, vid, utt + ".mp4")
    audio = os.path.join(args.audio_dir, args.partition, spk, vid, utt + ".wav")
    return uttid, video, audio


def out_npy(save_dir: str, video_path: str) -> str:
    """Mirror the trailing path components of the video, e.g.
    .../dev/id00708/s8d-CyvzvE0/00258.mp4 -> <save_dir>/dev/id00708/s8d-CyvzvE0/00258.npy
    """
    rel = "/".join(video_path.split(".mp4")[0].split("/")[-4:])
    return os.path.join(save_dir, rel + ".npy")


# --------------------------------------------------------------------------- #
# Feature build + single-utterance extraction
# --------------------------------------------------------------------------- #
def build_sample(task, video_path: str, audio_path: str):
    dataset = task.dataset
    video_feats, audio_feats = dataset.load_feature((video_path, audio_path))

    video_feats = (
        torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
    )
    audio_feats = (
        torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None
    )

    if audio_feats is not None and dataset.normalize and "audio" in dataset.modalities:
        with torch.no_grad():
            audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

    collated_videos = (
        dataset.collater_audio([video_feats], len(video_feats))[0]
        if video_feats is not None else None
    )
    collated_audios = (
        dataset.collater_audio([audio_feats], len(audio_feats))[0]
        if audio_feats is not None else None
    )

    return {"source": {"video": collated_videos, "audio": collated_audios}}


@torch.no_grad()
def extract_one(task, model, video_path: str, audio_path: str):
    sample = build_sample(task, video_path, audio_path)
    pred, cont_emb = task.inference_vanilla(model, sample)
    pred_ids = pred.detach().cpu().numpy().astype(np.int64)      # discrete tokens
    cont_emb = cont_emb.detach().cpu().float().numpy()           # continuous emb
    return pred_ids, cont_emb


def save_npy(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(args):
    setup_paths(args.fairseq_path)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    print(f"[info] device={device}  modalities={args.modalities}")

    task = load_av2unit_task(args.av2unit_path, args.modalities)
    model = load_vanilla_avhubert(args.avhubertvanilla_path, args.modalities, use_cuda)

    with open(args.scp, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]
    print(f"[info] {len(lines)} utterances from {args.scp}")

    try:
        from tqdm import tqdm
        iterator = tqdm(lines, desc="extract", ncols=80)
    except ImportError:
        iterator = lines

    n_ok = n_skip = n_fail = 0
    for line in iterator:
        uttid, video_path, audio_path = resolve_paths(line, args)
        tok_path = out_npy(args.token_save_dir, video_path)
        emb_path = out_npy(args.emb_save_dir, video_path)

        # resume: skip when both outputs already exist
        if not args.overwrite and os.path.exists(tok_path) and os.path.exists(emb_path):
            n_skip += 1
            continue

        if not os.path.exists(video_path) or (
            "audio" in args.modalities and not os.path.exists(audio_path)
        ):
            print(f"[warn] missing input, skip: {video_path} | {audio_path}")
            n_fail += 1
            continue

        try:
            pred_ids, cont_emb = extract_one(task, model, video_path, audio_path)
            save_npy(tok_path, pred_ids)
            save_npy(emb_path, cont_emb)
            n_ok += 1
        except Exception as e:
            print(f"[error] {uttid}: {e}")
            n_fail += 1

    print(f"[done] ok={n_ok} skipped={n_skip} failed={n_fail}")


def cli_main():
    p = argparse.ArgumentParser(description="Standalone AV-HuBERT token/emb extractor")

    # input / output
    p.add_argument("--scp", type=str, required=True, help="input scp list")
    p.add_argument("--token-save-dir", type=str, required=True, help="dir for discrete tokens (.npy)")
    p.add_argument("--emb-save-dir", type=str, required=True, help="dir for continuous emb (.npy)")

    # path reconstruction (layout B)
    p.add_argument("--visual-dir", type=str, default="", help="root dir of lip videos")
    p.add_argument("--audio-dir", type=str, default="", help="root dir of clean audios")
    p.add_argument("--partition", type=str, default="dev", help="video/audio partition subdir")

    # models
    p.add_argument("--av2unit-path", type=str, required=True, help="mAV-HuBERT unit ckpt (task/dataset)")
    p.add_argument("--avhubertvanilla-path", type=str, required=True, help="vanilla AV-HuBERT ckpt (forward)")
    p.add_argument("--fairseq-path", type=str,
                   default="/mnt/users/hccl.local/wwu/av2av/fairseq",
                   help="path to fairseq source (dir containing the fairseq package)")

    # behaviour
    p.add_argument("--modalities", type=str, default="video",
                   choices=["audio,video", "audio", "video"],
                   help="input modalities fed to the model")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--overwrite", action="store_true", help="recompute even if outputs exist")
    p.add_argument("--limit", type=int, default=-1, help="only process first N lines (-1 = all)")

    args = p.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
