"""Central model registry for LauraTSE model variants.

This replaces the old "comment / uncomment import" switching that used to
live inside ``build_model()`` in ``_funcodec.py``. Every available model is
declared explicitly below, so the mapping between a run script and the model
it builds is traceable.

Model selection priority in ``build_model(args)``:

1. ``args.model_name`` (CLI option ``--model_name``)
2. environment variable ``LAURA_MODEL_NAME``
3. the defaults (``DEFAULT_LAURA_MODEL`` / ``DEFAULT_LM_MODEL``)

Usage in shell scripts::

    python src/infer_visual_cue.py ... --model_name laura_tse
    # or
    export LAURA_MODEL_NAME=laura_tse

Model consolidation notes
-------------------------
- ``laura_tse`` (src/model_var/laura_tse.py) is THE main-line LauraTSE model.
  It covers audio-only / audio-visual (omni) / trimodal / transcript
  scenarios through the ``enroll_setting`` at inference time (training uses
  all 7 A/V/T cue combinations). The former variants ``laura_base``,
  ``av_unify``, ``front_align``, ``front_align_omni``,
  ``front_align_omni_trimodal``, ``front_align_transcript_only`` were merged
  into it and are kept below as aliases. Set ``use_lora: true`` in the yaml
  ``model_conf`` to reproduce the former ``front_align_omni_lora`` variant.
- Audio encoder of ALL models is a Conformer (``text_encoder`` /
  ``codec_encoder`` in the yaml config). The visual encoder consumes
  pre-extracted 512-dim VSR-frontend features through a 5x VisualConv1D stack.
- ``qwen2_omni_av`` absorbs the former ``qwen2_omni_av_lora`` (set
  ``use_lora: true``), ``qwen2_omni_av_small`` (set ``lm_variant: small``)
  and ``qwen2_omni`` (removed; only mixed audio+visual enrollment training
  is kept) via the yaml ``model_conf``.
"""

import importlib
import os

# model name -> (module path, class name)
MODEL_REGISTRY = {
    # ------------------------------------------------------------------
    # Main-line LauraTSE (--Laura 1). One model for all cue scenarios:
    # audio-only / AV-omni / trimodal / transcript, selected at inference
    # via enroll_setting. Aliases keep old --model_name values working.
    # ------------------------------------------------------------------
    "laura_tse": ("model_var.laura_tse", "LauraTSE"),
    # aliases of laura_tse (former standalone variants, now merged):
    "laura_base": ("model_var.laura_tse", "LauraTSE"),
    "laura_av_unify": ("model_var.laura_tse", "LauraTSE"),
    "laura_front_align": ("model_var.laura_tse", "LauraTSE"),
    "laura_front_align_omni": ("model_var.laura_tse", "LauraTSE"),
    "laura_front_align_omni_lora": ("model_var.laura_tse", "LauraTSE"),
    "laura_front_align_omni_trimodal": ("model_var.laura_tse", "LauraTSE"),
    "laura_front_align_transcript_only": ("model_var.laura_tse", "LauraTSE"),
    # robust to missing/corrupted visual cue (distinct architecture)
    "laura_front_align_switch": ("model_var.laura_model_only_clean_front_align_switch", "LauraTSE"),
    # ablation variant (distinct output/objective)
    "laura_vanilla_vtgt": ("model_var.laura_model_only_clean_vanilla_vtgt", "LauraTSE"),
    # ------------------------------------------------------------------
    # Gesture branch (LSTM over 30-dim gesture features), --Laura 1
    # ------------------------------------------------------------------
    "laura_gesture_rnn_omni": ("model_var.laura_model_only_clean_gesture_rnn_omni", "LauraTSE"),
    "laura_front_align_omni_gesture_trimodal": ("model_var.laura_model_only_clean_front_align_omni_gesture_trimodal", "LauraTSE"),
    # ------------------------------------------------------------------
    # LLM-backbone variants, selected with --Laura 0
    # ------------------------------------------------------------------
    "qwen2_omni_av": ("model_var.qwen2_model_omni_av", "QwenLM"),
    # aliases of qwen2_omni_av (merged; use model_conf use_lora / lm_variant):
    "qwen2_omni": ("model_var.qwen2_model_omni_av", "QwenLM"),
    "qwen2_omni_av_lora": ("model_var.qwen2_model_omni_av", "QwenLM"),
    "qwen2_omni_av_small": ("model_var.qwen2_model_omni_av", "QwenLM"),
    "llama_omni_av_small": ("model_var.llama_model_omni_av_small", "LlamaLM"),
}

# Defaults that preserve the most recent working configuration
DEFAULT_LAURA_MODEL = "laura_tse"
DEFAULT_LM_MODEL = "qwen2_omni_av"

ENV_VAR_NAME = "LAURA_MODEL_NAME"


def resolve_model_name(args, laura: bool) -> str:
    """Resolve which registered model to build.

    Args:
        args: argument namespace; may contain ``model_name``.
        laura: True for the LauraTSE branch (--Laura 1),
               False for the LLM-backbone branch (--Laura 0).
    """
    name = getattr(args, "model_name", None)
    if not name:
        name = os.environ.get(ENV_VAR_NAME)
    if not name:
        name = DEFAULT_LAURA_MODEL if laura else DEFAULT_LM_MODEL
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name '{name}'. Available models: {sorted(MODEL_REGISTRY)}"
        )
    return name


def get_model_class(args, laura: bool):
    """Lazily import and return the model class selected for this run."""
    name = resolve_model_name(args, laura)
    module_path, class_name = MODEL_REGISTRY[name]
    print(f"[model_registry] building '{name}' from {module_path}.{class_name}")
    return getattr(importlib.import_module(module_path), class_name)
