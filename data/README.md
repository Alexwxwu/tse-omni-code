# data — Data Preparation & Test-set Data

This directory holds the LauraTSE data-preparation code and the **test-set**
data artifacts (scp lists and FunCodec features). `data/` contains only
test-set data; train/validation data is produced by the pipeline in
[`data_preparation_vox2_en/`](data_preparation_vox2_en/).

## Directory layout

```
data/
├── data_preparation_vox2_en/   # Full pipeline: raw VoxCeleb2 -> train/test data
│   ├── step1_simulate_mixture/ # Mixture simulation (audio + visual occlusion)
│   ├── step2_enroll_scp/       # Enrollment scp generation
│   ├── step3_mix_s1_scp/       # mix / s1 scp generation
│   ├── step4_gt_funcodec/      # GT codec token generation
│   └── step5_vsr_feat_scp/     # VSR feature scp generation
├── list_recepts/               # Per-dataset scp generation scripts (paths centralized in paths.py)
├── utils/                      # Data-prep utilities (FunCodec encode/decode, scp extraction, ...)
├── dump/wavs/                  # Test-set scp lists (mix_clean / s1 / aux_s1 / vocc)
├── funcodec/                   # Test-set codec features (.npy + all.scp)
├── export_*_funcodec.sh        # Entry scripts to generate codec tokens
├── generate_list.py / .sh      # Generate scp lists
└── generate_codec.sh           # Entry to generate codec
```

## Data preparation steps

### Stage A: generate mixture data from raw data (`data_preparation_vox2_en/`)

Using VoxCeleb2 as an example, generate the mixture data (train/test) needed
from raw mp4/audio:

| Step | Script | Description |
| --- | --- | --- |
| 1. Simulate mixture | `step1_simulate_mixture/run_scale_data.sh` | Main flow; calls the 3 scripts below |
| 1a. Create mixture list | `step1_simulate_mixture/1_create_mixture_list.py` | Extract audio from mp4, select speakers, build mixture list |
| 1b. Generate mixture audio | `step1_simulate_mixture/2_create_mixture.py` | Generate `mix/s1/s2` mixture audio (with SNR mixing) |
| 1c. Visual occlusion list | `step1_simulate_mixture/3_create_LowQuality_visual_list.py` | Build occlusion/low-quality visual list (with `Visual_perturb.py`, `tools.py`) |
| 2. Enrollment scp | `step2_enroll_scp/get_vox2_enroll_list_621.py` | Generate `aux_s1.scp` (enrollment/reference audio) |
| 3. mix/s1 scp | `step3_mix_s1_scp/get_vox2_list_621.py` | Generate `mix_clean.scp`, `s1.scp` |
| 4. GT codec | `step4_gt_funcodec/export_librispeech_funcodec_normalize.sh` | Generate GT codec tokens with FunCodec |
| 5. VSR feature scp | `step5_vsr_feat_scp/replace_path_vox2_vsr_feat.py` | Generate VSR continuous feature scp |

### Stage B: generate scp & codec features (`data/` top-level + `list_recepts/` + `utils/`)

For existing audio data (including Stage A outputs), generate the scp lists and
features required by the model:

| Step | Script | Description |
| --- | --- | --- |
| 1. Generate scp lists | `list_recepts/*` or `generate_list.py` | Build `mix_clean.scp`, `s1.scp`, `aux_s1.scp` from mixture csv |
| 2. Generate codec tokens | `export_*_funcodec.sh` + `utils/2_export_libri2mix_funcodec.py` | Encode wav to `.npy` with FunCodec, producing `funcodec/<dataset>/<split>/` with `all.scp`, `0.scp`, `0_shape.scp` |
| 3. Replace feature paths | `list_recepts/replace_path_*.py` | Replace `all.scp` paths with VSR / AV-HuBERT feature paths |

### Stage C: data artifacts

- `dump/wavs/`: test-set scp lists (`mix_clean.scp`, `s1.scp`, `aux_s1.scp`, `mix_clean_vocc.scp`)
- `funcodec/`: test-set codec features (`.npy` files + `all.scp` / `all_shape.scp`)

## Path management

Paths in `list_recepts/` are centralized in
[`list_recepts/paths.py`](list_recepts/paths.py). To run on a new machine, edit
the root paths there or set the corresponding environment variables
(`LAURA_USER_ROOT`, `LAURA_OUTPUT_ROOT`, ...). Paths in
`data_preparation_vox2_en/` are set at the top of each script; adjust them to
your environment before running.
