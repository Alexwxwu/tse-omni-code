# list_recepts — scp/list generation recipes

Per-dataset recipes that build the scp lists consumed by
`src/model_var/train.sh` / `dataload.py`. Scripts are grouped by dataset:

```
list_recepts/
├── paths.py            # centralized path management (edit here or set env vars)
├── scp_utils.py        # shared utilities: scp/csv IO, path join, ffmpeg conversion
├── lrs3/               # LRS3 EN recipes
├── vox2/               # VoxCeleb2 EN recipes
├── ygd/                # YGD recipes
├── librimix/           # LibriMix (wesep) recipes
├── check_audio.py      # shared: validate audio files in a scp
└── check_repeat.py     # shared: check repeated utterances
```

## Path management (paths.py)

Scripts no longer hardcode absolute paths at the top; they import from
[`paths.py`](paths.py). To run on a new machine, edit the root paths in
`paths.py` or set the corresponding environment variables:

```bash
# e.g. switch to a new user root
LAURA_USER_ROOT=/new/user/root python get_vox2_list.py
```

Available environment variables:

| Variable | Default |
| --- | --- |
| `LAURA_USER_ROOT` | `/mnt/users/hccl.local/wwu` |
| `LAURA_OUTPUT_ROOT` | `${USER_ROOT}/lauraTSE_code_refact/data_dpo` |
| `LAURA_CORPUS_ROOT` | `/mnt/Corpus-Upload` |
| `LAURA_LEGACY_ROOT` | `/home/export/base/sc100138/sc100138/online1` |
| `LAURA_WESEP_ROOT` | `/home/export/base/sc100135/sc100135/online1` |
| `LAURA_VSR_LOW_ROOT` | `${USER_ROOT}/vsr-low` |
| `LAURA_YGD_ROOT` | `${USER_ROOT}/YGD-mix-data/YGD` |
| `LAURA_AVTSE_ROOT` | `${USER_ROOT}/AVTSE_Momentum` |

## Code structure

Each script follows a "config + `main()`" structure and takes arguments via
`argparse` (dataset, split, output type, ...) instead of editing code to switch.
Shared IO logic is reused from [`scp_utils.py`](scp_utils.py).

## Typical pipeline (per dataset)

| Step | Script pattern | Output |
| --- | --- | --- |
| 1. mixture/target lists | `get_<ds>_list.py` | `mix_clean.scp`, `s1.scp` (from the mixture csv) |
| 2. enrollment list | `get_<ds>_enroll_list.py` | enrollment (reference) scp |
| 3. occlusion variant | `get_<ds>_vocc_list.py` | visual-occlusion mix lists |
| 4. FunCodec tokens | `replace_path_<ds>_codec.py` | codec token scp (`.npy`) |
| 5. VSR features | `replace_path_<ds>_vsr_feat.py` | VSR-frontend continuous feature scp |
| 6. VSR occlusion feats | `replace_path_<ds>_vsr_vocc_feat.py` | occluded VSR feature scp |
| 7. AV-HuBERT feats | `replace_path_<ds>_vsr_avhubert_continous_feat.py` | AV-HuBERT feature scp |

vox2 extras: `v2a_vox2.py` (video→audio), `replace_vocc2vsr_scp_direct.py`
(vocc→vsr scp conversion), `replace_path_vox2.py` (path-prefix replacement).

## Usage examples

```bash
# Generate VoxCeleb2 test_scale s1 list
python vox2/get_vox2_list.py --split test_scale --output s1

# Generate LRS3 occlusion mix_clean list
python lrs3/get_lrs3_vocc_list.py --output mix_clean

# Replace LRS3 VSR feature paths (verylong type)
python lrs3/replace_path_lrs3_vsr_feat.py --kind verylong

# Check for repeated paths in a scp
python check_repeat.py --input X
```
