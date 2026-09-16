# LiteDSPS-Mamba

Reference implementation for **“LiteDSPS-Mamba: Decoupling Behavioral Evidence from Temporal Computation for Long-Form Audio-Visual Depression Detection.”**

LiteDSPS-Mamba reduces learned temporal computation before Mamba2 by converting complete facial/acoustic streams into a fixed budget of ordered behavioral-evidence tokens. It then applies Mamba2 only to a partial-width dynamic subspace, while a bypass subspace retains direct projected evidence. The temporal operator is shared across modalities and scan directions.

This repository contains the original R5 direction-shared main-model experiment source supplied for LiteDSPS-Mamba, together with a modular implementation for easier inspection and reuse. Dataset files are not redistributed; use the released D-Vlog/LMVD features according to their original licenses and access conditions.

## Original experiment source

The exact uploaded R5 source is preserved as a byte-verified split ZIP under:

```text
experiments/original/
├── LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part01
├── LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part02
├── LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part03
├── LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part04
├── LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part05
├── LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part06
├── LiteDSPS_Mamba_R5_DirShare_Deterministic_Final_20260916.zip.part07
└── reassemble.py
```

Reassemble and verify the archive with:

```bash
python experiments/original/reassemble.py
```

Expected ZIP SHA-256:

```text
36df017861e61bbb3600591227ca6b6ac04dd72d90d426b42b05f15b05096dbb
```

The archive contains the exact source file:

```text
LiteDSPS_Mamba_R5_DirShare_Deterministic_Final(20260916-004726).py
```

Raw-source SHA-256:

```text
efc05bb4613bb27a2a5bf1c1f09707fd9bc8934bfdf02fefbd18c443a812ee48
```

The modular implementation under `litedsps/` is aligned with the original main-model evidence construction. In particular, the first-order difference sequence has length `T-1` and is independently adaptively pooled to the same effective evidence budget as the raw-feature statistics.

## Architecture

For each modality, an input sequence `X ∈ R^(T×d)` is reduced to `K=96` ordered evidence positions. Each evidence token concatenates four per-feature statistics:

- regional mean `μ`
- `log1p(std)` dispersion
- mean absolute first-order difference
- maximum absolute first-order difference

The resulting `4d` evidence is projected to 128 dimensions and split into:

- **96-D dynamic path** → shared bidirectional Mamba2
- **32-D bypass path** → retained without state-space transformation

The two paths are reunited, followed by mask-aware mean-max pooling per modality and late concatenation for binary classification.

Default main-model settings:

| Item | Setting |
|---|---:|
| Evidence budget | `K = 96` |
| Projected width | `128` |
| Dynamic / bypass | `96 / 32` |
| Mamba2 depth | `1` |
| `d_state` | `32` |
| `d_conv` | `4` |
| `expand` | `2` |
| Dropout | `0.30` |
| Batch size | `16` |
| AdamW learning rate | `5e-5` |
| Weight decay | `5e-3` |
| Max epochs | `70` |
| LR scheduler | ReduceLROnPlateau, factor `0.6`, patience `5` |

## Repository layout

```text
LiteDSPS-Mamba/
├── experiments/original/   # byte-verified original R5 source archive
├── configs/
│   ├── dvlog.yaml
│   └── lmvd.yaml
├── litedsps/
│   ├── data.py
│   ├── evidence.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── scripts/run_three_seeds.sh
├── tests/
├── train.py
├── evaluate.py
├── profile.py
├── requirements.txt
└── requirements-mamba.txt
```

## Installation

The paper model uses the official `mamba_ssm.Mamba2` block and therefore requires a CUDA-compatible Mamba installation.

```bash
git clone https://github.com/shangmomo/LiteDSPS-Mamba.git
cd LiteDSPS-Mamba

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-mamba.txt
```

For CPU-only unit tests:

```bash
pip install -r requirements.txt pytest
pytest -q
```

The CPU test path uses a shape-compatible temporal stub only for CI. It is not used by the paper configurations.

## Data

This repository intentionally does not redistribute D-Vlog or LMVD features.

The modular loader uses a CSV manifest with:

```csv
sample_id,split,label,visual_path,audio_path
0001,train,0,features/0001_visual.npy,features/0001_audio.npy
0002,valid,1,features/0002_visual.npy,features/0002_audio.npy
0003,test,0,features/0003_visual.npy,features/0003_audio.npy
```

Expected feature dimensions:

| Dataset | Visual | Acoustic |
|---|---:|---:|
| D-Vlog | 136-D facial landmarks | 25-D OpenSMILE |
| LMVD | 136-D facial landmarks | 128-D VGGish |

Preserve the published data partitions rather than creating a new random split.

### Training-set normalization

Normalization statistics are estimated from the training split only and reused for validation/test:

```bash
python compute_normalization.py \
  --manifest data/dvlog/manifest.csv \
  --output data/dvlog/normalization.pt
```

### Optional deterministic evidence cache

The parameter-free evidence transform can be cached before repeated optimization:

```bash
python cache_evidence.py \
  --manifest data/dvlog/manifest.csv \
  --normalization data/dvlog/normalization.pt \
  --output-dir data/dvlog/evidence_cache
```

## Training

D-Vlog:

```bash
python train.py \
  --config configs/dvlog.yaml \
  --seed 1 \
  --output runs/dvlog/seed1
```

LMVD:

```bash
python train.py \
  --config configs/lmvd.yaml \
  --seed 1 \
  --output runs/lmvd/seed1
```

Three fixed optimization seeds:

```bash
bash scripts/run_three_seeds.sh configs/dvlog.yaml dvlog
bash scripts/run_three_seeds.sh configs/lmvd.yaml lmvd
```

Checkpoint selection uses

```text
Comp = (Accuracy + Positive Precision + Positive Recall + Positive F1) / 4
```

on validation data. The selected checkpoint is evaluated once on test data using argmax predictions, without threshold tuning.

## Evaluation

```bash
python evaluate.py \
  --checkpoint runs/dvlog/seed1/best.pt \
  --split test
```

## Controlled ablations

The modular implementation exposes structural/representation controls including full-width state modeling, direction/modality sharing, unidirectional processing, mean-only aggregation, removal of dynamic differences, and AvgPool-96.

The archived R5 source is the final direction-shared **main-model** script. It does not contain the separate Evidence-only, GRU, or Mamba no-bypass control scripts. Those should be released separately if direct reproduction of those controls is required.

## Profiling

```bash
python profile.py --visual-dim 136 --audio-dim 25 --lengths 512 4096 10000 50000 100000
```

## Evidence-construction note

For a sequence of length `T`, raw-feature mean/dispersion statistics are adaptively pooled to `K_m = min(T, K)` positions. For `T>1`, first-order absolute differences form a separate length-`(T-1)` sequence and are independently adaptively average/max pooled to the same `K_m` positions. For `T=1`, difference statistics are zero. This matches the supplied original R5 source and the paper's evidence definition.

## Citation

Bibliographic metadata should be updated to match the final submitted author list and publication record.

## Acknowledgment

The implementation depends on the official Mamba/Mamba2 package from the state-spaces project. Dataset access and usage remain governed by the original D-Vlog and LMVD releases.
