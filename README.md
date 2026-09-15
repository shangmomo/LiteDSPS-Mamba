# LiteDSPS-Mamba

Reference implementation for **“LiteDSPS-Mamba: Decoupling Behavioral Evidence from Temporal Computation for Long-Form Audio-Visual Depression Detection.”**

LiteDSPS-Mamba reduces learned temporal computation before Mamba2 by converting complete facial/acoustic streams into a fixed budget of ordered behavioral-evidence tokens. It then applies Mamba2 only to a partial-width dynamic subspace, while a bypass subspace retains direct projected evidence. The temporal operator is shared across modalities and scan directions.

> This repository is reconstructed from the manuscript specification. Dataset files are not redistributed. Use the released D-Vlog/LMVD features according to their original licenses and access conditions.

## Architecture

For each modality, an input sequence `X ∈ R^(T×d)` is reduced to `K=96` ordered regions. Each evidence token concatenates four per-feature statistics:

- regional mean `μ`
- `log1p(std)` dispersion
- mean absolute first-order difference
- maximum absolute first-order difference

The resulting `4d` evidence is projected to 128 dimensions and split into:

- **96-D dynamic path** → shared bidirectional Mamba2
- **32-D bypass path** → retained without state-space transformation

The two paths are reunited, followed by mask-aware mean-max pooling per modality and late concatenation for binary classification.

Default paper settings:

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
├── configs/
│   ├── dvlog.yaml
│   └── lmvd.yaml
├── litedsps/
│   ├── data.py
│   ├── evidence.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── scripts/
│   └── run_three_seeds.sh
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

For CPU-only unit tests, install the lightweight dependencies:

```bash
pip install -r requirements.txt pytest
pytest -q
```

The CPU test path uses a shape-compatible temporal stub only for CI. It is **not** used by the paper configurations.

## Data

This repository intentionally does not redistribute D-Vlog or LMVD features.

Create a CSV manifest for each dataset with these columns:

```csv
sample_id,split,label,visual_path,audio_path
0001,train,0,features/0001_visual.npy,features/0001_audio.npy
0002,valid,1,features/0002_visual.npy,features/0002_audio.npy
0003,test,0,features/0003_visual.npy,features/0003_audio.npy
```

Each feature file must contain a 2-D array shaped `[T, D]`.

Expected feature dimensions:

| Dataset | Visual | Acoustic |
|---|---:|---:|
| D-Vlog | 136-D facial landmarks | 25-D OpenSMILE |
| LMVD | 136-D facial landmarks | 128-D VGGish |

Set `data.manifest` in `configs/dvlog.yaml` or `configs/lmvd.yaml` to the corresponding manifest. Preserve the paper's published data partitions rather than creating a new random split.

### Training-set normalization

The manuscript estimates normalization statistics on the training set only and fixes them for validation/test. Compute them once:

```bash
python compute_normalization.py \
  --manifest data/dvlog/manifest.csv \
  --output data/dvlog/normalization.pt
```

Use the analogous paths for LMVD. The YAML files already point to `normalization.pt`.

### Optional deterministic evidence cache

The behavioral evidence transform is parameter-free, so it can be computed once and reused across epochs, matching the repeated-optimization protocol in the manuscript:

```bash
python cache_evidence.py \
  --manifest data/dvlog/manifest.csv \
  --normalization data/dvlog/normalization.pt \
  --output-dir data/dvlog/evidence_cache
```

Then set:

```yaml
data:
  mode: cached
  cached_manifest: data/dvlog/evidence_cache/manifest_cached.csv
```

The full four-statistic evidence is cached. Controlled `w/o dynamic differences` and `AvgPool-96` variants are applied after loading, so the same cache can be reused.

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

Checkpoint selection follows the manuscript: the best checkpoint maximizes

```text
Comp = (Accuracy + Positive Precision + Positive Recall + Positive F1) / 4
```

on validation data. The selected checkpoint is then evaluated once on test data using argmax predictions, with no threshold tuning.

## Evaluation

```bash
python evaluate.py \
  --checkpoint runs/dvlog/seed1/best.pt \
  --split test
```

## Controlled ablations

The principal ablations can be reproduced by editing the `model` section of a config:

| Ablation | Config change |
|---|---|
| Full-width | `dynamic_dim: 128` |
| w/o direction sharing | `share_directions: false` |
| w/o modality sharing | `share_modalities: false` |
| Unidirectional | `bidirectional: false` |
| Mean-only aggregation | `aggregation: mean` |
| w/o dynamic differences | `use_dynamic_diff: false` |
| AvgPool-96 control | `evidence_mode: avgpool96` |

`AvgPool-96` keeps the same `4d` projector input width: the regional mean occupies the first statistic slot and the remaining three slots are zeroed.

## Profiling

A CUDA profiling helper is included for end-to-end LiteDSPS inference:

```bash
python profile.py --visual-dim 136 --audio-dim 25 --lengths 512 4096 10000 50000 100000
```

The manuscript's comparison against CAF-Mamba used the same wrapper/environment for both models. This repository includes only LiteDSPS-Mamba, so cross-model numbers should be reproduced only after integrating the exact comparison implementation under the same software/hardware environment.

## Notes on evidence construction

The manuscript defines first-order changes before temporal reduction. In this reference implementation, a zero change is prepended at the first time step so change statistics use the same contiguous ordered-region boundaries as the raw features. This convention does not introduce future information and keeps `K_m = min(T_m, K)` valid regions for short sequences.

## Citation

If this code is useful for your work, please cite the accompanying manuscript. Bibliographic metadata can be updated after publication.

```bibtex
@inproceedings{shang2027litedsps,
  title={LiteDSPS-Mamba: Decoupling Behavioral Evidence from Temporal Computation for Long-Form Audio-Visual Depression Detection},
  author={Shang, Yongzheng and Wang, Junjie and Xue, Nian and Cui, Yukun and Wang, Lisheng and Cao, Junyuan and Li, Zhen and Wang, Zhiqiang},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2027}
}
```

## Acknowledgment

The implementation depends on the official Mamba/Mamba2 package from the state-spaces project. Dataset access and usage remain governed by the original D-Vlog and LMVD releases.
