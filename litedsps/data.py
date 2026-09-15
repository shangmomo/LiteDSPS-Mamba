from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def load_feature(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        arr = np.load(path)
        return torch.from_numpy(arr).float()
    if path.suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "tensor" in obj:
            obj = obj["tensor"]
        if not torch.is_tensor(obj):
            raise TypeError(f"{path} does not contain a tensor")
        return obj.float()
    raise ValueError(f"unsupported feature format: {path}")


class FeatureNormalizer:
    def __init__(self, stats_path: str | Path):
        stats = torch.load(stats_path, map_location="cpu")
        self.visual_mean = stats["visual_mean"].float()
        self.visual_std = stats["visual_std"].float()
        self.audio_mean = stats["audio_mean"].float()
        self.audio_std = stats["audio_std"].float()

    def visual(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.visual_mean) / self.visual_std

    def audio(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.audio_mean) / self.audio_std


class ManifestDataset(Dataset):
    """Dataset driven by CSV columns sample_id, split, label, visual_path, audio_path."""

    def __init__(self, manifest: str | Path, split: str, normalization: Optional[str | Path] = None):
        self.manifest = Path(manifest)
        self.base = self.manifest.parent
        with self.manifest.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split={split!r} in {self.manifest}")
        self.normalizer = FeatureNormalizer(normalization) if normalization else None

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else self.base / path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        visual = load_feature(self._resolve(row["visual_path"]))
        audio = load_feature(self._resolve(row["audio_path"]))
        if visual.ndim != 2 or audio.ndim != 2:
            raise ValueError("features must be 2-D [T, D]")
        if self.normalizer is not None:
            visual = self.normalizer.visual(visual)
            audio = self.normalizer.audio(audio)
        return {
            "sample_id": row["sample_id"],
            "visual": visual,
            "audio": audio,
            "label": int(row["label"]),
        }


class CachedEvidenceDataset(Dataset):
    """Dataset reading parameter-free K-token evidence saved by cache_evidence.py."""

    def __init__(self, cached_manifest: str | Path, split: str):
        self.manifest = Path(cached_manifest)
        self.base = self.manifest.parent
        with self.manifest.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]
        if not self.rows:
            raise ValueError(f"no rows for split={split!r} in {self.manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        p = Path(row["cache_path"])
        if not p.is_absolute():
            p = self.base / p
        obj = torch.load(p, map_location="cpu")
        return {
            "sample_id": row["sample_id"],
            "visual_evidence": obj["visual_evidence"].float(),
            "visual_mask": obj["visual_mask"].bool(),
            "audio_evidence": obj["audio_evidence"].float(),
            "audio_mask": obj["audio_mask"].bool(),
            "label": int(row["label"]),
        }


def collate_variable(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = len(batch)
    vdim = batch[0]["visual"].shape[1]
    adim = batch[0]["audio"].shape[1]
    vlen = torch.tensor([x["visual"].shape[0] for x in batch], dtype=torch.long)
    alen = torch.tensor([x["audio"].shape[0] for x in batch], dtype=torch.long)
    vmax, amax = int(vlen.max()), int(alen.max())
    visual = torch.zeros(b, vmax, vdim)
    audio = torch.zeros(b, amax, adim)
    for i, item in enumerate(batch):
        visual[i, : item["visual"].shape[0]] = item["visual"]
        audio[i, : item["audio"].shape[0]] = item["audio"]
    return {
        "sample_id": [x["sample_id"] for x in batch],
        "visual": visual,
        "visual_lengths": vlen,
        "audio": audio,
        "audio_lengths": alen,
        "label": torch.tensor([x["label"] for x in batch], dtype=torch.long),
    }


def collate_cached(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_id": [x["sample_id"] for x in batch],
        "visual_evidence": torch.stack([x["visual_evidence"] for x in batch]),
        "visual_mask": torch.stack([x["visual_mask"] for x in batch]),
        "audio_evidence": torch.stack([x["audio_evidence"] for x in batch]),
        "audio_mask": torch.stack([x["audio_mask"] for x in batch]),
        "label": torch.tensor([x["label"] for x in batch], dtype=torch.long),
    }


def build_dataset_and_collate(data_cfg: Dict[str, Any], split: str):
    mode = data_cfg.get("mode", "raw")
    if mode == "raw":
        ds = ManifestDataset(data_cfg["manifest"], split, normalization=data_cfg.get("normalization"))
        return ds, collate_variable
    if mode == "cached":
        ds = CachedEvidenceDataset(data_cfg["cached_manifest"], split)
        return ds, collate_cached
    raise ValueError(f"unknown data.mode: {mode}")
