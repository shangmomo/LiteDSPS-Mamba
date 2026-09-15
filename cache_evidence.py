from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch

from litedsps.data import FeatureNormalizer, load_feature
from litedsps.evidence import behavioral_evidence_single


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--normalization", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--k", type=int, default=96)
    args = p.parse_args()

    manifest = Path(args.manifest)
    normalizer = FeatureNormalizer(args.normalization)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    cached_rows = []
    for i, row in enumerate(rows):
        vp = Path(row["visual_path"]); ap = Path(row["audio_path"])
        if not vp.is_absolute(): vp = manifest.parent / vp
        if not ap.is_absolute(): ap = manifest.parent / ap
        visual = normalizer.visual(load_feature(vp))
        audio = normalizer.audio(load_feature(ap))
        ve, vm = behavioral_evidence_single(visual, k=args.k)
        ae, am = behavioral_evidence_single(audio, k=args.k)
        safe_id = row["sample_id"].replace(os.sep, "_").replace("/", "_")
        cache_path = out_dir / f"{i:06d}_{safe_id}.pt"
        torch.save({"visual_evidence": ve, "visual_mask": vm, "audio_evidence": ae, "audio_mask": am}, cache_path)
        cached_rows.append({"sample_id": row["sample_id"], "split": row["split"], "label": row["label"], "cache_path": cache_path.name})

    cached_manifest = out_dir / "manifest_cached.csv"
    with cached_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "split", "label", "cache_path"])
        w.writeheader(); w.writerows(cached_rows)
    print(f"cached {len(rows)} samples -> {cached_manifest}")


if __name__ == "__main__":
    main()
