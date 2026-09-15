from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from litedsps.data import load_feature


def update(sum_, sumsq, count, x):
    x = x.double()
    if sum_ is None:
        sum_ = torch.zeros(x.shape[1], dtype=torch.float64)
        sumsq = torch.zeros_like(sum_)
    return sum_ + x.sum(0), sumsq + x.square().sum(0), count + x.shape[0]


def finish(sum_, sumsq, count):
    mean = sum_ / count
    var = (sumsq / count - mean.square()).clamp_min(0)
    std = torch.sqrt(var).clamp_min(1e-6)
    return mean.float(), std.float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    manifest = Path(args.manifest)
    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    if not rows:
        raise ValueError("manifest has no train rows")
    vs = vss = aas = aass = None
    vc = ac = 0
    for row in rows:
        vp = Path(row["visual_path"]); ap = Path(row["audio_path"])
        if not vp.is_absolute(): vp = manifest.parent / vp
        if not ap.is_absolute(): ap = manifest.parent / ap
        vs, vss, vc = update(vs, vss, vc, load_feature(vp))
        aas, aass, ac = update(aas, aass, ac, load_feature(ap))
    vm, vst = finish(vs, vss, vc)
    am, ast = finish(aas, aass, ac)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"visual_mean": vm, "visual_std": vst, "audio_mean": am, "audio_std": ast}, out)
    print(f"saved training-only normalization statistics to {out}")


if __name__ == "__main__":
    main()
