from __future__ import annotations

import argparse
import torch
from torch.utils.data import DataLoader

from litedsps.data import build_dataset_and_collate
from litedsps.metrics import binary_metrics
from litedsps.model import LiteDSPSConfig, LiteDSPSMamba


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    model = LiteDSPSMamba(LiteDSPSConfig(**cfg["model"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds, collate = build_dataset_and_collate(cfg["data"], args.split)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], collate_fn=collate)
    logits, labels = [], []
    with torch.no_grad():
        for b in loader:
            if "visual_evidence" in b:
                kwargs = {
                    "visual_evidence": b["visual_evidence"].to(device),
                    "visual_mask": b["visual_mask"].to(device),
                    "audio_evidence": b["audio_evidence"].to(device),
                    "audio_mask": b["audio_mask"].to(device),
                }
            else:
                kwargs = {
                    "visual": b["visual"].to(device),
                    "visual_lengths": b["visual_lengths"].to(device),
                    "audio": b["audio"].to(device),
                    "audio_lengths": b["audio_lengths"].to(device),
                }
            out = model(**kwargs)["logits"]
            logits.append(out.cpu())
            labels.append(b["label"])
    print(binary_metrics(torch.cat(logits), torch.cat(labels)))


if __name__ == "__main__":
    main()
