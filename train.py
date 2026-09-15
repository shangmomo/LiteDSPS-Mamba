from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from litedsps.data import build_dataset_and_collate
from litedsps.metrics import binary_metrics
from litedsps.model import LiteDSPSConfig, LiteDSPSMamba
from litedsps.utils import load_yaml, save_json, seed_everything


def build_model(cfg: Dict) -> LiteDSPSMamba:
    return LiteDSPSMamba(LiteDSPSConfig(**cfg["model"]))


def run_epoch(model, loader, device, optimizer=None) -> Tuple[float, Dict[str, float]]:
    train = optimizer is not None
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    losses, logits_all, labels_all = [], [], []
    for batch in loader:
        labels = batch["label"].to(device)
        if "visual_evidence" in batch:
            kwargs = {
                "visual_evidence": batch["visual_evidence"].to(device),
                "visual_mask": batch["visual_mask"].to(device),
                "audio_evidence": batch["audio_evidence"].to(device),
                "audio_mask": batch["audio_mask"].to(device),
            }
        else:
            kwargs = {
                "visual": batch["visual"].to(device),
                "visual_lengths": batch["visual_lengths"].to(device),
                "audio": batch["audio"].to(device),
                "audio_lengths": batch["audio_lengths"].to(device),
            }
        with torch.set_grad_enabled(train):
            logits = model(**kwargs)["logits"]
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(loss.detach().item())
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    metrics = binary_metrics(torch.cat(logits_all), torch.cat(labels_all))
    return sum(losses) / len(losses), metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default="runs/default")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    built = {s: build_dataset_and_collate(cfg["data"], s) for s in ["train", "valid", "test"]}
    loaders = {
        s: DataLoader(
            built[s][0],
            batch_size=cfg["training"]["batch_size"],
            shuffle=(s == "train"),
            num_workers=cfg["training"].get("num_workers", 0),
            collate_fn=built[s][1],
        )
        for s in built
    }

    model = build_model(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(cfg["training"]["lr_factor"]),
        patience=int(cfg["training"]["lr_patience"]),
    )

    best = float("-inf")
    history = []
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        tr_loss, tr = run_epoch(model, loaders["train"], device, optimizer)
        va_loss, va = run_epoch(model, loaders["valid"], device)
        scheduler.step(va["comp"])
        row = {"epoch": epoch, "train_loss": tr_loss, "valid_loss": va_loss, "train": tr, "valid": va}
        history.append(row)
        print(row)
        if va["comp"] > best:
            best = va["comp"]
            torch.save({"model": model.state_dict(), "config": cfg, "seed": args.seed}, out / "best.pt")

    # Paper protocol: select by validation Comp, then evaluate the selected checkpoint once on test with argmax.
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te = run_epoch(model, loaders["test"], device)
    result = {"seed": args.seed, "best_valid_comp": best, "test_loss": te_loss, "test": te, "history": history}
    save_json(result, out / "metrics.json")
    print("TEST", te)


if __name__ == "__main__":
    main()
