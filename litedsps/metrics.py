from __future__ import annotations

from typing import Dict

import torch


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    labels = labels.long()
    acc = (pred == labels).float().mean().item()
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    comp = (acc + precision + recall + f1) / 4.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "comp": comp}
