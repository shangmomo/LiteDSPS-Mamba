from __future__ import annotations

from typing import Dict

import torch


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=-1)
    labels = labels.long()
    tp = ((pred == 1) & (labels == 1)).sum().item()
    fp = ((pred == 1) & (labels == 0)).sum().item()
    fn = ((pred == 0) & (labels == 1)).sum().item()
    acc = (pred == labels).float().mean().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    comp = (acc + precision + recall + f1) / 4.0
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "comp": comp}
