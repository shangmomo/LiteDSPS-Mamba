from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


NUM_BEHAVIOR_STATS = 4


def behavioral_evidence_single(
    x: torch.Tensor,
    k: int = 96,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ordered LiteDSPS behavioral evidence for one sequence.

    This mirrors the original experiment script used for the paper. For a
    sequence ``x`` of length ``T``, the raw feature sequence is adaptively
    pooled to ``K_m = min(T, K)`` positions to obtain regional mean and
    dispersion. First-order absolute differences are computed as the separate
    length-``T-1`` sequence ``abs(x[1:] - x[:-1])`` and are independently
    adaptively pooled to the same ``K_m`` positions for mean/max change.

    Args:
        x: Tensor [T, D].
        k: Fixed evidence-token budget.
        eps: Numerical stability constant inside the standard deviation.

    Returns:
        evidence: [K, 4D] padded with zeros when T < K.
        mask: [K] bool, True for valid evidence tokens.
    """
    if x.ndim != 2:
        raise ValueError(f"expected [T, D], got {tuple(x.shape)}")
    if k <= 0:
        raise ValueError("k must be positive")

    x = x.float()
    t, d = x.shape
    evidence = torch.zeros(k, NUM_BEHAVIOR_STATS * d, dtype=x.dtype, device=x.device)
    mask = torch.zeros(k, dtype=torch.bool, device=x.device)
    if t <= 0:
        return evidence, mask

    km = min(int(t), int(k))
    xt = x.transpose(0, 1).unsqueeze(0).contiguous()  # [1, D, T]

    mean = F.adaptive_avg_pool1d(xt, km)
    mean_sq = F.adaptive_avg_pool1d(xt * xt, km)
    var = (mean_sq - mean * mean).clamp_min(0.0)
    log_std = torch.log1p(torch.sqrt(var + eps))

    if t > 1:
        delta = (x[1:] - x[:-1]).abs().transpose(0, 1).unsqueeze(0).contiguous()
        delta_mean = F.adaptive_avg_pool1d(delta, km)
        delta_max = F.adaptive_max_pool1d(delta, km)
    else:
        delta_mean = torch.zeros_like(mean)
        delta_max = torch.zeros_like(mean)

    stats = torch.cat([mean, log_std, delta_mean, delta_max], dim=1)
    stats = stats.squeeze(0).transpose(0, 1).contiguous()
    if not torch.isfinite(stats).all():
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

    evidence[:km] = stats
    mask[:km] = True
    return evidence, mask


def behavioral_evidence_batch(
    x: torch.Tensor,
    lengths: torch.Tensor,
    k: int = 96,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch wrapper for padded input sequences [B, T, D]."""
    if x.ndim != 3:
        raise ValueError(f"expected [B, T, D], got {tuple(x.shape)}")
    if lengths.ndim != 1 or lengths.numel() != x.shape[0]:
        raise ValueError("lengths must have shape [B]")

    ev, masks = [], []
    for i, length in enumerate(lengths.tolist()):
        e, m = behavioral_evidence_single(x[i, : int(length)], k=k, eps=eps)
        ev.append(e)
        masks.append(m)
    return torch.stack(ev, dim=0), torch.stack(masks, dim=0)
