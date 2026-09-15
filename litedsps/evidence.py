from __future__ import annotations

from typing import Tuple

import torch


def _region_bounds(length: int, regions: int, device: torch.device) -> torch.Tensor:
    """Return integer boundaries for contiguous, non-empty ordered regions."""
    if length < 1:
        raise ValueError("length must be >= 1")
    regions = min(length, regions)
    return torch.floor(torch.linspace(0, length, regions + 1, device=device)).long()


def behavioral_evidence_single(
    x: torch.Tensor,
    k: int = 96,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build ordered LiteDSPS behavioral evidence for one sequence.

    Args:
        x: Tensor [T, D].
        k: Fixed evidence-token budget.
        eps: Numerical stability constant for dispersion.

    Returns:
        evidence: [K, 4D] padded with zeros when T < K.
        mask: [K] bool, True for valid evidence tokens.

    The four statistics per ordered region are:
        mean, log1p(std), mean(abs first difference), max(abs first difference).

    Differences are computed before regional reduction. A zero difference is
    prepended at t=0 so change statistics share the same ordered boundaries as
    the input sequence.
    """
    if x.ndim != 2:
        raise ValueError(f"expected [T, D], got {tuple(x.shape)}")
    t, d = x.shape
    if t < 1:
        raise ValueError("empty sequences are not supported")

    km = min(t, k)
    bounds = _region_bounds(t, km, x.device)
    diff = torch.zeros_like(x)
    if t > 1:
        diff[1:] = (x[1:] - x[:-1]).abs()

    tokens = []
    for r in range(km):
        start, end = int(bounds[r].item()), int(bounds[r + 1].item())
        region = x[start:end]
        dregion = diff[start:end]
        mu = region.mean(dim=0)
        q = (region.square().mean(dim=0) - mu.square()).clamp_min(0.0) + eps
        s = torch.log1p(torch.sqrt(q))
        d_avg = dregion.mean(dim=0)
        d_max = dregion.amax(dim=0)
        tokens.append(torch.cat([mu, s, d_avg, d_max], dim=-1))

    evidence = x.new_zeros((k, 4 * d))
    evidence[:km] = torch.stack(tokens, dim=0)
    mask = torch.zeros(k, dtype=torch.bool, device=x.device)
    mask[:km] = True
    return evidence, mask


def behavioral_evidence_batch(
    x: torch.Tensor,
    lengths: torch.Tensor,
    k: int = 96,
    eps: float = 1e-6,
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
