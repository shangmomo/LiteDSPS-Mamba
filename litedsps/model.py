from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .evidence import behavioral_evidence_batch


class TorchTemporalStub(nn.Module):
    """Shape-compatible CI/debug backend. Not the paper model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.pw = nn.Linear(d_model, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        return self.pw(self.act(y))


def _make_temporal(backend: str, d_model: int, d_state: int, d_conv: int, expand: int) -> nn.Module:
    if backend == "mamba2":
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:
            raise ImportError(
                "Mamba2 backend requested but mamba-ssm is not installed. "
                "Install requirements-mamba.txt on a CUDA-capable environment."
            ) from exc
        return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    if backend == "torch":
        return TorchTemporalStub(d_model)
    raise ValueError(f"unknown temporal backend: {backend}")


def reverse_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse only valid prefix positions; keep padded suffix at zero."""
    out = torch.zeros_like(x)
    lengths = mask.sum(dim=1).tolist()
    for i, length in enumerate(lengths):
        length = int(length)
        if length:
            out[i, :length] = torch.flip(x[i, :length], dims=[0])
    return out


def masked_mean_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1)
    mean = (x * m).sum(dim=1) / denom
    neg_inf = torch.finfo(x.dtype).min
    maxv = x.masked_fill(~m, neg_inf).amax(dim=1)
    return torch.cat([mean, maxv], dim=-1)


class ModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class LiteDSPSConfig:
    visual_dim: int = 136
    audio_dim: int = 25
    k: int = 96
    embed_dim: int = 128
    dynamic_dim: int = 96
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.30
    num_classes: int = 2
    share_modalities: bool = True
    share_directions: bool = True
    bidirectional: bool = True
    aggregation: str = "mean_max"
    use_dynamic_diff: bool = True
    evidence_mode: str = "full"  # full | avgpool96
    temporal_backend: str = "mamba2"

    @property
    def bypass_dim(self) -> int:
        return self.embed_dim - self.dynamic_dim


class LiteDSPSMamba(nn.Module):
    def __init__(self, cfg: LiteDSPSConfig):
        super().__init__()
        if cfg.dynamic_dim <= 0 or cfg.dynamic_dim > cfg.embed_dim:
            raise ValueError("dynamic_dim must be in (0, embed_dim]")
        self.cfg = cfg
        self.visual_projector = ModalityProjector(4 * cfg.visual_dim, cfg.embed_dim, cfg.dropout)
        self.audio_projector = ModalityProjector(4 * cfg.audio_dim, cfg.embed_dim, cfg.dropout)

        self.temporal = nn.ModuleDict()
        modality_keys = ["shared"] if cfg.share_modalities else ["visual", "audio"]
        direction_keys = ["shared"] if (cfg.share_directions or not cfg.bidirectional) else ["fwd", "bwd"]
        for mk in modality_keys:
            for dk in direction_keys:
                self.temporal[f"{mk}_{dk}"] = _make_temporal(
                    cfg.temporal_backend, cfg.dynamic_dim, cfg.d_state, cfg.d_conv, cfg.expand
                )

        self.post_temporal_norm = nn.LayerNorm(cfg.dynamic_dim)
        self.reunite_norm = nn.LayerNorm(cfg.embed_dim)
        pooled_dim = cfg.embed_dim if cfg.aggregation == "mean" else 2 * cfg.embed_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * pooled_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * pooled_dim, cfg.num_classes),
        )

    def _module(self, modality: str, direction: str) -> nn.Module:
        mk = "shared" if self.cfg.share_modalities else modality
        dk = "shared" if (self.cfg.share_directions or not self.cfg.bidirectional) else direction
        return self.temporal[f"{mk}_{dk}"]

    def _encode_dynamic(self, x: torch.Tensor, mask: torch.Tensor, modality: str) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * m
        fwd = self._module(modality, "fwd")(x)
        if not self.cfg.bidirectional:
            delta = fwd
        else:
            rev_x = reverse_valid(x, mask)
            bwd_rev = self._module(modality, "bwd")(rev_x)
            bwd = reverse_valid(bwd_rev, mask)
            delta = 0.5 * (fwd + bwd)
        return self.post_temporal_norm(x + delta) * m

    def _adjust_evidence(self, ev: torch.Tensor, feature_dim: int) -> torch.Tensor:
        # Cached evidence stores all four statistics. Ablations are applied here
        # so one deterministic cache can be reused across controlled variants.
        if not self.cfg.use_dynamic_diff:
            ev = ev.clone()
            ev[..., 2 * feature_dim :] = 0
        if self.cfg.evidence_mode == "avgpool96":
            ev = ev.clone()
            ev[..., feature_dim:] = 0
        elif self.cfg.evidence_mode != "full":
            raise ValueError(f"unknown evidence_mode: {self.cfg.evidence_mode}")
        return ev

    def _evidence_from_raw(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        feature_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ev, mask = behavioral_evidence_batch(x, lengths, k=self.cfg.k)
        return self._adjust_evidence(ev, feature_dim), mask

    def forward(
        self,
        visual: Optional[torch.Tensor] = None,
        visual_lengths: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        audio_lengths: Optional[torch.Tensor] = None,
        visual_evidence: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        audio_evidence: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if visual_evidence is None:
            if visual is None or visual_lengths is None:
                raise ValueError("provide visual+visual_lengths or cached visual_evidence+visual_mask")
            visual_evidence, visual_mask = self._evidence_from_raw(visual, visual_lengths, self.cfg.visual_dim)
        if audio_evidence is None:
            if audio is None or audio_lengths is None:
                raise ValueError("provide audio+audio_lengths or cached audio_evidence+audio_mask")
            audio_evidence, audio_mask = self._evidence_from_raw(audio, audio_lengths, self.cfg.audio_dim)
        assert visual_mask is not None and audio_mask is not None
        visual_evidence = self._adjust_evidence(visual_evidence, self.cfg.visual_dim)
        audio_evidence = self._adjust_evidence(audio_evidence, self.cfg.audio_dim)

        hv = self.visual_projector(visual_evidence)
        ha = self.audio_projector(audio_evidence)
        vd, vb = hv[..., : self.cfg.dynamic_dim], hv[..., self.cfg.dynamic_dim :]
        ad, ab = ha[..., : self.cfg.dynamic_dim], ha[..., self.cfg.dynamic_dim :]

        if self.cfg.share_modalities:
            stacked = torch.cat([vd, ad], dim=0)
            smask = torch.cat([visual_mask, audio_mask], dim=0)
            encoded = self._encode_dynamic(stacked, smask, "visual")
            vd2, ad2 = encoded.chunk(2, dim=0)
        else:
            vd2 = self._encode_dynamic(vd, visual_mask, "visual")
            ad2 = self._encode_dynamic(ad, audio_mask, "audio")

        vh = self.reunite_norm(torch.cat([vd2, vb], dim=-1))
        ah = self.reunite_norm(torch.cat([ad2, ab], dim=-1))

        if self.cfg.aggregation == "mean_max":
            vp = masked_mean_max(vh, visual_mask)
            ap = masked_mean_max(ah, audio_mask)
        elif self.cfg.aggregation == "mean":
            vp = masked_mean_max(vh, visual_mask)[..., : self.cfg.embed_dim]
            ap = masked_mean_max(ah, audio_mask)[..., : self.cfg.embed_dim]
        else:
            raise ValueError(f"unknown aggregation: {self.cfg.aggregation}")

        logits = self.classifier(torch.cat([vp, ap], dim=-1))
        return {"logits": logits, "visual_embedding": vp, "audio_embedding": ap}
