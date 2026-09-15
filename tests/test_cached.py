import torch

from litedsps.evidence import behavioral_evidence_batch
from litedsps.model import LiteDSPSConfig, LiteDSPSMamba


def test_cached_and_raw_paths_match_in_eval():
    cfg = LiteDSPSConfig(visual_dim=4, audio_dim=3, k=8, embed_dim=12, dynamic_dim=8, d_state=4, temporal_backend="torch", dropout=0.0)
    model = LiteDSPSMamba(cfg).eval()
    v = torch.randn(2, 10, 4)
    a = torch.randn(2, 9, 3)
    vl = torch.tensor([10, 7])
    al = torch.tensor([9, 6])
    ve, vm = behavioral_evidence_batch(v, vl, k=8)
    ae, am = behavioral_evidence_batch(a, al, k=8)
    with torch.no_grad():
        raw = model(visual=v, visual_lengths=vl, audio=a, audio_lengths=al)["logits"]
        cached = model(visual_evidence=ve, visual_mask=vm, audio_evidence=ae, audio_mask=am)["logits"]
    assert torch.allclose(raw, cached, atol=1e-6)
