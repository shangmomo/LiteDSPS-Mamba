import torch

from litedsps.model import LiteDSPSConfig, LiteDSPSMamba


def test_model_shape_with_ci_backend():
    cfg = LiteDSPSConfig(visual_dim=136, audio_dim=25, temporal_backend="torch")
    model = LiteDSPSMamba(cfg)
    v = torch.randn(2, 30, 136)
    a = torch.randn(2, 24, 25)
    out = model(
        visual=v,
        visual_lengths=torch.tensor([30, 17]),
        audio=a,
        audio_lengths=torch.tensor([24, 19]),
    )
    assert out["logits"].shape == (2, 2)
    assert out["visual_embedding"].shape == (2, 256)
    assert out["audio_embedding"].shape == (2, 256)
