import torch

from litedsps.evidence import behavioral_evidence_single


def test_evidence_shape_and_mask():
    x = torch.randn(20, 5)
    e, m = behavioral_evidence_single(x, k=96)
    assert e.shape == (96, 20)
    assert m.shape == (96,)
    assert int(m.sum()) == 20
    assert torch.all(e[20:] == 0)


def test_constant_sequence_has_zero_change():
    x = torch.ones(100, 3)
    e, m = behavioral_evidence_single(x, k=10)
    valid = e[m]
    assert torch.allclose(valid[:, 6:], torch.zeros_like(valid[:, 6:]))
