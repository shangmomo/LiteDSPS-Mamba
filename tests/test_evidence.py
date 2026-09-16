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


def test_difference_sequence_is_pooled_independently():
    # T=4, K=2. The original experiment code pools delta=[1,2,3]
    # independently to [1.5, 2.5] (mean) and [2, 3] (max).
    x = torch.tensor([[0.0], [1.0], [3.0], [6.0]])
    e, m = behavioral_evidence_single(x, k=2)
    valid = e[m]
    assert torch.allclose(valid[:, 2], torch.tensor([1.5, 2.5]))
    assert torch.allclose(valid[:, 3], torch.tensor([2.0, 3.0]))
