import torch
from neurobalance.modules.sparse_gating import SparseGating, SparseGatingConfig


def test_gating_shape_and_density():
    torch.manual_seed(0)
    B, T, D = 4, 7, 100
    x = torch.randn(B, T, D)

    cfg = SparseGatingConfig(target_density=0.15, mode="per_token")
    gate = SparseGating(cfg)

    y, mask = gate(x)

    assert y.shape == x.shape
    assert mask.shape == x.shape

    # density = fraction of ones in mask
    density = mask.mean().item()
    # allow small rounding tolerance because k is rounded
    assert abs(density - 0.15) < 0.02


def test_gating_keeps_topk_magnitudes():
    torch.manual_seed(0)
    x = torch.tensor([[[0.1, -3.0, 2.0, 0.2]]])  # [1,1,4]
    gate = SparseGating(SparseGatingConfig(target_density=0.5, mode="per_token", straight_through=False))
    y, mask = gate(x)

    # keep top-2 abs values: 3.0 and 2.0 -> indices 1 and 2
    expected_mask = torch.tensor([[[0.0, 1.0, 1.0, 0.0]]])
    assert torch.allclose(mask, expected_mask)
    assert torch.allclose(y, x * expected_mask)
