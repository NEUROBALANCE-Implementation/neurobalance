import torch
from neurobalance.modules.knowledge_injection import KnowledgeInjection, KnowledgeInjectionConfig


def test_injection_increases_selected_dims_when_gamma_positive():
    torch.manual_seed(0)
    B, T, D = 2, 4, 10
    h = torch.randn(B, T, D)

    # knowledge vector: strong signal in dims [2,3,4]
    knowledge = torch.zeros(B, D)
    knowledge[:, 2:5] = 5.0

    mask = torch.zeros(D)
    mask[2:5] = 1.0

    inj = KnowledgeInjection(KnowledgeInjectionConfig(gamma=0.0, learnable_gamma=False))

    inj.set_gamma(0.0)
    out0 = inj(h, knowledge, mask=mask)

    inj.set_gamma(1.0)
    out1 = inj(h, knowledge, mask=mask)

    # Compare average magnitude in selected dims
    sel0 = out0[..., 2:5].abs().mean().item()
    sel1 = out1[..., 2:5].abs().mean().item()

    assert sel1 > sel0
