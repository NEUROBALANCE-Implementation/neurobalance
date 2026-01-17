import torch
import torch.nn.functional as F
from neurobalance.modules.adaptive_attention import AdaptiveAttention, AdaptiveAttentionConfig


def attention_entropy(probs: torch.Tensor) -> torch.Tensor:
    # probs: [..., T]
    eps = 1e-9
    return -(probs * (probs + eps).log()).sum(dim=-1)


def test_attention_logits_scale_with_beta():
    torch.manual_seed(0)
    B, H, T = 2, 3, 5
    logits = torch.randn(B, H, T, T)

    mod = AdaptiveAttention(AdaptiveAttentionConfig(init_beta=1.0, learnable=False))

    mod.set_beta(1.0)
    out1 = mod(logits)

    mod.set_beta(2.0)
    out2 = mod(logits)

    assert torch.allclose(out2, out1 * 2.0, atol=1e-6)


def test_attention_sharpens_with_higher_beta():
    torch.manual_seed(0)
    B, H, T = 1, 1, 6
    logits = torch.randn(B, H, T, T)

    mod = AdaptiveAttention(AdaptiveAttentionConfig(init_beta=1.0, learnable=False))

    mod.set_beta(0.5)
    p_lo = F.softmax(mod(logits), dim=-1)
    ent_lo = attention_entropy(p_lo).mean()

    mod.set_beta(5.0)
    p_hi = F.softmax(mod(logits), dim=-1)
    ent_hi = attention_entropy(p_hi).mean()

    # higher beta => sharper distribution => lower entropy
    assert ent_hi < ent_lo
