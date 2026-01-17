from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn


@dataclass
class AdaptiveAttentionConfig:
    init_beta: float = 1.0
    learnable: bool = False
    clamp_min: float = 0.1
    clamp_max: float = 10.0


class AdaptiveAttention(nn.Module):
    """
    Applies domain scaling to attention logits:
      logits_out = beta * logits_in

    logits expected shape: [B, H, T, T] (standard attention logits)
    beta can be scalar or per-head/per-layer later; start scalar.
    """
    def __init__(self, cfg: AdaptiveAttentionConfig | None = None):
        super().__init__()
        self.cfg = cfg or AdaptiveAttentionConfig()
        if self.cfg.learnable:
            self.beta = nn.Parameter(torch.tensor(float(self.cfg.init_beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(self.cfg.init_beta)))

    def set_beta(self, beta: float) -> None:
        with torch.no_grad():
            self.beta.copy_(torch.tensor(float(beta), device=self.beta.device))

    def forward(self, attn_logits: torch.Tensor) -> torch.Tensor:
        beta = torch.clamp(self.beta, self.cfg.clamp_min, self.cfg.clamp_max)
        return attn_logits * beta
