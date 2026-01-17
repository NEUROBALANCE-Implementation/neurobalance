from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class KnowledgeInjectionConfig:
    gamma: float = 0.0
    learnable_gamma: bool = False
    clamp_min: float = 0.0
    clamp_max: float = 5.0


class KnowledgeInjection(nn.Module):
    """
    h_out = h + gamma * (mask * knowledge)

    - h: [B,T,D]
    - knowledge: [B,D] or [B,1,D] or [B,T,D]
    - mask: [D] or [1,1,D] or [B,1,D] ... broadcastable to h
    """
    def __init__(self, cfg: KnowledgeInjectionConfig | None = None):
        super().__init__()
        self.cfg = cfg or KnowledgeInjectionConfig()
        if self.cfg.learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(float(self.cfg.gamma)))
        else:
            self.register_buffer("gamma", torch.tensor(float(self.cfg.gamma)))

    def set_gamma(self, gamma: float) -> None:
        with torch.no_grad():
            self.gamma.copy_(torch.tensor(float(gamma), device=self.gamma.device))

    def forward(
        self,
        h: torch.Tensor,
        knowledge: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if h.dim() != 3:
            raise ValueError("h must be [B,T,D]")

        B, T, D = h.shape

        # reshape knowledge to broadcast to [B,T,D]
        if knowledge.dim() == 2:
            knowledge = knowledge[:, None, :]  # [B,1,D]
        if knowledge.dim() == 3 and knowledge.shape[1] == 1:
            knowledge = knowledge.expand(B, T, D)
        elif knowledge.dim() == 3 and knowledge.shape[1] == T:
            pass
        else:
            if knowledge.shape != (B, T, D):
                raise ValueError(f"knowledge not broadcastable to [B,T,D], got {knowledge.shape}")

        if mask is None:
            mask = 1.0
        else:
            # ensure broadcastable
            if mask.dim() == 1:
                mask = mask[None, None, :]  # [1,1,D]

        gamma = torch.clamp(self.gamma, self.cfg.clamp_min, self.cfg.clamp_max)
        return h + gamma * (knowledge * mask)
