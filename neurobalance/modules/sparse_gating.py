from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class SparseGatingConfig:
    target_density: float = 0.15  # keep 15% activations
    mode: str = "per_token"       # "per_token" or "per_sample"
    straight_through: bool = True # allow gradients through masked values


class SparseGating(nn.Module):
    """
    Sparse neuron gating:
      input x: [B, T, D] or [B, D]
      output: same shape, but only top-k dims kept (others zeroed)

    - per_token: top-k for each (B,T) independently
    - per_sample: top-k for each B over all tokens combined (rarely used early)
    """
    def __init__(self, cfg: SparseGatingConfig | None = None):
        super().__init__()
        self.cfg = cfg or SparseGatingConfig()

        if not (0.0 < self.cfg.target_density <= 1.0):
            raise ValueError("target_density must be in (0,1].")

        if self.cfg.mode not in ("per_token", "per_sample"):
            raise ValueError("mode must be 'per_token' or 'per_sample'.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          y: gated output (same shape as x)
          mask: 0/1 mask (same shape as x) indicating kept dims
        """
        if x.dim() == 2:
            # [B, D] -> treat as [B, 1, D]
            x_in = x.unsqueeze(1)
            squeeze_back = True
        elif x.dim() == 3:
            x_in = x
            squeeze_back = False
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {x.dim()}")

        B, T, D = x_in.shape
        k = max(1, int(round(self.cfg.target_density * D)))

        scores = x_in.abs()

        if self.cfg.mode == "per_token":
            # topk over last dim for each token
            topk_vals, topk_idx = torch.topk(scores, k=k, dim=-1)
            mask = torch.zeros_like(x_in)
            mask.scatter_(-1, topk_idx, 1.0)

        else:  # per_sample
            flat = scores.view(B, T * D)
            _, flat_idx = torch.topk(flat, k=max(1, int(round(self.cfg.target_density * T * D))), dim=-1)
            mask_flat = torch.zeros_like(flat)
            mask_flat.scatter_(-1, flat_idx, 1.0)
            mask = mask_flat.view(B, T, D)

        if self.cfg.straight_through:
            # Straight-through estimator: keep forward masking but pass gradient as if identity
            y = x_in * mask + x_in.detach() * (1.0 - mask)
        else:
            y = x_in * mask

        if squeeze_back:
            return y.squeeze(1), mask.squeeze(1)
        return y, mask
