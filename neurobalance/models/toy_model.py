# neurobalance/models/toy_model.py
from __future__ import annotations

from typing import Any, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyVQAModel(nn.Module):
    """
    A tiny model that:
    - Encodes image with a few conv layers
    - Encodes text question with a simple embedding of character codes (very crude)
    - Produces logits over a small toy answer vocabulary
    This is ONLY for pipeline validation.
    """
    def __init__(self, answer_vocab: List[str] | None = None):
        super().__init__()
        self.answer_vocab = answer_vocab or ["ct scan", "yes", "no", "3", "green"]
        self.answer_to_id = {a: i for i, a in enumerate(self.answer_vocab)}

        # Tiny vision encoder
        self.vision = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.vision_proj = nn.Linear(32, 64)

        # Tiny text encoder (very simple)
        self.text_proj = nn.Linear(1, 64)

        # Classifier
        self.classifier = nn.Linear(64, len(self.answer_vocab))

    def encode_question(self, questions: List[str], device: torch.device) -> torch.Tensor:
        # crude scalar feature: average character code / 1000
        vals = []
        for q in questions:
            if len(q) == 0:
                vals.append(0.0)
            else:
                vals.append(sum(ord(c) for c in q) / max(1, len(q)) / 1000.0)
        x = torch.tensor(vals, device=device, dtype=torch.float32).unsqueeze(-1)  # [B,1]
        return self.text_proj(x)  # [B,64]

    def forward(self, pixel_values: torch.Tensor, questions: List[str], answers: List[str], **kwargs: Any) -> Dict[str, Any]:
        device = pixel_values.device

        v = self.vision(pixel_values).flatten(1)      # [B,32]
        v = self.vision_proj(v)                       # [B,64]
        t = self.encode_question(questions, device)   # [B,64]

        h = torch.tanh(v + t)                         # [B,64]
        logits = self.classifier(h)                   # [B,V]

        # Targets
        y = torch.tensor([self.answer_to_id.get(a, 0) for a in answers], device=device, dtype=torch.long)
        loss = F.cross_entropy(logits, y)

        pred_ids = logits.argmax(dim=-1).tolist()
        preds = [self.answer_vocab[i] for i in pred_ids]

        return {
            "loss": loss,
            "logits": logits,
            "preds": preds,
        }
