# neurobalance/data/vqa_datasets.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any
import random
import torch
from torch.utils.data import Dataset


@dataclass
class ToySample:
    image: torch.Tensor
    question: str
    answer: str
    domain: str


class ToyVQADataset(Dataset):
    """
    Tiny dataset for pipeline validation:
    - Returns random image tensor [3, 336, 336]
    - Simple short question/answer strings
    - Domain label (med/path/drive) to mimic multi-domain setting
    """
    DOMAINS = ["med", "path", "drive"]

    def __init__(self, n: int = 64, image_size: int = 336, seed: int = 42):
        super().__init__()
        self.n = n
        self.image_size = image_size
        rng = random.Random(seed)

        # Small fixed question/answer pool (deterministic)
        self.qa_pool = [
            ("What is shown?", "ct scan"),
            ("Is there an abnormality?", "yes"),
            ("How many objects?", "3"),
            ("What color is the marker?", "green"),
            ("Is it safe to turn left?", "no"),
        ]
        self.domains = [rng.choice(self.DOMAINS) for _ in range(n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        C = 3
        H = W = self.image_size
        image = torch.rand(C, H, W)  # random image

        q, a = self.qa_pool[idx % len(self.qa_pool)]
        domain = self.domains[idx]

        return {
            "image": image,
            "question": q,
            "answer": a,
            "domain": domain,
            "id": f"toy_{idx}",
        }
