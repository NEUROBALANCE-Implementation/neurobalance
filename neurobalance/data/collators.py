# neurobalance/data/collators.py
from __future__ import annotations

from typing import Any, Dict, List
import torch


def toy_vqa_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collator for ToyVQADataset.
    Returns:
      pixel_values: [B, 3, 336, 336]
      questions: list[str]
      answers: list[str]
      domains: list[str]
      ids: list[str]
    """
    pixel_values = torch.stack([x["image"] for x in batch], dim=0)
    questions = [x["question"] for x in batch]
    answers = [x["answer"] for x in batch]
    domains = [x["domain"] for x in batch]
    ids = [x["id"] for x in batch]

    return {
        "pixel_values": pixel_values,
        "questions": questions,
        "answers": answers,
        "domains": domains,
        "ids": ids,
    }
