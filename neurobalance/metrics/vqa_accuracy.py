# neurobalance/metrics/vqa_accuracy.py
from __future__ import annotations

import re
import string
from typing import Iterable, List, Sequence, Union, Dict, Any


def _normalize_text(s: str) -> str:
    """
    Simple VQA-style normalization:
    - lowercase
    - strip punctuation
    - collapse whitespace
    """
    s = s.lower().strip()

    # remove punctuation
    table = str.maketrans("", "", string.punctuation)
    s = s.translate(table)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def vqa_exact_match(pred: str, gt: str) -> float:
    """
    Returns 1.0 if normalized strings match else 0.0
    """
    return 1.0 if _normalize_text(pred) == _normalize_text(gt) else 0.0


def vqa_accuracy_one(pred: str, gts: Union[str, Sequence[str]]) -> float:
    """
    Computes accuracy for a single prediction.
    If multiple ground truths exist, we treat it as correct if it matches ANY.

    (If you later want "soft" VQA accuracy (min(1, #humans_agree/3)),
    we can add it — this is the clean baseline.)
    """
    if isinstance(gts, str):
        gts_list = [gts]
    else:
        gts_list = list(gts)

    pred_n = _normalize_text(pred)
    for gt in gts_list:
        if pred_n == _normalize_text(gt):
            return 1.0
    return 0.0


def compute_vqa_accuracy(
    preds: Sequence[str],
    gts: Sequence[Union[str, Sequence[str]]],
) -> Dict[str, float]:
    """
    Returns dict so it plugs nicely into logs/tables:
      {"vqa_accuracy": ..., "n": ...}
    """
    if len(preds) != len(gts):
        raise ValueError(f"preds and gts must have same length. got {len(preds)} vs {len(gts)}")

    correct = 0.0
    for p, gt in zip(preds, gts):
        correct += vqa_accuracy_one(p, gt)

    n = float(len(preds))
    return {"vqa_accuracy": (correct / n if n > 0 else 0.0), "n": n}
