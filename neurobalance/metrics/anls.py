# neurobalance/metrics/anls.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Union, List
import re
import string


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    table = str.maketrans("", "", string.punctuation)
    s = s.translate(table)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _levenshtein(a: str, b: str) -> int:
    """
    Classic DP Levenshtein distance (edit distance), O(len(a)*len(b)).
    Fine for evaluation-size strings.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure b is the longer string for slightly better cache locality
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


@dataclass
class ANLSConfig:
    """
    ANLS is commonly defined as:
      sim = 1 - (edit_distance / max(len(gt), len(pred)))
      score = sim if sim >= tau else 0
    """
    tau: float = 0.5  # threshold


def anls_score_one(pred: str, gts: Union[str, Sequence[str]], cfg: ANLSConfig | None = None) -> float:
    cfg = cfg or ANLSConfig()

    pred_n = _normalize_text(pred)
    if isinstance(gts, str):
        gts_list = [gts]
    else:
        gts_list = list(gts)

    best = 0.0
    for gt in gts_list:
        gt_n = _normalize_text(gt)

        denom = max(len(gt_n), len(pred_n))
        if denom == 0:
            sim = 1.0
        else:
            d = _levenshtein(pred_n, gt_n)
            sim = 1.0 - (d / denom)

        score = sim if sim >= cfg.tau else 0.0
        if score > best:
            best = score

    return float(best)


def compute_anls(
    preds: Sequence[str],
    gts: Sequence[Union[str, Sequence[str]]],
    cfg: ANLSConfig | None = None,
) -> Dict[str, float]:
    if len(preds) != len(gts):
        raise ValueError(f"preds and gts must have same length. got {len(preds)} vs {len(gts)}")

    cfg = cfg or ANLSConfig()

    total = 0.0
    for p, gt in zip(preds, gts):
        total += anls_score_one(p, gt, cfg=cfg)

    n = float(len(preds))
    return {"anls": (total / n if n > 0 else 0.0), "n": n, "anls_tau": float(cfg.tau)}
