# neurobalance/metrics/bleu_rouge_optional.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union, Tuple
import math
import re
import string


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    table = str.maketrans("", "", string.punctuation)
    s = s.translate(table)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    s = _normalize_text(s)
    return s.split() if s else []


def _ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if n <= 0:
        return counts
    for i in range(0, max(0, len(tokens) - n + 1)):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _clipped_precision(candidate: List[str], references: List[List[str]], n: int) -> float:
    cand_counts = _ngram_counts(candidate, n)
    if not cand_counts:
        return 0.0

    max_ref_counts: Dict[Tuple[str, ...], int] = {}
    for ref in references:
        ref_counts = _ngram_counts(ref, n)
        for ng, c in ref_counts.items():
            max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), c)

    clipped = 0
    total = 0
    for ng, c in cand_counts.items():
        clipped += min(c, max_ref_counts.get(ng, 0))
        total += c

    return clipped / total if total > 0 else 0.0


def _brevity_penalty(cand_len: int, ref_lens: List[int]) -> float:
    if cand_len == 0:
        return 0.0
    # choose reference length closest to candidate
    closest = min(ref_lens, key=lambda rl: (abs(rl - cand_len), rl))
    if cand_len > closest:
        return 1.0
    return math.exp(1.0 - (closest / cand_len))


@dataclass
class BLEUConfig:
    max_n: int = 4
    smooth: float = 1e-9  # smoothing to avoid log(0)


def bleu_score_one(pred: str, gts: Union[str, Sequence[str]], cfg: BLEUConfig | None = None) -> float:
    cfg = cfg or BLEUConfig()
    cand = _tokenize(pred)

    if isinstance(gts, str):
        refs = [_tokenize(gts)]
    else:
        refs = [_tokenize(x) for x in gts]

    precisions = []
    for n in range(1, cfg.max_n + 1):
        p_n = _clipped_precision(cand, refs, n)
        precisions.append(max(p_n, cfg.smooth))

    # geometric mean of precisions
    log_p = sum(math.log(p) for p in precisions) / cfg.max_n
    bp = _brevity_penalty(len(cand), [len(r) for r in refs])
    return float(bp * math.exp(log_p))


def _lcs_len(a: List[str], b: List[str]) -> int:
    """
    LCS length DP for ROUGE-L.
    """
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev_diag = 0
        for j in range(1, len(b) + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev_diag + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev_diag = temp
    return dp[-1]


@dataclass
class ROUGEConfig:
    beta: float = 1.2  # typical ROUGE-L beta


def rouge_l_one(pred: str, gts: Union[str, Sequence[str]], cfg: ROUGEConfig | None = None) -> float:
    """
    ROUGE-L F-score against the best matching reference.
    """
    cfg = cfg or ROUGEConfig()
    cand = _tokenize(pred)

    if isinstance(gts, str):
        refs = [_tokenize(gts)]
    else:
        refs = [_tokenize(x) for x in gts]

    best = 0.0
    for ref in refs:
        lcs = _lcs_len(cand, ref)
        if lcs == 0:
            continue

        prec = lcs / len(cand) if len(cand) > 0 else 0.0
        rec = lcs / len(ref) if len(ref) > 0 else 0.0
        if prec == 0.0 or rec == 0.0:
            f = 0.0
        else:
            b2 = cfg.beta * cfg.beta
            f = (1 + b2) * prec * rec / (rec + b2 * prec)
        best = max(best, f)

    return float(best)


def compute_bleu_rouge(
    preds: Sequence[str],
    gts: Sequence[Union[str, Sequence[str]]],
) -> Dict[str, float]:
    if len(preds) != len(gts):
        raise ValueError(f"preds and gts must have same length. got {len(preds)} vs {len(gts)}")

    bleu_total = 0.0
    rouge_total = 0.0
    n = float(len(preds))

    for p, gt in zip(preds, gts):
        bleu_total += bleu_score_one(p, gt)
        rouge_total += rouge_l_one(p, gt)

    return {
        "bleu4": (bleu_total / n if n > 0 else 0.0),
        "rougeL": (rouge_total / n if n > 0 else 0.0),
        "n": n,
    }
