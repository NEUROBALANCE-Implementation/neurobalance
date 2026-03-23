# scripts/statistical_validation.py
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Optional SciPy (for exact p-values). Script works without it.
try:
    from scipy import stats as scipy_stats  # type: ignore
except Exception:
    scipy_stats = None


# -----------------------------
# Helpers: JSON / JSONL loading
# -----------------------------
def _get_nested(d: Dict[str, Any], key: str) -> Any:
    """
    Supports dot-path keys like: "metrics.vqa_accuracy" or "vqa_accuracy".
    """
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Key '{key}' not found (failed at '{part}').")
        cur = cur[part]
    return cur


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_scores(path: str, score_key: str = "score", id_key: str = "id") -> Tuple[List[str], np.ndarray]:
    """
    Reads a JSONL file with lines like:
      {"id":"...", "score": 0/1 or float}
    Returns ids + scores array.
    """
    ids: List[str] = []
    scores: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if id_key not in obj:
                raise KeyError(f"{path}:{ln} missing '{id_key}'")
            if score_key not in obj:
                raise KeyError(f"{path}:{ln} missing '{score_key}'")
            ids.append(str(obj[id_key]))
            scores.append(float(obj[score_key]))
    return ids, np.asarray(scores, dtype=np.float64)


# -----------------------------
# Stats: summary + bootstrap
# -----------------------------
@dataclass
class Summary:
    n: int
    mean: float
    std: float
    sem: float


def summarize(x: np.ndarray) -> Summary:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        raise ValueError("Empty array.")
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    return Summary(n=n, mean=mean, std=std, sem=sem)


def bootstrap_ci(
    x: np.ndarray,
    *,
    stat_fn=np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap CI for a statistic over samples in x.
    """
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = x.size
    if n == 0:
        raise ValueError("Empty array.")

    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = stat_fn(x[idx])

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stats, alpha))
    hi = float(np.quantile(stats, 1.0 - alpha))
    return lo, hi


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cohen's d for paired samples (aka dz):
      d = mean(diff) / std(diff)
    """
    diff = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return 0.0
    return float(diff.mean() / sd)


def paired_t_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, int, Optional[float]]:
    """
    Paired t-test between a and b.
    Returns (t_stat, df, p_value or None).
    If SciPy is available, returns exact two-sided p-value. Otherwise p_value=None.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired arrays must have same shape. Got {a.shape} vs {b.shape}")

    diff = b - a
    n = diff.size
    if n < 2:
        return 0.0, max(0, n - 1), None

    mean = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        return 0.0, n - 1, 1.0 if scipy_stats else None

    t = mean / (sd / math.sqrt(n))
    df = n - 1

    if scipy_stats is None:
        return float(t), df, None

    p = float(scipy_stats.ttest_rel(b, a).pvalue)  # two-sided
    return float(t), df, p


# -----------------------------
# Modes
# 1) Across-seeds: scalar metric per run
# 2) Paired: per-example scores for baseline vs model
# -----------------------------
def load_scalar_metric_files(paths: Sequence[str], metric_key: str) -> Tuple[List[str], np.ndarray]:
    """
    Each file must be JSON and contain the metric at metric_key (supports dot-path).
    Returns file_names + values.
    """
    names: List[str] = []
    vals: List[float] = []
    for p in paths:
        obj = load_json(p)
        v = _get_nested(obj, metric_key)
        names.append(os.path.basename(p))
        vals.append(float(v))
    return names, np.asarray(vals, dtype=np.float64)


def align_by_id(
    ids_a: List[str], scores_a: np.ndarray, ids_b: List[str], scores_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two (ids, scores) lists by id intersection and same ordering.
    """
    map_a = {i: float(s) for i, s in zip(ids_a, scores_a)}
    map_b = {i: float(s) for i, s in zip(ids_b, scores_b)}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common:
        raise ValueError("No overlapping ids between the two JSONL files.")
    a = np.asarray([map_a[i] for i in common], dtype=np.float64)
    b = np.asarray([map_b[i] for i in common], dtype=np.float64)
    return a, b


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical validation: mean±std, CI, paired tests, effect size.")
    sub = p.add_subparsers(dest="mode", required=True)

    # Mode A: across seeds (scalar metrics)
    p_seed = sub.add_parser("across_seeds", help="Compute mean±std/CI across multiple seed runs (scalar metric).")
    p_seed.add_argument("--metrics_glob", type=str, required=True,
                        help='Glob pattern for JSON metric files, e.g. "results/metrics/*_seed*.json"')
    p_seed.add_argument("--metric_key", type=str, required=True,
                        help='Metric key in JSON (supports dot paths), e.g. "metrics.vqa_accuracy"')
    p_seed.add_argument("--n_boot", type=int, default=2000)
    p_seed.add_argument("--ci", type=float, default=0.95)
    p_seed.add_argument("--seed", type=int, default=42)

    # Mode B: paired per-example (baseline vs model)
    p_pair = sub.add_parser("paired", help="Paired significance using per-example JSONL scores.")
    p_pair.add_argument("--baseline_jsonl", type=str, required=True,
                        help='Baseline per-example scores JSONL (id + score).')
    p_pair.add_argument("--model_jsonl", type=str, required=True,
                        help='Model per-example scores JSONL (id + score).')
    p_pair.add_argument("--score_key", type=str, default="score")
    p_pair.add_argument("--id_key", type=str, default="id")
    p_pair.add_argument("--n_boot", type=int, default=2000)
    p_pair.add_argument("--ci", type=float, default=0.95)
    p_pair.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def print_table(rows: List[Tuple[str, str]]) -> None:
    col1 = max(len(r[0]) for r in rows)
    print("")
    for k, v in rows:
        print(f"{k:<{col1}} : {v}")
    print("")


def main() -> int:
    args = parse_args()

    if args.mode == "across_seeds":
        paths = sorted(glob.glob(args.metrics_glob))
        if not paths:
            raise SystemExit(f"No files matched: {args.metrics_glob}")

        names, vals = load_scalar_metric_files(paths, args.metric_key)
        s = summarize(vals)
        lo, hi = bootstrap_ci(vals, n_boot=args.n_boot, ci=args.ci, seed=args.seed)

        rows = [
            ("Mode", "across_seeds (scalar metric per run)"),
            ("Files matched", str(len(paths))),
            ("Metric key", args.metric_key),
            ("Values", ", ".join(f"{n}={v:.6f}" for n, v in zip(names, vals.tolist()))),
            ("Mean", f"{s.mean:.6f}"),
            ("Std (ddof=1)", f"{s.std:.6f}"),
            ("Mean ± Std", f"{s.mean:.6f} ± {s.std:.6f}"),
            (f"Bootstrap {int(args.ci*100)}% CI", f"[{lo:.6f}, {hi:.6f}]"),
            ("Note", "CI is over seed-level values (few seeds => wide CI)."),
        ]
        print_table(rows)

        if scipy_stats is None:
            print("SciPy not found → exact p-values not available (this mode typically doesn't need them).")
        return 0

    if args.mode == "paired":
        ids_a, a = load_jsonl_scores(args.baseline_jsonl, score_key=args.score_key, id_key=args.id_key)
        ids_b, b = load_jsonl_scores(args.model_jsonl, score_key=args.score_key, id_key=args.id_key)

        a_aligned, b_aligned = align_by_id(ids_a, a, ids_b, b)
        diff = b_aligned - a_aligned

        s_a = summarize(a_aligned)
        s_b = summarize(b_aligned)
        s_d = summarize(diff)

        # Bootstrap CI over examples for mean difference
        lo, hi = bootstrap_ci(diff, n_boot=args.n_boot, ci=args.ci, seed=args.seed)

        t, df, p = paired_t_test(a_aligned, b_aligned)
        d = cohen_d_paired(a_aligned, b_aligned)

        rows = [
            ("Mode", "paired (per-example scores)"),
            ("Common examples (aligned by id)", str(diff.size)),
            ("Baseline mean ± std", f"{s_a.mean:.6f} ± {s_a.std:.6f}"),
            ("Model mean ± std", f"{s_b.mean:.6f} ± {s_b.std:.6f}"),
            ("Mean improvement (model - base)", f"{s_d.mean:.6f}"),
            (f"Bootstrap {int(args.ci*100)}% CI (improvement)", f"[{lo:.6f}, {hi:.6f}]"),
            ("Paired t-stat (df)", f"{t:.6f} (df={df})"),
            ("Paired p-value (two-sided)", f"{p:.6g}" if p is not None else "SciPy not installed → p-value unavailable"),
            ("Cohen's d (paired dz)", f"{d:.6f}"),
        ]
        print_table(rows)

        if scipy_stats is None:
            print("Tip: install SciPy for exact p-values:")
            print("  pip install scipy")
        return 0

    raise SystemExit("Unknown mode.")

if __name__ == "__main__":
    raise SystemExit(main())
