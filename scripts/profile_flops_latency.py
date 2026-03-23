# scripts/profile_flops_latency.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import torch

from neurobalance.utils.config import load_yaml, apply_overrides
from neurobalance.models.neurobalance_model import NeuroBalanceModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile FLOPs (approx), latency, and memory for NeuroBalanceModel.")
    p.add_argument("--config", type=str, required=True, help="YAML config path.")
    p.add_argument("--override", type=str, action="append", default=[], help="Override config: key=value")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument("--seq_len", type=int, default=32, help="Approx text length (only used to synthesize dummy questions).")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", type=str, default="", help="cuda/cpu. If empty, auto.")
    p.add_argument("--dtype", type=str, default="fp16", help="fp32|fp16|bf16")
    p.add_argument("--deterministic", type=str, default="false", help="true/false")
    p.add_argument("--out_json", type=str, default="", help="Optional: write results JSON to this path.")
    return p.parse_args()


def str2bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "yes", "y")


def choose_device(arg_device: str) -> torch.device:
    if arg_device:
        return torch.device(arg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError("dtype must be one of: fp32, fp16, bf16")


def make_dummy_batch(
    batch_size: int,
    image_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    # Dummy images
    pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=device, dtype=dtype)
    # Dummy text
    questions = [("What is shown in the image? " + "x" * max(0, seq_len - 24)) for _ in range(batch_size)]
    answers = [("dummy " + "y" * max(0, seq_len - 6)) for _ in range(batch_size)]
    return {"pixel_values": pixel_values, "questions": questions, "answers": answers}


def approx_flops_llm_forward(model: NeuroBalanceModel, batch_size: int, seq_total: int) -> Optional[float]:
    """
    Very rough FLOPs estimate for GPT2-like transformer:
      FLOPs ~ 2 * L * ( (4*H^2 + 2*H*seq_total) * seq_total )  [order-of-mag]
    This is NOT exact. It's a heuristic to compare configs (baseline vs full).
    If model isn't GPT2-like, returns None.
    """
    try:
        lm = model.backbone.lm
        cfg = lm.config
        L = int(getattr(cfg, "n_layer", 0))
        H = int(getattr(cfg, "n_embd", 0) or getattr(cfg, "hidden_size", 0))
        if L <= 0 or H <= 0:
            return None
        # crude transformer block estimate
        # per layer: attention (QKV+proj ~ 4*H^2*seq) + attn matmul (~2*seq^2*H) + MLP (~8*H^2*seq)
        # combine ~ (12*H^2*seq + 2*seq^2*H)
        flops_per_layer = (12 * (H**2) * seq_total) + (2 * (seq_total**2) * H)
        # forward only
        return float(batch_size * L * flops_per_layer)
    except Exception:
        return None


@torch.no_grad()
def measure_latency(model: torch.nn.Module, batch: Dict[str, Any], warmup: int, iters: int) -> Dict[str, float]:
    model.eval()

    # Warmup
    for _ in range(max(0, warmup)):
        _ = model(batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total = t1 - t0
    avg = total / max(1, iters)
    return {"total_s": total, "avg_s": avg, "iters": float(iters)}


def get_memory_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
        "cuda_max_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
    }


def main() -> int:
    args = parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)
    cfg["_config_path"] = args.config

    # Allow CLI deterministic override (useful for Colab determinism error)
    if args.deterministic:
        cfg["deterministic"] = str2bool(args.deterministic)

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype)

    if str2bool(str(cfg.get("deterministic", False))):
        # Determinism can crash on CUDA unless env var set.
        # We keep this safe: if deterministic requested on CUDA, set env var suggestion.
        if device.type == "cuda":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    # Build model
    model = NeuroBalanceModel(cfg).to(device)
    model.eval()

    # If using fp16/bf16 on CPU, keep fp32 (CPU half precision can be slow/unsupported)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    # Dummy batch
    batch = make_dummy_batch(
        batch_size=args.batch_size,
        image_size=args.image_size,
        seq_len=args.seq_len,
        device=device,
        dtype=dtype,
    )

    # Reset CUDA memory counters
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Latency
    lat = measure_latency(model, batch, warmup=args.warmup, iters=args.iters)

    # Approx FLOPs (very rough; intended for relative comparison)
    # total sequence length = visual tokens + text tokens
    num_vis = int(getattr(model.backbone.cfg, "num_image_tokens", 16))
    seq_total = num_vis + int(args.seq_len)
    flops = approx_flops_llm_forward(model, batch_size=args.batch_size, seq_total=seq_total)

    mem = get_memory_stats(device)

    report: Dict[str, Any] = {
        "config_path": args.config,
        "overrides": list(args.override),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seq_len_text_est": args.seq_len,
        "seq_total_est": seq_total,
        "latency": lat,
        "approx_flops_forward": flops,
        "memory": mem,
        "neurobalance_toggles": {
            "gating_enabled": bool(getattr(model, "gating", None) is not None),
            "attention_enabled": bool(getattr(model, "adaptive_attention", None) is not None),
            "injection_enabled": bool(getattr(model, "injection", None) is not None),
        },
    }

    print("\n=== PROFILE REPORT ===")
    print(json.dumps(report, indent=2))
    print("======================\n")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"✅ Wrote report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
