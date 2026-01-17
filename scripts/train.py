# scripts/train.py
from __future__ import annotations

import argparse
import json
import os
from itertools import cycle
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from neurobalance.utils.seed import set_seed
from neurobalance.utils.logging import SimpleLogger, make_run_id
from neurobalance.utils.config import load_yaml, apply_overrides

from neurobalance.data.vqa_datasets import ToyVQADataset
from neurobalance.data.collators import toy_vqa_collate

from neurobalance.models.neurobalance_model import NeuroBalanceModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", type=str, action="append", default=[])
    p.add_argument("--debug_toy_data", type=str, default="false")  # "true"/"false"
    p.add_argument("--max_steps", type=int, default=0)
    return p.parse_args()


def str2bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "yes", "y")


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, run_id: str
) -> str:
    os.makedirs("results/checkpoints", exist_ok=True)
    path = os.path.join("results/checkpoints", f"{run_id}_step{step}.pt")
    torch.save(
        {"step": step, "model_state": model.state_dict(), "optim_state": optimizer.state_dict()},
        path,
    )
    return path


def _to_loggable(v: Any) -> Any:
    """Make values JSONL-safe and consistent (no tensors)."""
    if torch.is_tensor(v):
        if v.numel() == 1:
            return float(v.detach().cpu().item())
        return v.detach().cpu().tolist()
    return v


def _format_logs_for_print(logs: Dict[str, Any]) -> str:
    """Pretty console print: k=v pairs, stable order."""
    if not logs:
        return ""
    parts = []
    for k in sorted(logs.keys()):
        v = logs[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " | " + " ".join(parts)


def main() -> int:
    args = parse_args()

    # ---- load config ----
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)
    cfg["_config_path"] = args.config

    # ---- seed ----
    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("deterministic", True))
    set_seed(seed=seed, deterministic=deterministic)

    # ---- logging ----
    run_id = cfg.get("run_id") or make_run_id(prefix="train")
    logger = SimpleLogger(run_id=run_id, write_jsonl=True)
    logger.log("startup", seed=seed, deterministic=deterministic, config_path=args.config)

    print("\n=== LOADED CONFIG ===")
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print("=====================\n")

    # ---- toy data gate ----
    if not str2bool(args.debug_toy_data):
        logger.log("exit", reason="debug_toy_data is false. (Toy pipeline requires true)")
        print("Set --debug_toy_data true to run the toy pipeline.")
        return 0

    # ---- device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log("device", device=str(device))

    # ---- toy dataset/dataloader ----
    ds = ToyVQADataset(n=64, image_size=336, seed=seed)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=toy_vqa_collate)

    # ---- model ----
    model = NeuroBalanceModel(cfg).to(device)

    # ---- optimizer (use config if present) ----
    lr = cfg.get("training", {}).get("lr", 1e-4)
    try:
        lr = float(lr)  # YAML might store as string "5e-5"
    except Exception:
        lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    max_steps = args.max_steps if args.max_steps > 0 else 5
    step = 0
    model.train()

    for batch in cycle(dl):
        step += 1

        # Move tensors to device; keep strings/lists untouched
        batch = dict(batch)
        if "pixel_values" in batch and torch.is_tensor(batch["pixel_values"]):
            batch["pixel_values"] = batch["pixel_values"].to(device)

        out = model(batch)
        loss = out["loss"]
        logs = out.get("logs", {}) or {}

        # Make logs JSONL-safe
        logs = {k: _to_loggable(v) for k, v in logs.items()}

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # ✅ JSONL clean: each log key becomes a top-level field
        logger.log("train_step", step=step, loss=float(loss.item()), **logs)

        # ✅ Console clean: print k=v pairs (not a dict blob)
        print(
            f"Step {step}/{max_steps} | loss={loss.item():.4f}"
            + _format_logs_for_print(logs)
        )

        if step >= max_steps:
            break

    ckpt_path = save_checkpoint(model, optimizer, step=step, run_id=run_id)
    logger.log("checkpoint_saved", path=ckpt_path)

    print(f"\n✅ Toy training completed. Checkpoint saved to: {ckpt_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
