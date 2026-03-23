# scripts/train.py
from __future__ import annotations

import argparse
import json
import os
import re
from itertools import cycle
from pathlib import Path
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

    # ✅ Step 8.2: Drive persistence + auto-resume
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--resume", type=str, default="true")

    return p.parse_args()


def str2bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "yes", "y")


def _to_loggable(v: Any) -> Any:
    if torch.is_tensor(v):
        if v.numel() == 1:
            return float(v.detach().cpu().item())
        return v.detach().cpu().tolist()
    return v


def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    p = Path(ckpt_dir)
    if not p.exists():
        return None
    cands = list(p.glob("*.pt"))
    if not cands:
        return None

    def step_num(fp: Path) -> int:
        m = re.search(r"_step(\d+)\.pt$", fp.name)
        return int(m.group(1)) if m else -1

    cands.sort(key=step_num, reverse=True)
    return str(cands[0])


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    return int(ckpt.get("step", 0))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    run_id: str,
    outdir: str,
) -> str:
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{run_id}_step{step}.pt")
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )
    return path


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

    # ---- outputs ----
    os.makedirs(args.outdir, exist_ok=True)
    logger.log("outdir", outdir=args.outdir)

    # ---- toy dataset/dataloader ----
    ds = ToyVQADataset(n=64, image_size=336, seed=seed)
    dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=toy_vqa_collate)

    # ---- model ----
    model = NeuroBalanceModel(cfg).to(device)

    # ---- optimizer ----
    lr = cfg.get("training", {}).get("lr", 1e-4)
    try:
        lr = float(lr)
    except Exception:
        lr = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ---- resume ----
    start_step = 0
    if str2bool(args.resume):
        ckpt_dir = os.path.join(args.outdir, "checkpoints")
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            start_step = load_checkpoint(latest, model, optimizer)
            logger.log("resume", checkpoint=latest, start_step=start_step)
            print(f"✅ Resumed from {latest} at step={start_step}")

    # ---- train loop ----
    max_steps = args.max_steps if args.max_steps > 0 else 5
    step = start_step
    model.train()

    for batch in cycle(dl):
        step += 1

        batch = dict(batch)
        if "pixel_values" in batch and torch.is_tensor(batch["pixel_values"]):
            batch["pixel_values"] = batch["pixel_values"].to(device)

        out = model(batch)
        loss = out["loss"]
        logs = out.get("logs", {}) or {}
        logs = {k: _to_loggable(v) for k, v in logs.items()}

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        logger.log("train_step", step=step, loss=float(loss.item()), **logs)

        # Save every N steps
        if args.save_every > 0 and (step % args.save_every == 0):
            ckpt_path = save_checkpoint(model, optimizer, step=step, run_id=run_id, outdir=args.outdir)
            logger.log("checkpoint_saved", path=ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        print(f"Step {step}/{max_steps} | loss={loss.item():.4f} | logs={logs}")

        if step >= max_steps:
            break

    # Final save
    ckpt_path = save_checkpoint(model, optimizer, step=step, run_id=run_id, outdir=args.outdir)
    logger.log("checkpoint_saved", path=ckpt_path)
    print(f"\n✅ Training completed. Final checkpoint: {ckpt_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
