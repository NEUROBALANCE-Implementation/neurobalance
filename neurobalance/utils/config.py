from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/dict, got: {type(data)}")
    return data


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    """
    Apply CLI overrides in KEY=VALUE format.
    Supports nested keys using dot notation, e.g.:
      training.lr=5e-5
      model.name=llava
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Override must be KEY=VALUE, got: {ov}")
        key, raw_val = ov.split("=", 1)
        val: Any = _parse_scalar(raw_val)

        # nested set
        keys = key.split(".")
        cur = cfg
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = val
    return cfg


def _parse_scalar(x: str) -> Any:
    """
    Parse a simple scalar from string:
      "true"/"false" -> bool
      numbers -> int/float
      otherwise -> str
    """
    xl = x.strip().lower()
    if xl in ("true", "false"):
        return xl == "true"

    # int?
    try:
        if xl.startswith("0") and len(xl) > 1 and xl[1].isdigit():
            # keep as string to avoid octal-like confusion
            return x
        return int(x)
    except Exception:
        pass

    # float?
    try:
        return float(x)
    except Exception:
        pass

    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NEUROBALANCE")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override config values: KEY=VALUE (can be repeated)",
    )
    return p.parse_args()


def load_config_from_cli() -> Dict[str, Any]:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg["_config_path"] = args.config
    cfg["_overrides"] = args.override
    return apply_overrides(cfg, args.override)
