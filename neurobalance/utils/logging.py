from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class SimpleLogger:
    """
    Minimal logger:
      - prints to console
      - optionally writes JSONL into results/logs/<run_id>.jsonl

    Designed to work on both laptop and Colab.
    """
    run_id: str
    log_dir: str = "results/logs"
    write_jsonl: bool = True

    def __post_init__(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.log_dir, f"{self.run_id}.jsonl")

    def log(self, event: str, **fields: Any) -> None:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(),
            "event": event,
            **fields,
        }

        # Console output (human-friendly)
        printable = " ".join([f"{k}={v}" for k, v in payload.items() if k != "ts"])
        print(f"[{payload['ts']}] {printable}")

        # JSONL output (machine-friendly)
        if self.write_jsonl:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
