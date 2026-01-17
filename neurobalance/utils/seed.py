from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets seeds for reproducibility.
    If torch is installed, also sets CUDA seeds and deterministic flags.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Deterministic behavior (may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For full determinism in newer PyTorch versions:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
