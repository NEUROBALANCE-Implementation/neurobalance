from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except Exception:
    torch = None


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets seeds for reproducibility.

    Notes for CUDA determinism:
    - If deterministic=True and CUDA is used, PyTorch may require:
        CUBLAS_WORKSPACE_CONFIG=:4096:8  (or :16:8)
      to make some GEMM operations deterministic.
    - We set it here (early) so training won't crash on Colab GPUs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    # ---- seeds ----
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if not deterministic:
        return

    # ---- determinism knobs ----
    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Disable TF32 for closer determinism (optional but recommended)
    # (TF32 can change numerical results slightly on Ampere+ GPUs)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    # cuBLAS reproducibility requirement for CUDA >= 10.2
    # Must be set BEFORE the first cuBLAS call.
    # Use setdefault so user can override externally if they want.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # For full determinism in newer PyTorch versions
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
