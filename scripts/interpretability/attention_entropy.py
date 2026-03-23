# scripts/interpretability/attention_entropy.py
from __future__ import annotations

import argparse
import os
import math
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from PIL import Image

from neurobalance.utils.config import load_yaml, apply_overrides
from neurobalance.utils.seed import set_seed
from neurobalance.models.neurobalance_model import NeuroBalanceModel


# -----------------------------
# Utils
# -----------------------------
def str2bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "yes", "y")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_image_as_tensor(image_path: str, image_size: int = 336) -> torch.Tensor:
    """
    Returns pixel_values tensor: [1,3,H,W] float32 in [0,1].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def safe_entropy(p: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Shannon entropy H(p) = -sum p log p computed safely.
    p: probability distribution along `axis`.
    Returns entropy with that axis reduced.
    """
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute attention entropy statistics (GPT2 self-attn).")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", type=str, action="append", default=[])
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--answer", type=str, default="", help="Optional answer (to include in prompt).")
    p.add_argument("--outdir", type=str, default="results/interpretability")
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument("--device", type=str, default="", help="cuda/cpu. Empty=auto.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", type=str, default="false")
    p.add_argument("--json", type=str, default="true", help="Write a JSON summary file.")
    return p.parse_args()


# -----------------------------
# Core: build inputs exactly like your switchboard path
# -----------------------------
@torch.no_grad()
def build_fused_inputs(
    model: NeuroBalanceModel,
    pixel_values: torch.Tensor,
    question: str,
    answer: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Returns:
      inputs_embeds: [1,S,H]
      attention_mask: [1,S]
      M: number of visual prefix tokens
      T: number of text tokens
    """
    device = pixel_values.device

    # vision prefix
    vis_tokens = model.backbone.vision(pixel_values)          # [1,M,Dv]
    vis_tokens = model.backbone.vision_to_lm(vis_tokens)      # [1,M,H]

    # same proxy attention scaling used in NeuroBalanceModel (if enabled)
    if model.adaptive_attention is not None:
        beta = model.adaptive_attention.beta
        vis_tokens = vis_tokens * (1.0 + beta)

    # tokenize prompt
    prompts = model.backbone.build_prompts([question], answers=[answer] if (answer and answer != "") else None)
    tok = model.backbone.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=model.backbone.cfg.max_text_len,
        return_tensors="pt",
    ).to(device)

    input_ids = tok["input_ids"]          # [1,T]
    attn_mask = tok["attention_mask"]     # [1,T]

    text_embeds = model.backbone.lm.get_input_embeddings()(input_ids)  # [1,T,H]
    fused_embeds = torch.cat([vis_tokens, text_embeds], dim=1)         # [1,S,H]

    # optional gating + injection (to reflect the active config)
    if model.gating is not None:
        fused_embeds, _ = model.gating(fused_embeds)

    if model.injection is not None:
        knowledge_vec = vis_tokens.mean(dim=1)  # [1,H]
        fused_embeds = model.injection(fused_embeds, knowledge_vec)

    # fused attention mask
    B, M, _ = vis_tokens.shape
    vis_mask = torch.ones((B, M), device=device, dtype=attn_mask.dtype)
    fused_mask = torch.cat([vis_mask, attn_mask], dim=1)               # [1,S]

    T = input_ids.shape[1]
    return fused_embeds, fused_mask, M, T


@torch.no_grad()
def get_attentions(
    model: NeuroBalanceModel,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], int, int]:
    """
    Returns:
      attentions: tuple of [1, heads, S, S] per layer
      n_layers
      n_heads (for layer 0)
    """
    out = model.backbone.lm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    attns = out.attentions
    n_layers = len(attns)
    n_heads = attns[0].shape[1] if n_layers > 0 else 0
    return attns, n_layers, n_heads


def summarize_entropy(attentions: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
    """
    Compute entropy over the key dimension for each query position:
      For each layer L and head H: attn[L,H,q,:] is a distribution over keys
      H_entropy[L,H,q] = -sum_k p log p

    Then report:
      - mean entropy per layer (averaged over heads and queries)
      - mean entropy per head (averaged over layers and queries)
      - global mean/std/min/max
    """
    layer_means: List[float] = []
    head_means_accum: Optional[np.ndarray] = None
    global_vals: List[float] = []

    n_layers = len(attentions)
    for l in range(n_layers):
        # [B=1, heads, S, S] -> [heads, S, S]
        attn = attentions[l][0].detach().cpu().float().numpy()
        # entropy over keys (last dim): -> [heads, S]
        ent = safe_entropy(attn, axis=-1)
        # mean over queries S and heads
        layer_mean = float(ent.mean())
        layer_means.append(layer_mean)

        if head_means_accum is None:
            head_means_accum = ent.mean(axis=1)  # [heads]
        else:
            head_means_accum += ent.mean(axis=1)

        global_vals.extend(ent.reshape(-1).tolist())

    global_arr = np.array(global_vals, dtype=np.float64) if global_vals else np.array([0.0], dtype=np.float64)

    head_means = []
    if head_means_accum is not None and n_layers > 0:
        head_means = (head_means_accum / float(n_layers)).astype(np.float64).tolist()

    return {
        "layer_mean_entropy": layer_means,
        "head_mean_entropy": head_means,
        "global": {
            "mean": float(global_arr.mean()),
            "std": float(global_arr.std(ddof=0)),
            "min": float(global_arr.min()),
            "max": float(global_arr.max()),
        },
    }


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    set_seed(seed=int(args.seed), deterministic=str2bool(args.deterministic))

    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuroBalanceModel(cfg).to(device)
    model.eval()

    pixel_values = load_image_as_tensor(args.image_path, image_size=int(args.image_size)).to(device)

    fused_embeds, fused_mask, M, T = build_fused_inputs(
        model=model,
        pixel_values=pixel_values,
        question=args.question,
        answer=args.answer if args.answer else None,
    )

    attentions, n_layers, n_heads = get_attentions(model, fused_embeds, fused_mask)

    summary = summarize_entropy(attentions)

    # Add extra context
    summary.update(
        {
            "config": args.config,
            "device": str(device),
            "sequence": {
                "visual_tokens_M": int(M),
                "text_tokens_T": int(T),
                "total_S": int(M + T),
            },
            "model": {
                "n_layers": int(n_layers),
                "n_heads": int(n_heads),
            },
            "notes": {
                "entropy_definition": "Shannon entropy over attention distribution p(keys|query). Lower = sharper/more focused attention.",
                "proxy_warning": "This script measures GPT2 self-attention entropy for the fused (visual+text) sequence. Your 'adaptive attention reweighting' is currently proxied via scaling visual token embeddings.",
            },
        }
    )

    # Print a compact console report
    print("\n=== Attention Entropy Summary ===")
    print(f"device: {summary['device']}")
    print(f"layers: {summary['model']['n_layers']} | heads: {summary['model']['n_heads']}")
    print(f"S={summary['sequence']['total_S']} (M={summary['sequence']['visual_tokens_M']} + T={summary['sequence']['text_tokens_T']})")
    print(f"global mean={summary['global']['mean']:.4f} std={summary['global']['std']:.4f} min={summary['global']['min']:.4f} max={summary['global']['max']:.4f}")
    print("per-layer mean entropy:", [round(x, 4) for x in summary["layer_mean_entropy"]][:8], "..." if len(summary["layer_mean_entropy"]) > 8 else "")
    print("================================\n")

    ensure_dir(args.outdir)
    if str2bool(args.json):
        out_path = os.path.join(args.outdir, "attention_entropy_summary.json")
        import json

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("✅ Saved JSON:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
