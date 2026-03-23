# scripts/interpretability/attention_heatmaps.py
from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from neurobalance.utils.config import load_yaml, apply_overrides
from neurobalance.utils.seed import set_seed
from neurobalance.models.neurobalance_model import NeuroBalanceModel


# -----------------------------
# Helpers
# -----------------------------
def str2bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "yes", "y")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_image_as_tensor(image_path: str, image_size: int = 336) -> torch.Tensor:
    """
    Returns pixel_values tensor: [1,3,H,W] float32 in [0,1].
    (Keep it simple; you can later add ImageNet norm to match training.)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    # [H,W,3] -> [1,3,H,W]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate attention heatmaps (GPT2 self-attn) for a prompt.")
    p.add_argument("--config", type=str, required=True, help="YAML config path (baseline/partial/full).")
    p.add_argument("--override", type=str, action="append", default=[], help="Optional overrides key=value.")
    p.add_argument("--image_path", type=str, required=True, help="Path to an image file.")
    p.add_argument("--question", type=str, required=True, help="Question string.")
    p.add_argument("--answer", type=str, default="", help="Optional answer (to include in prompt).")
    p.add_argument("--layer", type=int, default=-1, help="Which layer to visualize (-1 = last).")
    p.add_argument("--head", type=int, default=0, help="Which head to visualize.")
    p.add_argument("--outdir", type=str, default="results/interpretability", help="Output directory.")
    p.add_argument("--image_size", type=int, default=336, help="Image resize (square).")
    p.add_argument("--device", type=str, default="", help="cuda/cpu. Empty = auto.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", type=str, default="false", help="Force deterministic (can break on GPU).")
    p.add_argument("--show", type=str, default="false", help="Show plot interactively.")
    return p.parse_args()


# -----------------------------
# Main extraction logic
# -----------------------------
@torch.no_grad()
def build_fused_inputs(
    model: NeuroBalanceModel,
    pixel_values: torch.Tensor,
    question: str,
    answer: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Rebuilds the same fused embeddings path used inside NeuroBalanceModel forward,
    but returns (inputs_embeds, attention_mask, token_strings_for_text_part).

    Notes:
    - The HF GPT2 attention output corresponds to the fused sequence length (visual prefix + text tokens)
      only if we pass inputs_embeds and ask output_attentions=True.
    - Token strings: for labeling only (we label just the text tokens for clarity).
    """
    device = pixel_values.device

    # 1) vision tokens (prefix)
    vis_tokens = model.backbone.vision(pixel_values)          # [B,M,Dv]
    vis_tokens = model.backbone.vision_to_lm(vis_tokens)      # [B,M,H]

    # Apply the same proxy "adaptive attention" effect as your NeuroBalanceModel does (optional)
    if model.adaptive_attention is not None:
        beta = model.adaptive_attention.beta
        vis_tokens = vis_tokens * (1.0 + beta)

    # 2) prompt tokenize
    questions = [question]
    answers = [answer] if (answer is not None and answer != "") else None
    prompts = model.backbone.build_prompts(questions, answers=answers)

    tok = model.backbone.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=model.backbone.cfg.max_text_len,
        return_tensors="pt",
    ).to(device)

    input_ids = tok["input_ids"]          # [1,T]
    attn_mask = tok["attention_mask"]     # [1,T]

    # 3) text embeds
    text_embeds = model.backbone.lm.get_input_embeddings()(input_ids)  # [1,T,H]

    # 4) fuse
    fused_embeds = torch.cat([vis_tokens, text_embeds], dim=1)         # [1,M+T,H]

    # 5) optional sparse gating + injection (to match your switchboard behavior)
    if model.gating is not None:
        fused_embeds, _mask = model.gating(fused_embeds)

    if model.injection is not None:
        knowledge_vec = vis_tokens.mean(dim=1)  # [1,H]
        fused_embeds = model.injection(fused_embeds, knowledge_vec)

    # 6) attention mask for fused seq
    B, M, _ = vis_tokens.shape
    vis_mask = torch.ones((B, M), device=device, dtype=attn_mask.dtype)
    fused_mask = torch.cat([vis_mask, attn_mask], dim=1)               # [1,M+T]

    # token strings for text part
    text_tokens = model.backbone.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return fused_embeds, fused_mask, text_tokens


@torch.no_grad()
def extract_attentions(
    model: NeuroBalanceModel,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Runs the LM with output_attentions=True and returns:
      attentions: tuple(layer) of [B, heads, S, S]
      logits: [B, S, vocab]
    """
    out = model.backbone.lm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=True,
        output_hidden_states=False,
        return_dict=True,
    )
    # out.attentions is a tuple of length n_layers
    # each element: [B, n_heads, S, S]
    return out.attentions, out.logits


def plot_attention_heatmap(
    attn: np.ndarray,
    title: str,
    out_path: str,
    xticks: Optional[List[str]] = None,
    yticks: Optional[List[str]] = None,
    max_ticks: int = 60,
) -> None:
    """
    attn: [S,S] numpy
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(attn, aspect="auto")
    plt.title(title)
    plt.xlabel("Key positions")
    plt.ylabel("Query positions")
    plt.colorbar()

    S = attn.shape[0]
    if xticks is not None and yticks is not None:
        # Reduce tick clutter
        step = max(1, int(math.ceil(S / max_ticks)))
        idx = list(range(0, S, step))
        plt.xticks(idx, [xticks[i] for i in idx], rotation=90, fontsize=7)
        plt.yticks(idx, [yticks[i] for i in idx], fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    # seed / determinism
    set_seed(seed=int(args.seed), deterministic=str2bool(args.deterministic))

    # device
    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = NeuroBalanceModel(cfg).to(device)
    model.eval()

    # load image
    pixel_values = load_image_as_tensor(args.image_path, image_size=int(args.image_size)).to(device)

    # build fused inputs
    fused_embeds, fused_mask, text_tokens = build_fused_inputs(
        model=model,
        pixel_values=pixel_values,
        question=args.question,
        answer=args.answer if args.answer else None,
    )

    # run LM to get attention weights
    attentions, _logits = extract_attentions(model, fused_embeds, fused_mask)

    n_layers = len(attentions)
    layer_idx = args.layer if args.layer >= 0 else (n_layers - 1)
    if layer_idx < 0 or layer_idx >= n_layers:
        raise ValueError(f"Invalid --layer {args.layer}. Model has {n_layers} layers.")

    attn_layer = attentions[layer_idx][0]  # [heads, S, S] because B=1
    n_heads = attn_layer.shape[0]
    head_idx = int(args.head)
    if head_idx < 0 or head_idx >= n_heads:
        raise ValueError(f"Invalid --head {head_idx}. Layer has {n_heads} heads.")

    attn_head = attn_layer[head_idx].detach().cpu().float().numpy()  # [S,S]

    # build simple labels (visual prefix positions + text tokens)
    # We don’t know exact visual-token semantics, so label them V0..VM-1
    S = attn_head.shape[0]
    # derive M = visual prefix length from fused embeds: [1, M+T, H]
    M = fused_embeds.shape[1] - len(text_tokens)
    labels = [f"V{i}" for i in range(M)] + text_tokens
    labels = labels[:S]  # safety

    # output paths
    ensure_dir(args.outdir)
    out_path = os.path.join(
        args.outdir,
        f"attention_heatmap_layer{layer_idx}_head{head_idx}.png",
    )

    title = f"Attention Heatmap | layer={layer_idx} head={head_idx} | S={S}"
    plot_attention_heatmap(attn_head, title=title, out_path=out_path, xticks=labels, yticks=labels)

    print("✅ Saved:", out_path)

    if str2bool(args.show):
        img = Image.open(out_path)
        img.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
