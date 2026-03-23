# scripts/interpretability/dape_neuron_mining.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Tuple, Optional

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


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    p: [..., D] probabilities (sum to 1 over last dim)
    returns: [...] entropy
    """
    p = torch.clamp(p, min=eps, max=1.0)
    return -(p * p.log()).sum(dim=-1)


def softmax_probs(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DAPE-based neuron mining: find high-entropy (uncertain) vs low-entropy (confident) activation patterns."
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", type=str, action="append", default=[])

    # Single-sample mode (simple + robust)
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--answer", type=str, default="", help="Optional answer text (to include in prompt).")

    # Output
    p.add_argument("--outdir", type=str, default="results/interpretability")
    p.add_argument("--name", type=str, default="sample0")

    # Runtime
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument("--device", type=str, default="", help="cuda/cpu. Empty=auto.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", type=str, default="false")

    # Mining knobs
    p.add_argument("--layer", type=int, default=-1, help="Which hidden-state layer to mine (-1 = last layer).")
    p.add_argument("--topk_neurons", type=int, default=50, help="How many neuron dims to report.")
    p.add_argument("--token_index", type=int, default=-1, help="Which token to analyze (-1 last token).")
    p.add_argument("--use_abs", type=str, default="true", help="Rank neurons by abs activation.")
    p.add_argument("--json", type=str, default="true", help="Write JSON report.")
    return p.parse_args()


# -----------------------------
# Build fused inputs (same logic as your current switchboard)
# -----------------------------
@torch.no_grad()
def build_fused_inputs(
    model: NeuroBalanceModel,
    pixel_values: torch.Tensor,
    question: str,
    answer: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int, torch.Tensor]:
    """
    Returns:
      inputs_embeds: [1,S,H]
      attention_mask: [1,S]
      M: number of visual prefix tokens
      T: number of text tokens
      vis_tokens_H: [1,M,H] (already projected)
    """
    device = pixel_values.device

    # vision prefix -> projected to LM hidden size
    vis_tokens = model.backbone.vision(pixel_values)          # [1,M,Dv]
    vis_tokens = model.backbone.vision_to_lm(vis_tokens)      # [1,M,H]

    # proxy adaptive attention scaling (matches NeuroBalanceModel.forward())
    if model.adaptive_attention is not None:
        beta = model.adaptive_attention.beta
        vis_tokens = vis_tokens * (1.0 + beta)

    # prompt tokenize
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

    # gating/injection (to reflect cfg)
    if model.gating is not None:
        fused_embeds, _mask = model.gating(fused_embeds)

    if model.injection is not None:
        knowledge_vec = vis_tokens.mean(dim=1)  # [1,H]
        fused_embeds = model.injection(fused_embeds, knowledge_vec)

    # attention mask
    B, M, _ = vis_tokens.shape
    vis_mask = torch.ones((B, M), device=device, dtype=attn_mask.dtype)
    fused_mask = torch.cat([vis_mask, attn_mask], dim=1)               # [1,S]

    T = input_ids.shape[1]
    return fused_embeds, fused_mask, M, T, vis_tokens


@torch.no_grad()
def run_hidden_states(
    model: NeuroBalanceModel,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
    out = model.backbone.lm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    hs = out.hidden_states  # tuple: (embeddings, layer1, ..., layerN)
    return hs, hs[-1]


def pick_token(vecs: torch.Tensor, token_index: int) -> torch.Tensor:
    """
    vecs: [1,S,H]
    return: [H]
    """
    S = vecs.shape[1]
    idx = token_index if token_index >= 0 else (S + token_index)
    idx = max(0, min(S - 1, idx))
    return vecs[0, idx, :]


# -----------------------------
# DAPE-style mining
# -----------------------------
def dape_for_vector(h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A practical "DAPE-like" measure on a single token hidden vector h:[H].

    We convert activation magnitudes to a probability distribution over neurons,
    then compute entropy:
      p_j = softmax(|h_j|)
      DAPE = H(p)

    Returns:
      p: [H] probabilities
      H: scalar entropy
    """
    p = softmax_probs(h.abs(), dim=-1)
    H = entropy_from_probs(p, eps=1e-12)
    return p, H


def topk_report(h: torch.Tensor, k: int, use_abs: bool = True) -> Dict[str, Any]:
    """
    Rank neuron dimensions by (abs) activation and return top-k indices + values.
    """
    k = max(1, min(int(k), h.numel()))
    scores = h.abs() if use_abs else h
    vals, idx = torch.topk(scores, k=k, dim=-1)

    # also return signed activations for those dims
    signed = h[idx]

    return {
        "topk": int(k),
        "indices": idx.detach().cpu().tolist(),
        "scores": vals.detach().cpu().tolist(),
        "signed_activations": signed.detach().cpu().tolist(),
    }


def tailk_report(h: torch.Tensor, k: int, use_abs: bool = True) -> Dict[str, Any]:
    """
    Return bottom-k (least active) dims.
    """
    k = max(1, min(int(k), h.numel()))
    scores = h.abs() if use_abs else h
    vals, idx = torch.topk(scores, k=k, largest=False, dim=-1)
    signed = h[idx]
    return {
        "tailk": int(k),
        "indices": idx.detach().cpu().tolist(),
        "scores": vals.detach().cpu().tolist(),
        "signed_activations": signed.detach().cpu().tolist(),
    }


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    set_seed(seed=int(args.seed), deterministic=str2bool(args.deterministic))

    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NeuroBalanceModel(cfg).to(device).eval()

    pixel_values = load_image_as_tensor(args.image_path, image_size=int(args.image_size)).to(device)
    inputs_embeds, attention_mask, M, T, _ = build_fused_inputs(
        model=model,
        pixel_values=pixel_values,
        question=args.question,
        answer=args.answer if args.answer else None,
    )

    hidden_states, _ = run_hidden_states(model, inputs_embeds, attention_mask)

    # choose layer
    L = len(hidden_states)  # includes embedding layer at index 0
    layer = int(args.layer)
    if layer < 0:
        layer = L + layer
    layer = max(0, min(L - 1, layer))

    # token vector
    h_tok = pick_token(hidden_states[layer], token_index=int(args.token_index))  # [H]

    # DAPE-like entropy
    p, H = dape_for_vector(h_tok)

    # Top-k / Tail-k mining
    use_abs = str2bool(args.use_abs)
    topk = topk_report(h_tok, k=int(args.topk_neurons), use_abs=use_abs)
    tailk = tailk_report(h_tok, k=min(int(args.topk_neurons), 50), use_abs=use_abs)

    report: Dict[str, Any] = {
        "name": args.name,
        "config": args.config,
        "device": str(device),
        "sequence": {
            "visual_tokens_M": int(M),
            "text_tokens_T": int(T),
            "total_S": int(M + T),
            "layer_used": int(layer),
            "num_hidden_states": int(L),
            "token_index": int(args.token_index),
        },
        "dape": {
            "definition": "DAPE-like entropy of neuron activation probabilities p=softmax(|h|).",
            "entropy": float(H.detach().cpu().item()),
            "prob_top1": float(p.max().detach().cpu().item()),
            "prob_top5_sum": float(torch.topk(p, k=min(5, p.numel())).values.sum().detach().cpu().item()),
        },
        "mining": {
            "ranking": "abs_activation" if use_abs else "signed_activation",
            "topk": topk,
            "tailk": tailk,
        },
        "notes": {
            "interpretation": "Lower entropy => activation mass concentrated on few neurons (more 'peaky' / selective). Higher entropy => more spread (less selective).",
            "use_case": "Compare baseline vs gating/partial/full: gating should increase selectivity (often lowering entropy) and shift top-k neuron sets.",
        },
    }

    # Print short summary
    print("\n=== DAPE Neuron Mining Summary ===")
    print("device:", report["device"])
    print(f"Layer={layer}/{L-1} | TokenIndex={report['sequence']['token_index']} | HiddenDim={h_tok.numel()}")
    print(f"DAPE entropy={report['dape']['entropy']:.4f} | prob_top1={report['dape']['prob_top1']:.4f}")
    print("Top-10 neuron indices:", report["mining"]["topk"]["indices"][:10])
    print("=================================\n")

    ensure_dir(args.outdir)
    if str2bool(args.json):
        out_path = os.path.join(args.outdir, f"dape_neuron_mining_{args.name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("✅ Saved JSON:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
