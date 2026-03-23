# scripts/interpretability/logit_lens.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

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
    (Keep it simple/consistent with toy pipeline.)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def softmax_probs(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)


def entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, min=eps, max=1.0)
    return -(p * p.log()).sum(dim=-1)


def topk_tokens(
    logits: torch.Tensor,
    tokenizer,
    k: int = 10,
) -> List[Tuple[str, float, int]]:
    """
    logits: [V]
    Returns list of (token_str, prob, token_id)
    """
    k = max(1, min(int(k), logits.numel()))
    probs = softmax_probs(logits, dim=-1)
    vals, idx = torch.topk(probs, k=k, dim=-1)
    out: List[Tuple[str, float, int]] = []
    for p, tid in zip(vals.detach().cpu().tolist(), idx.detach().cpu().tolist()):
        tok = tokenizer.decode([tid])
        out.append((tok, float(p), int(tid)))
    return out


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Logit Lens: decode intermediate hidden states into token distributions to inspect evolving predictions."
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", type=str, action="append", default=[])

    # Single-sample mode
    p.add_argument("--image_path", type=str, required=True)
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--answer", type=str, default="", help="Optional answer to include in prompt (teacher forcing prompt).")

    # Output
    p.add_argument("--outdir", type=str, default="results/interpretability")
    p.add_argument("--name", type=str, default="sample0")
    p.add_argument("--json", type=str, default="true", help="Write JSON report.")

    # Runtime
    p.add_argument("--image_size", type=int, default=336)
    p.add_argument("--device", type=str, default="", help="cuda/cpu. Empty=auto.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", type=str, default="false")

    # Lens knobs
    p.add_argument("--layers", type=str, default="0,-1", help="Comma list of layers to inspect (e.g., '0,3,6,-1').")
    p.add_argument("--token_index", type=int, default=-1, help="Which sequence position to decode (-1 last).")
    p.add_argument("--topk", type=int, default=10, help="Top-K tokens to show per layer.")
    p.add_argument("--show_entropy", type=str, default="true", help="Compute entropy of token distribution.")
    return p.parse_args()


# -----------------------------
# Build fused inputs (match your switchboard)
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
      M: number of visual tokens
      T: number of text tokens
    """
    device = pixel_values.device

    vis_tokens = model.backbone.vision(pixel_values)          # [1,M,Dv]
    vis_tokens = model.backbone.vision_to_lm(vis_tokens)      # [1,M,H]

    # proxy adaptive attention scaling
    if model.adaptive_attention is not None:
        beta = model.adaptive_attention.beta
        vis_tokens = vis_tokens * (1.0 + beta)

    # build prompt and tokenize
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

    # gating + injection to reflect cfg
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
    return fused_embeds, fused_mask, int(M), int(T)


@torch.no_grad()
def run_hidden_states(
    model: NeuroBalanceModel,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[Tuple[torch.Tensor, ...], int]:
    """
    Returns hidden states tuple and hidden size.
    hidden_states includes embedding layer at index 0.
    """
    out = model.backbone.lm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        output_hidden_states=True,
        output_attentions=False,
        return_dict=True,
    )
    hs = out.hidden_states
    hidden_size = hs[-1].shape[-1]
    return hs, int(hidden_size)


def parse_layer_list(spec: str, max_layer: int) -> List[int]:
    """
    spec: "0,3,6,-1"
    max_layer: last valid index in hidden_states tuple
    """
    raw = []
    for s in spec.split(","):
        s = s.strip()
        if not s:
            continue
        raw.append(int(s))

    out: List[int] = []
    for l in raw:
        if l < 0:
            l = (max_layer + 1) + l
        l = max(0, min(max_layer, l))
        out.append(l)

    # unique but stable order
    seen = set()
    uniq = []
    for l in out:
        if l not in seen:
            seen.add(l)
            uniq.append(l)
    return uniq


def pick_token_vec(layer_h: torch.Tensor, token_index: int) -> torch.Tensor:
    """
    layer_h: [1,S,H]
    returns: [H]
    """
    S = layer_h.shape[1]
    idx = token_index if token_index >= 0 else (S + token_index)
    idx = max(0, min(S - 1, idx))
    return layer_h[0, idx, :]


@torch.no_grad()
def decode_logit_lens(
    model: NeuroBalanceModel,
    h: torch.Tensor,
) -> torch.Tensor:
    """
    h: [H] hidden state at some layer & token position
    returns logits [V] using model's LM head.
    GPT2-style models apply final layernorm before lm_head in standard forward,
    but for a simple logit lens, projecting directly via lm_head is common.

    If your backbone is GPT2 (AutoModelForCausalLM), lm_head is model.backbone.lm.lm_head
    """
    lm = model.backbone.lm
    if hasattr(lm, "lm_head"):
        # [V,H] weight in Linear expects [*,H]
        return lm.lm_head(h)
    # fallback: try get_output_embeddings
    out_emb = lm.get_output_embeddings()
    return out_emb(h)


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

    inputs_embeds, attention_mask, M, T = build_fused_inputs(
        model=model,
        pixel_values=pixel_values,
        question=args.question,
        answer=args.answer if args.answer else None,
    )

    hidden_states, hidden_size = run_hidden_states(model, inputs_embeds, attention_mask)
    L = len(hidden_states)  # includes embedding layer at 0
    max_layer = L - 1

    layers = parse_layer_list(args.layers, max_layer=max_layer)
    token_index = int(args.token_index)
    topk = int(args.topk)
    show_entropy = str2bool(args.show_entropy)

    report: Dict[str, Any] = {
        "name": args.name,
        "config": args.config,
        "device": str(device),
        "sequence": {
            "visual_tokens_M": int(M),
            "text_tokens_T": int(T),
            "total_S": int(M + T),
            "token_index": int(token_index),
            "hidden_size": int(hidden_size),
            "num_hidden_states": int(L),
        },
        "lens": {
            "layers_requested": args.layers,
            "layers_used": layers,
            "topk": topk,
            "entropy_enabled": bool(show_entropy),
        },
        "results": [],
    }

    print("\n=== Logit Lens Summary ===")
    print("device:", report["device"])
    print(f"Total hidden states={L} (0=embeds ... {max_layer}=last)")
    print(f"Token index={token_index} | Visual M={M} | Text T={T} | Total S={M+T}")
    print("=========================\n")

    tok = model.backbone.tokenizer

    for l in layers:
        h_tok = pick_token_vec(hidden_states[l], token_index=token_index)  # [H]
        logits = decode_logit_lens(model, h_tok)  # [V]

        top = topk_tokens(logits, tokenizer=tok, k=topk)

        layer_entry: Dict[str, Any] = {
            "layer": int(l),
            "topk_tokens": [{"token": t, "prob": p, "token_id": tid} for (t, p, tid) in top],
        }

        if show_entropy:
            probs = softmax_probs(logits, dim=-1)
            ent = float(entropy_from_probs(probs).detach().cpu().item())
            layer_entry["entropy"] = ent

        report["results"].append(layer_entry)

        # console print (human-friendly)
        print(f"[Layer {l}]")
        if show_entropy:
            print(f"  entropy={layer_entry['entropy']:.4f}")
        for i, item in enumerate(layer_entry["topk_tokens"][: min(10, topk)], start=1):
            # show token safely
            token_show = item["token"].replace("\n", "\\n")
            print(f"  {i:>2}. {token_show!r}  prob={item['prob']:.4f}  id={item['token_id']}")
        print()

    ensure_dir(args.outdir)
    if str2bool(args.json):
        out_path = os.path.join(args.outdir, f"logit_lens_{args.name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print("✅ Saved JSON:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
