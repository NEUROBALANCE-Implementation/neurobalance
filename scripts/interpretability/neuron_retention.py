# scripts/interpretability/neuron_retention.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
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
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t


def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity over the last dim. Shapes must broadcast.
    """
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_n * b_n).sum(dim=-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neuron retention analysis (hidden state retention across steps/layers).")
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

    # Retention settings
    p.add_argument("--topk", type=int, default=64, help="How many 'most active' dims to track for retention.")
    p.add_argument("--token_index", type=int, default=-1, help="Which token to track (-1 = last token).")
    p.add_argument("--json", type=str, default="true", help="Write a JSON summary file.")
    return p.parse_args()


# -----------------------------
# Core: build fused inputs like NeuroBalanceModel.forward()
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

    # proxy attention scaling (matches your current switchboard)
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

    # gating & injection (to reflect config)
    gate_mask = None
    if model.gating is not None:
        fused_embeds, gate_mask = model.gating(fused_embeds)

    if model.injection is not None:
        knowledge_vec = vis_tokens.mean(dim=1)  # [1,H]
        fused_embeds = model.injection(fused_embeds, knowledge_vec)

    # fused attention mask
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
    """
    Returns:
      hidden_states: tuple of tensors (embeddings + each layer output)
          each: [1,S,H]
      last_hidden: [1,S,H]
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
    last = hs[-1]
    return hs, last


def pick_token(h: torch.Tensor, token_index: int) -> torch.Tensor:
    """
    h: [1,S,H]
    return: [H]
    """
    S = h.shape[1]
    idx = token_index if token_index >= 0 else (S + token_index)
    idx = max(0, min(S - 1, idx))
    return h[0, idx, :]


def topk_dims(vec: torch.Tensor, k: int) -> torch.Tensor:
    """
    vec: [H]
    returns indices [k] of largest absolute activations.
    """
    k = max(1, min(int(k), vec.numel()))
    return torch.topk(vec.abs(), k=k, dim=-1).indices


def retention_metrics_over_layers(hidden_states: Tuple[torch.Tensor, ...], token_index: int, k: int) -> Dict[str, Any]:
    """
    Track a fixed set of top-k dims selected at the *input embedding layer* (hidden_states[0])
    and report how well those dimensions are retained across layers.

    Metrics reported per layer:
      - cosine similarity of the top-k subvector vs reference (layer0)
      - mean absolute activation on tracked dims
      - mean signed activation on tracked dims
    """
    # reference dims chosen at layer 0 token vector
    ref_vec = pick_token(hidden_states[0], token_index=token_index)  # [H]
    idx = topk_dims(ref_vec, k=k)                                    # [k]
    ref_sub = ref_vec[idx]                                           # [k]

    per_layer = []
    for l, h in enumerate(hidden_states):
        v = pick_token(h, token_index=token_index)     # [H]
        sub = v[idx]                                   # [k]
        cos = float(cosine_sim(sub, ref_sub).detach().cpu().item())
        mean_abs = float(sub.abs().mean().detach().cpu().item())
        mean_signed = float(sub.mean().detach().cpu().item())
        per_layer.append(
            {
                "layer": int(l),
                "cosine_to_layer0_topk": cos,
                "mean_abs_topk": mean_abs,
                "mean_signed_topk": mean_signed,
            }
        )

    return {
        "tracked_topk": int(k),
        "tracked_dims": idx.detach().cpu().tolist(),
        "per_layer": per_layer,
    }


def compare_with_vs_without_injection(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """
    If injection is enabled in config, compute retention twice:
      - as-is
      - with injection temporarily disabled
    This gives "injection improves retention" evidence on the same sample.

    NOTE: This is a controlled interpretability diagnostic, not a training claim.
    """
    # Run 1: as-is (possibly injection enabled)
    model_on = NeuroBalanceModel(cfg).to(device).eval()
    pv = load_image_as_tensor(args.image_path, image_size=int(args.image_size)).to(device)

    embeds_on, mask_on, M, T, _ = build_fused_inputs(
        model=model_on, pixel_values=pv, question=args.question, answer=args.answer if args.answer else None
    )
    hs_on, _ = run_hidden_states(model_on, embeds_on, mask_on)
    ret_on = retention_metrics_over_layers(hs_on, token_index=int(args.token_index), k=int(args.topk))

    # Run 2: force injection off via override
    cfg_off = json.loads(json.dumps(cfg))  # safe deep copy
    if "neurobalance" in cfg_off and "injection" in cfg_off["neurobalance"]:
        cfg_off["neurobalance"]["injection"]["enabled"] = False
    else:
        cfg_off.setdefault("neurobalance", {}).setdefault("injection", {})["enabled"] = False

    model_off = NeuroBalanceModel(cfg_off).to(device).eval()
    embeds_off, mask_off, _, _, _ = build_fused_inputs(
        model=model_off, pixel_values=pv, question=args.question, answer=args.answer if args.answer else None
    )
    hs_off, _ = run_hidden_states(model_off, embeds_off, mask_off)
    ret_off = retention_metrics_over_layers(hs_off, token_index=int(args.token_index), k=int(args.topk))

    # Compare (cosine improvement at last layer)
    cos_on_last = ret_on["per_layer"][-1]["cosine_to_layer0_topk"]
    cos_off_last = ret_off["per_layer"][-1]["cosine_to_layer0_topk"]

    return {
        "with_injection": ret_on,
        "without_injection": ret_off,
        "delta_last_layer_cosine": float(cos_on_last - cos_off_last),
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

    # Determine if injection is enabled in cfg
    inj_enabled = bool(
        cfg.get("neurobalance", {}).get("injection", {}).get("enabled", False)
    )

    # Run retention analysis (always), and if injection enabled also run a comparison on/off
    model = NeuroBalanceModel(cfg).to(device).eval()

    pixel_values = load_image_as_tensor(args.image_path, image_size=int(args.image_size)).to(device)
    inputs_embeds, attention_mask, M, T, _ = build_fused_inputs(
        model=model,
        pixel_values=pixel_values,
        question=args.question,
        answer=args.answer if args.answer else None,
    )
    hidden_states, _ = run_hidden_states(model, inputs_embeds, attention_mask)

    retention = retention_metrics_over_layers(hidden_states, token_index=int(args.token_index), k=int(args.topk))

    summary: Dict[str, Any] = {
        "config": args.config,
        "device": str(device),
        "sequence": {
            "visual_tokens_M": int(M),
            "text_tokens_T": int(T),
            "total_S": int(M + T),
            "token_index_tracked": int(args.token_index),
        },
        "retention": retention,
        "notes": {
            "what_is_measured": "Tracks a fixed set of top-k dimensions (chosen at layer0 token vector) and measures cosine similarity across layers.",
            "interpretation": "Higher cosine at deeper layers indicates better retention of the initially-strong dimensions through the network.",
        },
    }

    if inj_enabled:
        summary["injection_ablation_same_sample"] = compare_with_vs_without_injection(cfg, args, device)
        summary["notes"]["injection_check"] = (
            "If delta_last_layer_cosine > 0, injection improved retention on this sample (controlled diagnostic)."
        )

    # Console report
    per_layer = retention["per_layer"]
    last = per_layer[-1]
    print("\n=== Neuron Retention Summary ===")
    print("device:", summary["device"])
    print(f"S={summary['sequence']['total_S']} (M={summary['sequence']['visual_tokens_M']} + T={summary['sequence']['text_tokens_T']})")
    print(f"tracked_topk={retention['tracked_topk']} | token_index={summary['sequence']['token_index_tracked']}")
    print(f"last_layer cosine_to_layer0_topk={last['cosine_to_layer0_topk']:.4f} | mean_abs_topk={last['mean_abs_topk']:.4f}")
    if inj_enabled:
        delta = summary["injection_ablation_same_sample"]["delta_last_layer_cosine"]
        print(f"injection ablation delta_last_layer_cosine={delta:+.4f}")
    print("================================\n")

    ensure_dir(args.outdir)
    if str2bool(args.json):
        out_path = os.path.join(args.outdir, "neuron_retention_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("✅ Saved JSON:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
