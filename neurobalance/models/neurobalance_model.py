# neurobalance/models/neurobalance_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
import torch.nn as nn

from neurobalance.modules.sparse_gating import SparseGating, SparseGatingConfig
from neurobalance.modules.adaptive_attention import AdaptiveAttention, AdaptiveAttentionConfig
from neurobalance.modules.knowledge_injection import KnowledgeInjection, KnowledgeInjectionConfig

from neurobalance.models.llava_next_wrapper import LlavaNextMiniWrapper, LlavaMiniConfig


def _get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _first_present(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


@dataclass
class NeuroBalanceBuildInfo:
    backbone_name: str
    gating_enabled: bool
    attention_enabled: bool
    injection_enabled: bool


class NeuroBalanceModel(nn.Module):
    """
    Switchboard model that wraps a baseline backbone and optionally applies:
      - sparse gating          (on fused embeddings)
      - adaptive attention     (proxy: scale visual token embeddings)
      - knowledge injection    (inject knowledge vector into fused embeddings)

    Forward expects a batch dict and returns:
      { "loss": Tensor, "logits": Tensor, "logs": Dict[str, float] }
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        # ---- Backbone (baseline) ----
        text_model_name = _get(cfg, "model.text_model_name", "distilgpt2")
        self.backbone = LlavaNextMiniWrapper(LlavaMiniConfig(text_model_name=text_model_name))

        # ---- Module toggles ----
        nb = cfg.get("neurobalance", {})

        gating_enabled = bool(_get(nb, "gating.enabled", False))
        attn_enabled = bool(_get(nb, "attention.enabled", False))
        inj_enabled = bool(_get(nb, "injection.enabled", False))

        self.gating: Optional[SparseGating] = None
        self.adaptive_attention: Optional[AdaptiveAttention] = None
        self.injection: Optional[KnowledgeInjection] = None

        if gating_enabled:
            dens = float(_get(nb, "gating.target_density", 0.15))
            self.gating = SparseGating(SparseGatingConfig(target_density=dens, mode="per_token"))

        if attn_enabled:
            beta = float(_get(nb, "attention.beta", 1.0))
            self.adaptive_attention = AdaptiveAttention(
                AdaptiveAttentionConfig(init_beta=beta, learnable=False)
            )

        if inj_enabled:
            gamma = float(_get(nb, "injection.gamma", 0.0))
            self.injection = KnowledgeInjection(
                KnowledgeInjectionConfig(gamma=gamma, learnable_gamma=False)
            )

        self.build_info = NeuroBalanceBuildInfo(
            backbone_name=str(_get(cfg, "model.name", "unknown")),
            gating_enabled=gating_enabled,
            attention_enabled=attn_enabled,
            injection_enabled=inj_enabled,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expected toy batch keys:
          - pixel_values: Tensor [B,3,H,W]   (aliases: image, images)
          - questions:    list[str]         (aliases: question, qs)
          - answers:      list[str] or None (aliases: answer, as)

        Returns:
          dict with "loss" (train), "logits" (train), and "logs" (always)
        """
        # ---------- Parse batch (support aliases) ----------
        pixel_values = _first_present(batch, ["pixel_values", "image", "images"])
        questions = _first_present(batch, ["questions", "question", "qs"])
        answers = _first_present(batch, ["answers", "answer", "as"])

        if pixel_values is None or questions is None:
            raise KeyError(
                "Batch must contain at least 'pixel_values' (or 'image') and 'questions' (or 'question')."
            )

        device = pixel_values.device

        # ---------- Vision tokens ----------
        # [B,M,Dv] -> [B,M,H]
        vis_tokens = self.backbone.vision(pixel_values)
        vis_tokens = self.backbone.vision_to_lm(vis_tokens)

        logs: Dict[str, float] = {
            "gating_enabled": float(self.gating is not None),
            "attention_enabled": float(self.adaptive_attention is not None),
            "injection_enabled": float(self.injection is not None),
        }

        # ---------- Adaptive attention (proxy) ----------
        # Use clamped beta from config limits for stability/reproducibility.
        if self.adaptive_attention is not None:
            beta = torch.clamp(
                self.adaptive_attention.beta,
                self.adaptive_attention.cfg.clamp_min,
                self.adaptive_attention.cfg.clamp_max,
            )
            # proxy scaling; beta=0 would be "no change", but your beta clamp min is 0.1
            # so we use (1 + beta) to keep identity-ish when beta is small.
            vis_tokens = vis_tokens * (1.0 + beta)
            logs["attention_beta"] = float(beta.detach().cpu().item())

        # ---------- Tokenize text ----------
        prompts = self.backbone.build_prompts(questions, answers=answers)
        tok = self.backbone.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.backbone.cfg.max_text_len,
            return_tensors="pt",
        ).to(device)

        input_ids = tok["input_ids"]          # [B,T]
        attn_mask = tok["attention_mask"]     # [B,T]

        # ---------- Text embeddings ----------
        text_embeds = self.backbone.lm.get_input_embeddings()(input_ids)  # [B,T,H]

        # ---------- Fuse embeddings: [visual + text] ----------
        fused_embeds = torch.cat([vis_tokens, text_embeds], dim=1)        # [B,M+T,H]

        # ---------- Sparse gating on fused embeddings ----------
        if self.gating is not None:
            fused_embeds, mask = self.gating(fused_embeds)  # mask is same shape as fused_embeds
            # mask is [B,Seq,H] => density is simply mean over all elements
            logs["gating_density"] = float(mask.float().mean().detach().cpu().item())
            logs["gating_target_density"] = float(self.gating.cfg.target_density)

        # ---------- Knowledge injection on fused embeddings ----------
        if self.injection is not None:
            # Placeholder knowledge vector: mean visual embedding per sample -> [B,H]
            knowledge_vec = vis_tokens.mean(dim=1)
            fused_embeds = self.injection(fused_embeds, knowledge_vec)
            gamma = torch.clamp(
                self.injection.gamma,
                self.injection.cfg.clamp_min,
                self.injection.cfg.clamp_max,
            )
            logs["injection_gamma"] = float(gamma.detach().cpu().item())

        # ---------- Fused attention mask ----------
        B, M, _ = vis_tokens.shape
        vis_mask = torch.ones((B, M), device=device, dtype=attn_mask.dtype)
        fused_mask = torch.cat([vis_mask, attn_mask], dim=1)              # [B,M+T]

        # ---------- Training ----------
        if answers is not None:
            labels = input_ids.clone()
            ignore_prefix = torch.full((B, M), -100, device=device, dtype=labels.dtype)
            fused_labels = torch.cat([ignore_prefix, labels], dim=1)

            out = self.backbone.lm(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                labels=fused_labels,
                use_cache=False,
            )
            return {
                "loss": out.loss,
                "logits": out.logits,
                "logs": logs,
            }

        # ---------- Inference ----------
        gen_ids = self.backbone.lm.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            max_new_tokens=int(_get(self.cfg, "generation.max_new_tokens", 32)),
            do_sample=False,
            num_beams=1,
            pad_token_id=self.backbone.tokenizer.pad_token_id,
            eos_token_id=self.backbone.tokenizer.eos_token_id,
        )
        texts = self.backbone.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return {"generated_text": texts, "logs": logs}
