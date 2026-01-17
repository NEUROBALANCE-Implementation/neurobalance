# neurobalance/models/llava_next_wrapper.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LlavaMiniConfig:
    text_model_name: str = "distilgpt2"  # or "gpt2"
    max_text_len: int = 256
    num_image_tokens: int = 16          # how many visual tokens to produce
    image_token_dim: int = 256          # internal vision token dim


class TinyVisionEncoder(nn.Module):
    """
    Simple CNN -> produces num_image_tokens tokens.
    Output shape: [B, num_image_tokens, image_token_dim]
    """
    def __init__(self, num_image_tokens: int, image_token_dim: int):
        super().__init__()
        self.num_image_tokens = num_image_tokens
        self.image_token_dim = image_token_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),   # 336 -> 168
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # 168 -> 84
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2), # 84 -> 42
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# 42 -> 21
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, num_image_tokens))
        self.proj = nn.Linear(128, image_token_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B,3,H,W]
        returns: [B, M, Dv]
        """
        x = self.backbone(pixel_values)              # [B,128,h,w]
        x = self.pool(x).squeeze(2)                  # [B,128,M]
        x = x.transpose(1, 2).contiguous()           # [B,M,128]
        x = self.proj(x)                              # [B,M,Dv]
        return x


class LlavaNextMiniWrapper(nn.Module):
    """
    Minimal LLaVA-like baseline:
      - Vision encoder -> visual tokens
      - Project visual tokens to LLM hidden size
      - Concatenate [visual_tokens + text_tokens] as input embeddings to a small causal LM
      - Train with LM loss to generate the answer

    IMPORTANT:
      This wrapper supports BOTH calling styles:

      1) Old style:
         out = model(pixel_values=..., questions=[...], answers=[...])

      2) Switchboard style:
         out = model(batch_dict)
         where batch_dict has keys: "pixel_values", "questions", "answers"
    """
    def __init__(self, cfg: Optional[LlavaMiniConfig] = None):
        super().__init__()
        self.cfg = cfg or LlavaMiniConfig()

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # LM
        self.lm = AutoModelForCausalLM.from_pretrained(self.cfg.text_model_name)
        self.hidden_size = self.lm.config.hidden_size

        # Vision
        self.vision = TinyVisionEncoder(
            num_image_tokens=self.cfg.num_image_tokens,
            image_token_dim=self.cfg.image_token_dim,
        )
        self.vision_to_lm = nn.Linear(self.cfg.image_token_dim, self.hidden_size)

    def build_prompts(self, questions: List[str], answers: Optional[List[str]] = None) -> List[str]:
        prompts: List[str] = []
        for i, q in enumerate(questions):
            if answers is None:
                prompts.append(f"Question: {q}\nAnswer:")
            else:
                prompts.append(f"Question: {q}\nAnswer: {answers[i]}")
        return prompts

    def _unpack_inputs(
        self,
        maybe_batch: Optional[Dict[str, Any]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        questions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
    ) -> tuple[torch.Tensor, List[str], Optional[List[str]]]:
        """
        Allows:
          forward(batch_dict)
          forward(pixel_values=..., questions=..., answers=...)
        """
        if maybe_batch is not None:
            if not isinstance(maybe_batch, dict):
                raise TypeError("If provided, first positional arg must be a batch dict.")
            if pixel_values is None:
                pixel_values = maybe_batch.get("pixel_values", None)
            if questions is None:
                questions = maybe_batch.get("questions", None)
            if answers is None:
                answers = maybe_batch.get("answers", None)

        if pixel_values is None:
            raise ValueError("pixel_values is required (either in batch['pixel_values'] or as arg).")
        if questions is None:
            raise ValueError("questions is required (either in batch['questions'] or as arg).")

        if not isinstance(questions, list) or (len(questions) > 0 and not isinstance(questions[0], str)):
            raise TypeError("questions must be List[str].")

        if answers is not None:
            if not isinstance(answers, list) or (len(answers) > 0 and not isinstance(answers[0], str)):
                raise TypeError("answers must be List[str] if provided.")

        return pixel_values, questions, answers

    def forward(
        self,
        batch: Optional[Dict[str, Any]] = None,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        questions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        max_new_tokens: int = 32,
        return_hidden: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Training:
          out = model(batch)  OR  model(pixel_values=..., questions=..., answers=...)
          returns {"loss": ..., "logits": ...}

        Inference:
          answers=None
          returns {"generated_text": [...]}
        """
        pixel_values, questions, answers = self._unpack_inputs(
            maybe_batch=batch,
            pixel_values=pixel_values,
            questions=questions,
            answers=answers,
        )

        device = pixel_values.device

        # 1) Vision tokens -> LM space
        vis_tokens = self.vision(pixel_values)             # [B,M,Dv]
        vis_tokens = self.vision_to_lm(vis_tokens)         # [B,M,H]

        # Simple "knowledge" signal for later modules:
        # mean visual embedding per sample: [B,H]
        knowledge_vec = vis_tokens.mean(dim=1)

        # 2) Text tokens
        prompts = self.build_prompts(questions, answers=answers)
        tok = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_text_len,
            return_tensors="pt",
        ).to(device)

        input_ids = tok["input_ids"]          # [B,T]
        attn_mask = tok["attention_mask"]     # [B,T]

        # 3) Text embeddings
        text_embeds = self.lm.get_input_embeddings()(input_ids)  # [B,T,H]

        # 4) Fuse embeddings
        fused_embeds = torch.cat([vis_tokens, text_embeds], dim=1)  # [B,M+T,H]

        # 5) Fuse attention mask
        B, M, _ = vis_tokens.shape
        vis_mask = torch.ones((B, M), device=device, dtype=attn_mask.dtype)
        fused_mask = torch.cat([vis_mask, attn_mask], dim=1)         # [B,M+T]

        if answers is not None:
            # TRAINING: labels ignore visual prefix
            labels = input_ids.clone()
            ignore_prefix = torch.full((B, M), -100, device=device, dtype=labels.dtype)
            fused_labels = torch.cat([ignore_prefix, labels], dim=1)

            lm_out = self.lm(
                inputs_embeds=fused_embeds,
                attention_mask=fused_mask,
                labels=fused_labels,
                use_cache=False,
                output_hidden_states=bool(return_hidden),
                return_dict=True,
            )

            out: Dict[str, Any] = {
                "loss": lm_out.loss,
                "logits": lm_out.logits,
                "knowledge": knowledge_vec,   # [B,H]
                "vis_tokens": vis_tokens,     # [B,M,H]
            }

            if return_hidden and lm_out.hidden_states is not None:
                out["hidden"] = lm_out.hidden_states[-1]  # [B,M+T,H]

            return out

        # INFERENCE
        gen_ids = self.lm.generate(
            inputs_embeds=fused_embeds,
            attention_mask=fused_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return {
            "generated_text": texts,
            "knowledge": knowledge_vec,
            "vis_tokens": vis_tokens,
        }
