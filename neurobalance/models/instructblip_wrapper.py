from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    InstructBlipProcessor,
    InstructBlipForConditionalGeneration,
    BitsAndBytesConfig,
)


@dataclass
class InstructBLIPConfig:
    model_name: str = "Salesforce/instructblip-vicuna-7b"
    use_quantization: bool = False
    quantization_bits: int = 8
    max_new_tokens: int = 32
    max_text_len: int = 256


class InstructBLIPWrapper(nn.Module):
    """
    InstructBLIP model wrapper for vision-language tasks.

    InstructBLIP is a vision-language model that takes images and text prompts
    and generates text responses. This wrapper provides:
    - Loading from HuggingFace (Salesforce/instructblip-vicuna-7b)
    - Support for both training (return loss) and inference (generate text)
    - Optional quantization for memory efficiency
    - Batch processing with proper device handling
    """

    def __init__(self, cfg: Optional[InstructBLIPConfig] = None):
        super().__init__()
        self.cfg = cfg or InstructBLIPConfig()

        # Load processor
        self.processor = InstructBlipProcessor.from_pretrained(self.cfg.model_name)

        # Configure quantization if requested
        quantization_config = None
        if self.cfg.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True if self.cfg.quantization_bits == 8 else False,
                load_in_4bit=True if self.cfg.quantization_bits == 4 else False,
                bnb_4bit_compute_dtype=torch.float16,
            )

        # Load model
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.cfg.model_name,
            quantization_config=quantization_config,
            device_map="auto" if self.cfg.use_quantization else None,
        )

        # Get hidden size from the language model
        if hasattr(self.model, "language_model"):
            self.hidden_size = self.model.language_model.config.hidden_size
        else:
            self.hidden_size = self.model.config.text_config.hidden_size

    def _unpack_inputs(
        self,
        maybe_batch: Optional[Dict[str, Any]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
    ) -> tuple[torch.Tensor, List[str], Optional[List[str]]]:
        """
        Allows multiple calling styles:
        - forward(batch_dict)
        - forward(pixel_values=..., prompts=..., texts=...)
        """
        if maybe_batch is not None:
            if not isinstance(maybe_batch, dict):
                raise TypeError("If provided, first positional arg must be a batch dict.")
            if pixel_values is None:
                pixel_values = maybe_batch.get("pixel_values", None)
            if prompts is None:
                prompts = maybe_batch.get("prompts", None)
            if texts is None:
                texts = maybe_batch.get("texts", None)

        if pixel_values is None:
            raise ValueError("pixel_values is required (either in batch['pixel_values'] or as arg).")
        if prompts is None:
            raise ValueError("prompts is required (either in batch['prompts'] or as arg).")

        if not isinstance(prompts, list) or (len(prompts) > 0 and not isinstance(prompts[0], str)):
            raise TypeError("prompts must be List[str].")

        if texts is not None:
            if not isinstance(texts, list) or (len(texts) > 0 and not isinstance(texts[0], str)):
                raise TypeError("texts must be List[str] if provided.")

        return pixel_values, prompts, texts

    def forward(
        self,
        batch: Optional[Dict[str, Any]] = None,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        prompts: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        max_new_tokens: Optional[int] = None,
        return_hidden: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Forward pass for InstructBLIP.

        Training:
          out = model(batch)  OR  model(pixel_values=..., prompts=..., texts=...)
          returns {"loss": ..., "logits": ...}

        Inference:
          texts=None
          returns {"generated_text": [...]}

        Args:
            batch: Optional dict with keys "pixel_values", "prompts", "texts"
            pixel_values: [B, 3, H, W] image tensor
            prompts: List[str] input prompts
            texts: Optional[List[str]] target texts for training
            max_new_tokens: Max tokens to generate (default from config)
            return_hidden: Whether to return hidden states
            **kwargs: Additional arguments passed to model

        Returns:
            Dict with loss/logits (training) or generated_text (inference)
        """
        pixel_values, prompts, texts = self._unpack_inputs(
            maybe_batch=batch,
            pixel_values=pixel_values,
            prompts=prompts,
            texts=texts,
        )

        if max_new_tokens is None:
            max_new_tokens = self.cfg.max_new_tokens

        device = pixel_values.device

        if texts is not None:
            # TRAINING: prepare inputs with target texts
            inputs = self.processor(
                images=pixel_values,
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(device)

            # Add labels for training
            labels = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
            )["input_ids"].to(device)

            # Forward pass with labels
            outputs = self.model(
                pixel_values=inputs["pixel_values"],
                qformer_input_ids=inputs.get("qformer_input_ids"),
                qformer_attention_mask=inputs.get("qformer_attention_mask"),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
                output_hidden_states=return_hidden,
                return_dict=True,
                **kwargs,
            )

            out = {
                "loss": outputs.loss,
                "logits": outputs.logits,
            }

            if return_hidden and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                out["hidden"] = outputs.hidden_states[-1]

            return out

        else:
            # INFERENCE: generate text
            inputs = self.processor(
                images=pixel_values,
                text=prompts,
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values=inputs["pixel_values"],
                    qformer_input_ids=inputs.get("qformer_input_ids"),
                    qformer_attention_mask=inputs.get("qformer_attention_mask"),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    **kwargs,
                )

            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            return {
                "generated_text": generated_texts,
            }
