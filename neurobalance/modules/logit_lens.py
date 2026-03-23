from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LogitLensConfig:
    """Configuration for Logit Lens Coherence metric."""
    pass


class LogitLensCoherence(nn.Module):
    """
    Logit Lens Coherence (LLC) metric.

    Measures how coherently information flows through transformer layers
    by comparing intermediate layer predictions with the final layer predictions.

    For each layer l:
    - Project hidden states h_l through the final LM head to get intermediate logits
    - Compute cosine similarity between intermediate and final layer predictions
    - LLC = mean cosine similarity across layers

    This measures the consistency of predictions across layers.
    """

    def __init__(self, cfg: LogitLensConfig | None = None):
        super().__init__()
        self.cfg = cfg or LogitLensConfig()

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Logit Lens Coherence.

        Args:
            hidden_states: [num_layers, B, T, D] or list of [B, T, D] tensors
                Hidden states from all layers of the transformer
            lm_head: The language model head (final projection + softmax)
            attention_mask: [B, T] optional attention mask

        Returns:
            Dictionary with:
                - "llc": float, mean cosine similarity across layers
                - "layer_similarities": [num_layers-1] cosine similarities
                - "final_logits": [B, T, vocab_size] logits from final layer
        """
        # Handle both tensor stack and list of tensors
        if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 4:
            # [num_layers, B, T, D]
            hidden_list = [hidden_states[i] for i in range(hidden_states.shape[0])]
        elif isinstance(hidden_states, (list, tuple)):
            hidden_list = list(hidden_states)
        else:
            raise ValueError(
                f"hidden_states must be [num_layers, B, T, D] tensor or list of tensors, "
                f"got {type(hidden_states)} with shape {hidden_states.shape if isinstance(hidden_states, torch.Tensor) else 'N/A'}"
            )

        if len(hidden_list) < 2:
            raise ValueError("Need at least 2 layers to compute coherence")

        # Get final layer logits
        final_hidden = hidden_list[-1]  # [B, T, D]
        final_logits = lm_head(final_hidden)  # [B, T, vocab_size]

        # Get final layer predictions (take argmax)
        final_preds = torch.argmax(final_logits, dim=-1)  # [B, T]

        # Compute similarities for each intermediate layer
        layer_similarities = []

        for layer_idx in range(len(hidden_list) - 1):
            intermediate_hidden = hidden_list[layer_idx]  # [B, T, D]
            intermediate_logits = lm_head(intermediate_hidden)  # [B, T, vocab_size]

            # Get predictions from this layer
            intermediate_preds = torch.argmax(intermediate_logits, dim=-1)  # [B, T]

            # Compute cosine similarity between prediction distributions
            # Convert logits to probabilities
            final_probs = F.softmax(final_logits, dim=-1)  # [B, T, vocab_size]
            intermediate_probs = F.softmax(intermediate_logits, dim=-1)  # [B, T, vocab_size]

            # Reshape for cosine similarity: [B*T, vocab_size]
            B, T, V = final_probs.shape
            final_flat = final_probs.reshape(-1, V)  # [B*T, vocab_size]
            intermediate_flat = intermediate_probs.reshape(-1, V)  # [B*T, vocab_size]

            # Compute cosine similarity
            similarity = F.cosine_similarity(intermediate_flat, final_flat, dim=-1)  # [B*T]

            # Apply attention mask if provided
            if attention_mask is not None:
                mask = attention_mask.reshape(-1)  # [B*T]
                similarity = similarity * mask
                valid_count = mask.sum().clamp(min=1)
                similarity = similarity.sum() / valid_count
            else:
                similarity = similarity.mean()

            layer_similarities.append(similarity)

        # Compute mean LLC across all layers
        layer_similarities_tensor = torch.stack(layer_similarities)
        llc = layer_similarities_tensor.mean()

        return {
            "llc": llc,
            "layer_similarities": layer_similarities_tensor,
            "final_logits": final_logits,
            "final_preds": final_preds,
        }

    def compute_layer_coherence(
        self,
        hidden_states: List[torch.Tensor],
        lm_head: nn.Module,
    ) -> torch.Tensor:
        """
        Simplified version: just return the LLC score.

        Args:
            hidden_states: List of [B, T, D] tensors from each layer
            lm_head: The language model head

        Returns:
            Scalar LLC score
        """
        result = self.forward(hidden_states, lm_head)
        return result["llc"]
