from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


@dataclass
class DAPEConfig:
    entropy_threshold_percentile: float = 1.0
    """Percentile threshold for selecting domain-specific neurons (lower = more specific)"""


class DAPEAnalyzer(nn.Module):
    """
    Domain-Aware Parameter Entropy (DAPE) analyzer.

    Identifies domain-specific neurons using entropy:
    - P(u|d) = M_{u,d} / N_d (activation frequency per domain)
    - H_DAPE(u) = -sum_d P(u|d) * log2(P(u|d))
    - Select neurons with H_DAPE <= percentile threshold as domain-specific

    Works with any transformer model that has `.language_model.model.layers[i].mlp`
    Uses forward hooks to capture FFN activations.
    """

    def __init__(self, cfg: DAPEConfig | None = None):
        super().__init__()
        self.cfg = cfg or DAPEConfig()

        # Activation tracking
        self.activation_counts: Dict[int, Dict[str, torch.Tensor]] = {}
        self.domain_counts: Dict[int, int] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def register_hooks(self, model: nn.Module, num_layers: int) -> None:
        """
        Register forward hooks on all FFN layers to capture activations.

        Args:
            model: The model with .language_model.model.layers structure
            num_layers: Number of transformer layers
        """
        self.hooks.clear()
        self.activation_counts.clear()
        self.domain_counts.clear()

        for layer_idx in range(num_layers):
            try:
                ffn = model.language_model.model.layers[layer_idx].mlp
            except (AttributeError, IndexError):
                continue

            def make_hook(layer_id):
                def hook(module, input, output):
                    # output shape: [B, T, D] after activation
                    if isinstance(output, torch.Tensor):
                        acts = (output > 0).float()  # [B, T, D] binary activation
                        neuron_acts = acts.sum(dim=(0, 1))  # [D] sum over batch and time

                        if layer_id not in self.activation_counts:
                            self.activation_counts[layer_id] = {}

                        # Track activation count for current domain
                        domain_id = self.domain_counts.get(layer_id, 0)
                        if domain_id not in self.activation_counts[layer_id]:
                            self.activation_counts[layer_id][domain_id] = neuron_acts.detach().cpu()
                        else:
                            self.activation_counts[layer_id][domain_id] += neuron_acts.detach().cpu()

                        # Increment sequence count for this domain at this layer
                        if not hasattr(self, '_domain_seq_counts'):
                            self._domain_seq_counts = {}
                        if layer_id not in self._domain_seq_counts:
                            self._domain_seq_counts[layer_id] = {}
                        if domain_id not in self._domain_seq_counts[layer_id]:
                            self._domain_seq_counts[layer_id][domain_id] = 0
                        self._domain_seq_counts[layer_id][domain_id] += acts.shape[0] * acts.shape[1]

                return hook

            handle = ffn.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def set_domain(self, domain_id: int) -> None:
        """Set the current domain for the next forward pass."""
        for layer_id in list(self.activation_counts.keys()) + list(range(32)):
            if layer_id not in self.domain_counts:
                self.domain_counts[layer_id] = domain_id

    def compute_entropy(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DAPE entropy for a specific layer.

        Returns:
            entropy: [D] entropy per neuron
            domain_specific_mask: [D] binary mask of domain-specific neurons (1 if entropy <= threshold)
        """
        if layer_idx not in self.activation_counts:
            raise ValueError(f"No activation data for layer {layer_idx}")

        layer_acts = self.activation_counts[layer_idx]

        # Get all domains that have data
        domains = sorted(layer_acts.keys())
        if not domains:
            raise ValueError(f"No activation data collected for layer {layer_idx}")

        # Compute total activations per domain (N_d)
        domain_totals = {}
        for domain_id in domains:
            acts = layer_acts[domain_id]  # [D]
            # Use sequence counts to get proper domain size
            if hasattr(self, '_domain_seq_counts') and layer_idx in self._domain_seq_counts:
                domain_totals[domain_id] = self._domain_seq_counts[layer_idx].get(domain_id, acts.sum().item())
            else:
                domain_totals[domain_id] = acts.sum().item()

        # Stack activations: [num_domains, D]
        all_acts = torch.stack([layer_acts[d].float() for d in domains], dim=0)
        D = all_acts.shape[1]

        # Compute P(u|d) = M_{u,d} / N_d
        domain_totals_tensor = torch.tensor(
            [domain_totals[d] for d in domains],
            dtype=torch.float32,
            device=all_acts.device
        ).unsqueeze(1)  # [num_domains, 1]

        # Avoid division by zero
        domain_totals_tensor = torch.clamp(domain_totals_tensor, min=1.0)

        P_u_given_d = all_acts / domain_totals_tensor  # [num_domains, D]

        # Compute H_DAPE(u) = -sum_d P(u|d) * log2(P(u|d))
        # Avoid log(0) by clamping
        P_safe = torch.clamp(P_u_given_d, min=1e-10)
        entropy = -(P_u_given_d * torch.log2(P_safe)).sum(dim=0)  # [D]

        # Select neurons with entropy <= percentile threshold
        percentile_value = np.percentile(entropy.detach().cpu().numpy(), self.cfg.entropy_threshold_percentile)
        domain_specific_mask = (entropy <= percentile_value).float()  # [D]

        return entropy, domain_specific_mask

    def get_domain_specific_neurons(self, layer_idx: int) -> torch.Tensor:
        """
        Get the mask of domain-specific neurons for a layer.

        Returns:
            mask: [D] binary mask (1 = domain-specific, 0 = general)
        """
        _, mask = self.compute_entropy(layer_idx)
        return mask
