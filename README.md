# NEUROBALANCE

**Beyond Dense Activation via Adaptive Neuron Gating and Knowledge Injection for Efficient Domain-Specialized Vision-Language Models**

*Austin Olom Ogar — Nile University of Nigeria (2026)*

NEUROBALANCE is a framework for improving domain-specific performance of Multimodal Large Language Models (MLLMs) while simultaneously reducing computational cost and improving interpretability. It works by identifying domain-specific neurons and selectively activating them during inference, rather than using the full dense network.

## Architecture

NEUROBALANCE consists of four modules that wrap around a frozen MLLM backbone (LLaVA-NeXT or InstructBLIP):

**DAPE (Domain-Aware Prototype Extraction)** — Identifies domain-specific neurons by computing per-neuron activation entropy across domain-labelled data. Neurons with low entropy (activation concentrated in a single domain) are designated as domain-specific. Uses forward hooks on FFN layers to record activation frequencies, then selects neurons at the 1st percentile of entropy.

**AAR (Adaptive Attention Re-weighting)** — Counteracts attention collapse in domain-specialized settings by learning per-head scaling factors (beta) that modulate attention logits. This prevents the model from over-attending to generic tokens at the expense of domain-critical visual-textual alignments.

**SNG (Sparse Neuron Gating)** — Restricts feed-forward computation to only the top-k most relevant neurons per token. A learned gating policy selects which neurons to activate, targeting ~15% density for LLaVA-NeXT. Uses straight-through gradient estimation for end-to-end training.

**KIP (Knowledge Injection Pathways)** — Preserves domain-specific activations across layers via protected residual connections. Uses learnable gamma coefficients to control the injection strength, with a separate (lower) learning rate to prevent catastrophic forgetting.

## Key Results

Evaluated on three VQA benchmarks across medical imaging (PMC-VQA), histopathology (PathVQA), and autonomous driving (LingoQA):

- **+16.3 pp** mean accuracy gain over baseline
- **36.4% FLOPs reduction** via sparse gating
- **32.4% latency reduction** in inference
- **44 pp Domain Neuron Retention** improvement
- **28.5% Attention Entropy** reduction (more focused attention)
- **LLC 0.612 → 0.834** (improved layer-wise coherence)

## Repository Structure

```
neurobalance/
├── neurobalance/
│   ├── __init__.py
│   ├── modules/
│   │   ├── adaptive_attention.py   # AAR: attention logit scaling
│   │   ├── sparse_gating.py        # SNG: top-k neuron selection
│   │   ├── knowledge_injection.py  # KIP: protected residual injection
│   │   ├── dape_entropy.py         # DAPE: domain neuron identification
│   │   └── logit_lens.py           # LLC: logit lens coherence metric
│   ├── models/
│   │   ├── neurobalance_model.py   # Main wrapper (switchboard model)
│   │   ├── llava_next_wrapper.py   # LLaVA-NeXT backbone wrapper
│   │   ├── instructblip_wrapper.py # InstructBLIP backbone wrapper
│   │   └── toy_model.py            # Lightweight model for testing
│   ├── data/
│   │   ├── vqa_datasets.py         # Dataset classes
│   │   └── collators.py            # Batch collation
│   ├── metrics/
│   │   ├── vqa_accuracy.py         # Exact match & VQA accuracy
│   │   ├── anls.py                 # ANLS metric
│   │   └── bleu_rouge_optional.py  # Optional text metrics
│   └── utils/
│       ├── config.py               # YAML config loading
│       ├── seed.py                 # Reproducibility utilities
│       └── logging.py              # Logging setup
├── configs/
│   ├── llava_next_base.yaml        # LLaVA-NeXT baseline config
│   ├── instructblip_base.yaml      # InstructBLIP baseline config
│   ├── neurobalance_full.yaml      # Full NEUROBALANCE config
│   ├── neurobalance_partial.yaml   # Partial module config
│   └── ablations/                  # Ablation study configs
│       ├── attention_only.yaml
│       ├── gating_only.yaml
│       └── injection_only.yaml
├── scripts/
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   ├── profile_flops_latency.py    # Efficiency profiling
│   ├── statistical_validation.py   # Statistical testing (t-tests, CIs)
│   ├── preprocess_pmc_vqa.py       # PMC-VQA preprocessing
│   ├── preprocess_pathvqa.py       # PathVQA preprocessing
│   ├── preprocess_lingoqa.py       # LingoQA preprocessing
│   └── interpretability/
│       ├── attention_entropy.py    # Attention entropy analysis
│       ├── attention_heatmaps.py   # Attention visualization
│       ├── dape_neuron_mining.py   # Domain neuron analysis
│       ├── logit_lens.py           # Logit lens visualization
│       └── neuron_retention.py     # Domain neuron retention metric
├── tests/
│   ├── test_attention_scaling.py
│   ├── test_dape.py
│   ├── test_gating.py
│   └── test_injection.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick Start

### Installation

```bash
git clone https://github.com/NEUROBALANCE-Implementation/neurobalance.git
cd neurobalance
pip install -e .
```

### Running on Google Colab (Recommended)

Since NEUROBALANCE requires a GPU with at least 16 GB VRAM, we provide a self-contained Colab notebook that handles the full pipeline:

1. Upload `NEUROBALANCE_Full_Training_Colab.ipynb` to Google Colab
2. Select a GPU runtime (T4 for 4-bit quantized, A100 for fp16)
3. Run all cells sequentially

The notebook auto-detects your GPU and adjusts quantization accordingly:
- **T4 (16 GB)**: 4-bit NF4 quantization, batch size 8
- **A100 (40 GB)**: fp16, batch size 32

### Local Training

```bash
# Train with LLaVA-NeXT backbone
python scripts/train.py --config configs/neurobalance_full.yaml

# Evaluate
python scripts/evaluate.py --config configs/neurobalance_full.yaml --checkpoint results/best_model.pt

# Profile efficiency
python scripts/profile_flops_latency.py --config configs/neurobalance_full.yaml
```

### Running Ablations

```bash
# AAR only
python scripts/train.py --config configs/ablations/attention_only.yaml

# SNG only
python scripts/train.py --config configs/ablations/gating_only.yaml

# KIP only
python scripts/train.py --config configs/ablations/injection_only.yaml
```

## Training Pipeline

The training procedure has three phases:

**Phase 1 — DAPE Analysis (offline, ~2 hours on A100)**
Pass domain-labelled data through the frozen backbone, record per-neuron activation frequencies, compute entropy, and identify the ~1% most domain-specific neurons per layer.

**Phase 2 — Module Training (8 epochs, ~6 hours on A100)**
Train AAR, SNG, and KIP modules with the backbone frozen. Uses AdamW with cosine learning rate schedule (warmup 1000 steps). KIP uses a separate learning rate at 1/100th of the global rate.

**Phase 3 — RL Fine-tuning for SNG (3000 steps)**
Fine-tune the SNG gating policy using reinforcement learning to optimize the accuracy-sparsity tradeoff.

## Datasets

The framework is evaluated on three domain-specific VQA benchmarks:

| Dataset | Domain | Train | Test | Source |
|---------|--------|-------|------|--------|
| PMC-VQA | Medical imaging | ~149K | ~5K | [HuggingFace](https://huggingface.co/datasets/xmcmic/PMC-VQA) |
| PathVQA | Histopathology | ~19.7K | ~6.7K | [HuggingFace](https://huggingface.co/datasets/flaviagiammarino/path-vqa) |
| LingoQA | Autonomous driving | ~9.4K | ~2.4K | [GitHub](https://github.com/wayveai/LingoQA) |

## Hyperparameters (Table I)

| Parameter | Value |
|-----------|-------|
| Learning rate | 5e-5 |
| KIP learning rate | 5e-7 |
| Weight decay | 0.01 |
| Warmup steps | 1000 |
| Max epochs | 8 |
| Batch size | 32 (A100) / 8 (T4) |
| Gradient accumulation | 4 steps |
| SNG sparsity target | 15% |
| SNG RL learning rate | 1e-4 |
| AAR scoring rank | 2 |
| KIP gamma init | 0.3 |
| DAPE entropy percentile | 1st |
| Seeds | 42, 43, 44, 45, 46 |

## Interpretability Tools

```bash
# Generate attention heatmaps
python scripts/interpretability/attention_heatmaps.py --checkpoint results/best_model.pt

# Analyze domain neuron retention
python scripts/interpretability/neuron_retention.py --checkpoint results/best_model.pt

# Logit lens coherence analysis
python scripts/interpretability/logit_lens.py --checkpoint results/best_model.pt

# Attention entropy measurement
python scripts/interpretability/attention_entropy.py --checkpoint results/best_model.pt
```

## Citation

```bibtex
@article{ogar2026neurobalance,
  title={Beyond Dense Activation via Adaptive Neuron Gating and Knowledge Injection
         for Efficient Domain-Specialized Vision-Language Models},
  author={Ogar, Austin Olom},
  journal={IEEE Access},
  year={2026},
  institution={Nile University of Nigeria}
}
```

## License

MIT License. See `pyproject.toml` for details.
