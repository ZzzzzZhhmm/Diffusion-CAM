# Diffusion-CAM

Official implementation of **Diffusion-CAM**, a gradient-based visual explanation framework for **diffusion-based multimodal large language models (dMLLMs)**.

Diffusion-CAM adapts CAM-style visual attribution from autoregressive MLLMs to the **masked-denoising generation paradigm**, enabling spatially grounded explanations for diffusion MLLMs.

---

## Overview

Most existing explanation methods for multimodal large language models are designed for **autoregressive** generation.  

Diffusion MLLMs instead generate responses through **iterative masked denoising** under fixed multimodal conditioning, which changes both where reliable visual evidence appears and how gradients should be traced.

**Diffusion-CAM** is designed for this setting. It extracts attribution from **structurally valid intermediate multimodal states** along the denoising trajectory, and traces gradients from the final response back to image-grounded hidden features.

The full framework includes:

- **Diffusion-CAM base extractor**

- **AKD** — Adaptive Kernel Denoising

- **DACG** — Distribution-Aware Confidence Gating

- **CBA** — Contextual Background Attenuation

- **SICD** — Single-Instance Causal Debiasing

These modules improve localization quality, suppress background noise, and reduce syntactic interference in activation maps.

---

## Quick Start

This repository currently supports **two usage modes**.

### 1. Method-only demo

Runs the post-processing / refinement part only.  

**No model checkpoint is required.**

```Bash

python examples/toy_example.py
```

### 2. Full Diffusion-CAM pipeline

Runs generation, hidden-state hooking, gradient backpropagation, base Diffusion-CAM extraction, and optional refinement modules.

```Bash

python predict.py \
  --selected_images path/to/ids.txt \
  --ablation_mode all_methods
```

**Notes:**

- `--selected_images` requires `COCO_DATASET_PATH`

- The full pipeline requires a compatible LaViDa / LLaDA-style backend

- Outputs are written according to the current script / experiment configuration

---

## Installation

We recommend **Python 3.10+**.

### Core install

```Bash

pip install -e .[train]
```

### Evaluation utilities

```Bash

cd eval
pip install -e .
cd ..
```

---

## Full Pipeline Setup

The current full pipeline is tested with a LaViDa / LLaDA-style backend.

Before running `predict.py`, set:

```Bash

export LAVIDA_MODEL_PATH=/path/to/your/model
export LAVIDA_VISION_TOWER=/path/to/your/vision_tower
```

For COCO-style evaluation, also set:

```Bash

export COCO_DATASET_PATH=/path/to/coco
```

---

## Repository Structure

```Bash

.
├── baselines/              # Baseline methods such as Grad-CAM
├── eval/                   # Evaluation package / metric utilities
├── examples/               # Minimal runnable examples
├── method/diffusion_cam/   # Core Diffusion-CAM implementation
├── scripts/                # Launch scripts / experiment configs
├── vendor/            # External backend code (currently llava-style path)
├── predict.py              # Full pipeline entry
├── pyproject.toml          # Package / dependency configuration
├── LICENSE
└── README.md
```

---

## Key components

- `examples/toy_example.py`Minimal demo for the post-processing modules only.

- `predict.py`Main entry for the full pipeline:

    1. Multimodal generation

    2. Hidden-state hook registration

    3. Gradient backpropagation

    4. Image-span feature slicing

    5. Base Diffusion-CAM construction

    6. Optional refinement with AKD / DACG / CBA / SICD

- `method/diffusion_cam/`Core implementation of Diffusion-CAM and its refinement modules.

- `eval/`Evaluation-related code for quantitative analysis.

- `vendor/llava/`Vendored backend code used by the current full pipeline.

---

## Diffusion-Specific Note

A key difference from autoregressive CAM extraction is that the attribution step is **not hard-coded**.

Diffusion-CAM only extracts attribution from denoising steps whose hidden states still preserve the full image-token span required for spatial grounding. This model-aware feasibility check is central to making CAM work under masked denoising.

---

## License

This repository is released under the license specified in `LICENSE`.
