# Diffusion-CAM

Official implementation of **Diffusion-CAM**, a gradient-based visual explanation framework for **diffusion-based multimodal large language models (dMLLMs)**.

> Diffusion-CAM adapts CAM-style visual attribution to the masked-denoising generation paradigm, enabling spatially grounded explanations for diffusion MLLMs.

---

## Overview

Recent explainability methods for multimodal large language models are largely built on **autoregressive** generation. In contrast, diffusion MLLMs generate responses through **iterative masked denoising** under fixed multimodal conditioning, which changes where and how reliable visual evidence should be extracted.

Diffusion-CAM is designed for this setting. Instead of relying on autoregressive token dependencies, it identifies **structurally valid intermediate multimodal states** along the denoising trajectory and traces gradients from the final response back to image-grounded hidden features.

Our framework includes:

- **Diffusion-CAM base extractor** for dMLLMs
- **AKD**: Adaptive Kernel Denoising
- **DACG**: Distribution-Aware Confidence Gating
- **CBA**: Contextual Background Attenuation
- **SICD**: Single-Instance Causal Debiasing

These modules improve localization quality, reduce background noise, and suppress syntactic interference in activation maps.

---

## Highlights

- First CAM-style visual explanation framework tailored to **diffusion MLLMs**
- Supports attribution under **masked denoising** rather than next-token prediction
- Uses a **model-aware feasibility check** to select valid denoising steps for CAM extraction
- Provides both **quantitative evaluation** and **qualitative visualization**
- Includes sensitivity analysis, efficiency analysis, and controlled validation of the linguistic-economy hypothesis

---

## Method Summary

Given an image and a prompt, the model generates the response through iterative denoising. Diffusion-CAM:

1. Registers hooks on intermediate transformer blocks
2. Identifies the valid image-token span from multimodal packing metadata
3. Backpropagates gradients from selected answer-token scores
4. Builds a base CAM from image-region hidden features and gradients
5. Refines the heatmap with AKD, DACG, CBA, and SICD

A key point is that the attribution step is **not hard-coded**.  
We only extract CAM from denoising steps whose hidden states still contain the full image-token span.

---

## Repository Structure
| Path | Role                                                                                                       |
|------|------------------------------------------------------------------------------------------------------------|
| `method/diffusion_cam/` | Core                                                                                                       |
| `examples/toy_example.py` | Minimal script: post-processing only, no VLM                                                               |
| `predict.py` | Full pipeline: generation, forward, contrastive Grad-CAM, heatmap refinement |
| `third_party/llava/` | Vendored LLaVA-NeXT–style code for one reproducible checkpoint path; replaceable                           |
| `baselines/gradcam.py` | Optional GradCAM baseline                                                                                  |
| `eval/` |                                                              |
| `scripts/` | Training / DeepSpeed configs                                                                               |

## Optional checkpoint (full pipeline)
Download weights from a compatible Hugging Face collection (e.g. LaViDa / LLaVA-NeXT family) and set:

- `LAVIDA_MODEL_PATH` — e.g. `lavida-llada-v1.0-instruct` (use the real folder or HF id for that model)
- `LAVIDA_VISION_TOWER` — e.g. `siglip-so400m-patch14-384`

Alternatively, clone [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) or the LaViDa release you rely on and install or submodule it; keep `import llava` working, or adapt `predict.py` to your model API.

## Install

```bash
pip install -e .[train]
cd eval && pip install -e . && cd ..
```

## Run

Method-only (no GPU model):

```bash
python examples/toy_example.py
```

Full pipeline (needs `third_party/llava` + checkpoints):

```bash
python predict.py --selected_images path/to/ids.txt --ablation_mode all_methods
```

`--selected_images` requires `COCO_DATASET_PATH` to resolve images and masks.

## License

See `LICENSE`. Third-party snippets in `method/diffusion_cam/` are documented in `method/diffusion_cam/NOTICE.txt`. Vendored code under `third_party/` remains subject to its upstream licenses.
