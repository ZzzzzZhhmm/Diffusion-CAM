# Diffusion-CAM

Contrastive gradient CAM with TAM-style post-processing. The **method** is defined in `method/diffusion_cam/` and does not assume a specific backbone.

## Layout

| Path | Role |
|------|------|
| `method/diffusion_cam/` | Core: rank Gaussian filter, least-squares, COCO/GranDf listing (`NOTICE.txt` cites TAM origins) |
| `examples/toy_example.py` | Minimal script: post-processing only, no VLM |
| `predict.py` | Full pipeline: generation, forward, contrastive Grad-CAM, TAM-style enhancement (requires optional VLM stack) |
| `third_party/llava/` | Vendored LLaVA-NeXT–style code for one reproducible checkpoint path; replaceable |
| `baselines/gradcam.py` | Optional GradCAM baseline |
| `eval/` | Upstream LMM evaluation package (optional) |
| `scripts/` | Training / DeepSpeed configs |

## Optional checkpoint (full pipeline)

For the paper-style demo, download weights from a compatible Hugging Face collection (e.g. LaViDa / LLaVA-NeXT family) and set:

- `LAVIDA_MODEL_PATH` — e.g. `lavida-llada-v1.0-instruct`
- `LAVIDA_VISION_TOWER` — e.g. `siglip-so400m-patch14-384`

Alternatively, clone [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) or a LaViDa release and install or submodule it; keep `import llava` working, or adapt `predict.py` to your model API.

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
