examples/

  toy_example.py
    Uses only method/diffusion_cam (post-processing helpers). No LLaVA / checkpoint.

  Full paper-style pipeline (contrastive CAM + optional VLM) stays at repo root:
    predict.py
    Requires third_party/llava and checkpoints; see main README.

Optional: install upstream LLaVA-NeXT or LaViDa tooling in a separate clone and point PYTHONPATH,
or use git submodule if you prefer not to vendor third_party/llava.
