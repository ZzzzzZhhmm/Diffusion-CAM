"""
Minimal example: Diffusion-CAM post-processing only (numpy), no vision-language model.

Run from repo root: python examples/toy_example.py
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METHOD = os.path.join(_REPO, "method")
if _METHOD not in sys.path:
    sys.path.insert(0, _METHOD)

import numpy as np
from diffusion_cam import rank_guassian_filter

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    h, w = 32, 48
    raw = rng.random((h, w), dtype=np.float64)
    smoothed = rank_guassian_filter(raw, kernel_size=3)
    print("input shape", raw.shape, "output shape", smoothed.shape, "finite", np.isfinite(smoothed).all())
