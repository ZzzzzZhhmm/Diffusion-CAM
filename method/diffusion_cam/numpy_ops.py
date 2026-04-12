# SPDX-License-Identifier: Apache-2.0
# Adapted from the TAM (Token Activation Mapping) reference implementation.
# See NOTICE.txt in this directory for attribution.

import numpy as np
from scipy.optimize import minimize_scalar


def rank_guassian_filter(img, kernel_size=3):
    """Rank-based Gaussian-weighted filter for robust activation map denoising."""
    filtered_img = np.zeros_like(img)
    pad_width = kernel_size // 2
    padded_img = np.pad(img, pad_width, mode="reflect")
    ax = np.array(range(kernel_size**2)) - kernel_size**2 // 2

    for i in range(pad_width, img.shape[0] + pad_width):
        for j in range(pad_width, img.shape[1] + pad_width):
            window = padded_img[i - pad_width : i + pad_width + 1, j - pad_width : j + pad_width + 1]

            sorted_window = np.sort(window.flatten())
            mean = sorted_window.mean()
            if mean > 0:
                sigma = sorted_window.std() / mean
                kernel = np.exp(-(ax**2) / (2 * sigma**2))
                kernel = kernel / np.sum(kernel)
                value = (sorted_window * kernel).sum()
            else:
                value = 0
            filtered_img[i - pad_width, j - pad_width] = value

    return filtered_img


def least_squares(map1, map2):
    """Scalar minimizing squared difference between map1 and scalar * map2."""

    def diff(x, map1, map2):
        return np.sum((map1 - map2 * x) ** 2)

    result = minimize_scalar(diff, args=(map1, map2))
    return result.x
