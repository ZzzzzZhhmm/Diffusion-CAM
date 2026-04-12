import os
import sys
import copy
import time
import json
import random
import argparse
import re

os.environ['PYTHONBREAKPOINT'] = '0'

import builtins
original_breakpoint = builtins.breakpoint
def disabled_breakpoint(*args, **kwargs):
    pass
builtins.breakpoint = disabled_breakpoint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import cv2
from PIL import Image
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
from scipy.optimize import minimize_scalar

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib as mpl

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_THIRD_PARTY = os.path.join(_REPO_ROOT, "third_party")
_METHOD = os.path.join(_REPO_ROOT, "method")
for _p in (_METHOD, _THIRD_PARTY, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Optional VLM stack (vendored under third_party/llava); not required for method/diffusion_cam alone.
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.language_model.llada.generate import generate as llada_generate
from llava.model.language_model.llada.log_likelyhood import get_logits as llada_get_logits

from diffusion_cam.numpy_ops import rank_guassian_filter
from diffusion_cam.dataset_prep import prepare_input

try:
    from pycocotools import mask as mask_utils
except Exception:
    pass

model = None
tokenizer = None
image_processor = None
conv = None
full_sequence_hook = None
feature_maps = []
gradients = []

class FullSequenceHook:
    """Captures features and gradients from the same tensor object in a transformer block.

    Ensures identity consistency: features A and gradients dL/dA come from the
    same tensor, avoiding misalignment caused by LayerNorm or hidden_states copies.
    During full forward, captures shape [Batch, Seq_Len, Dim].
    """
    def __init__(self, module):
        self.features = None
        self.gradients = None
        self.fhook = module.register_forward_hook(self.save_features)
        self.bhook = module.register_full_backward_hook(self.save_gradients)
    
    def save_features(self, module, input, output):
        """Save forward output as features."""
        if isinstance(output, tuple):
            self.features = output[0] if isinstance(output[0], torch.Tensor) else output
        else:
            self.features = output
    
    def save_gradients(self, module, grad_input, grad_output):
        """Save backward gradients. grad_output is a tuple; takes the first element."""
        if isinstance(grad_output, tuple):
            if grad_output[0] is not None:
                self.gradients = grad_output[0]
            else:
                for g in grad_output:
                    if g is not None:
                        self.gradients = g
                        break
        else:
            self.gradients = grad_output
    
    def remove(self):
        """Remove hooks."""
        self.fhook.remove()
        self.bhook.remove()
    
    def clear(self):
        """Clear cached features and gradients without removing hooks."""
        self.features = None
        self.gradients = None

def apply_single_image_eci(heatmap, tokens, cam_logits):
    """Single-image ECI: identify and reduce interference from anomalous activation patterns.

    Redefines "interference" for single-image context:
    - Original ECI: interference = same token activations across different images
    - Single-image ECI: interference = anomalous activation patterns within the current image

    Args:
        heatmap: 
        tokens: tokens
        cam_logits: CAM logits

    Returns:
        enhanced_heatmap: ECI
    """
    try:

        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        repeated_tokens = [token for token, count in token_counts.items() if count > 1]

        activation_mean = heatmap.mean()
        activation_std = heatmap.std()
        threshold = activation_mean + 2.0 * activation_std


        interference_map = np.zeros_like(heatmap)

        if repeated_tokens:
            repeat_ratio = len(repeated_tokens) / len(tokens)
            interference_map += repeat_ratio * 0.05

        high_activation_mask = heatmap > threshold
        if high_activation_mask.sum() > 0:
            interference_map[high_activation_mask] += 0.02

        if interference_map.max() > 0:

            def diff(x, main_map, interference_map):
                return np.sum((main_map - interference_map * x)**2)

            result = minimize_scalar(diff, args=(heatmap, interference_map))
            optimal_scalar = result.x

            max_scalar = 0.5
            optimal_scalar = min(optimal_scalar, max_scalar)


            enhanced_heatmap = heatmap - interference_map * optimal_scalar
            enhanced_heatmap = np.maximum(enhanced_heatmap, 0)

            return enhanced_heatmap
        else:
            return heatmap

    except Exception as e:
        return heatmap

def diffusion_aware_kernel_size(step_ratio, activation_std, image_size):
    """Compute adaptive kernel size based on diffusion step ratio and activation statistics."""
    base_kernel = 3

    denoising_steps = int(1 / step_ratio)
    step_factor = min(denoising_steps / 16, 2.0)

    std_factor = min(activation_std * 5, 2.0)

    size_factor = min(image_size / 512, 1.5)

    adaptive_kernel = int(base_kernel * step_factor * std_factor * size_factor)

    if adaptive_kernel % 2 == 0:
        adaptive_kernel += 1

    adaptive_kernel = max(3, min(adaptive_kernel, 11))

    return adaptive_kernel

def apply_diffusion_enhanced_rank_gaussian_filter(heatmap, step_ratio, activation_std):
    """Apply diffusion-enhanced rank Gaussian filter for robust CAM denoising."""
    try:

        kernel_size = diffusion_aware_kernel_size(step_ratio, activation_std, heatmap.shape[0])

        try:
            filtered_heatmap = rank_guassian_filter(heatmap, kernel_size)

        except Exception:
            filtered_heatmap = heatmap.copy()

            blur1 = cv2.GaussianBlur(filtered_heatmap, (kernel_size, kernel_size), 0.5)
            blur2 = cv2.GaussianBlur(filtered_heatmap, (kernel_size+2, kernel_size+2), 1.0)
            blur3 = cv2.GaussianBlur(filtered_heatmap, (kernel_size+4, kernel_size+4), 1.5)

            filtered_heatmap = 0.6 * blur1 + 0.3 * blur2 + 0.1 * blur3


        return filtered_heatmap

    except Exception as e:
        return heatmap

def apply_tam_enhancement(heatmap, image, info4cam, tokens, logits, processor, target_token_idx=0, img_scores_list=None, cam_logits=None, ablation_mode="all_methods", dacg_params=None):
    """Apply TAM enhancement pipeline to a raw LaViDA CAM heatmap.

    Modules applied based on ablation_mode:
    - "baseline": raw CAM, no enhancement
    - "gaussian_only": rank Gaussian filter only (AKD)
    - "confidence_only": adaptive confidence thresholding only (DACG)
    - "background_only": intelligent background suppression only (IBS)
    - "all_methods": full pipeline (AKD + DACG + IBS + SICD)
    """
    try:

        USE_GAUSSIAN_FILTER = ablation_mode in ["gaussian_only", "all_methods"]
        USE_CONFIDENCE_THRESHOLD = ablation_mode in ["confidence_only", "all_methods"]
        USE_BACKGROUND_SUPPRESSION = ablation_mode in ["background_only", "all_methods"]


        if ablation_mode == "baseline":
            return heatmap

        img_scores = heatmap.flatten()

        if cam_logits and len(cam_logits) > 0:
            try:
                if len(cam_logits) > 0 and len(cam_logits[-1]) > 0:
                    last_logits = cam_logits[-1][0]  # [seq_len, vocab_size]

                    attention_scores = torch.softmax(last_logits, dim=-1)

                    if target_token_idx < attention_scores.shape[0]:
                        i = target_token_idx
                        mean_score = attention_scores[i].mean().item()
                        txt_scores = [mean_score]
                    else:
                        text_length = min(10, attention_scores.shape[0])
                        txt_scores = []
                        for i in range(text_length):
                            mean_score = attention_scores[i].mean().item()
                            max_score = attention_scores[i].max().item()
                            entropy = -torch.sum(attention_scores[i] * torch.log(attention_scores[i] + 1e-8)).item()
                            token_score = 0.7 * mean_score + 0.3 * max_score
                            txt_scores.append(token_score)

                    txt_scores = np.array(txt_scores)

                    if txt_scores.max() > txt_scores.min():
                        txt_scores = (txt_scores - txt_scores.min()) / (txt_scores.max() - txt_scores.min())
                    else:
                        raise ValueError("")
                else:
                    raise ValueError("cam_logits")

            except Exception as e:
                img_mean = img_scores.mean()
                img_std = img_scores.std()
                txt_scores = np.random.normal(img_mean * 0.3, img_std * 0.1, 10)
                txt_scores = np.clip(txt_scores, 0, 1)
                if txt_scores.max() == txt_scores.min():
                    txt_scores = np.linspace(0.1, 0.3, 10)
        else:
            img_mean = img_scores.mean()
            img_std = img_scores.std()
            txt_scores = np.random.normal(img_mean * 0.4, img_std * 0.2, 10)
            txt_scores = np.clip(txt_scores, 0, 1)


        original_std = img_scores.std()
        original_mean = img_scores.mean()


        try:
            h, w = heatmap.shape
            img_scores_2d = img_scores.reshape(h, w)


            if USE_GAUSSIAN_FILTER:

                try:

                    stable_input = img_scores_2d.copy()
                    if stable_input.max() > 0:
                        noise_level = stable_input.max() * 1e-6
                        stable_input = stable_input + np.random.normal(0, noise_level, stable_input.shape)

                    step_ratio = 0.5
                    activation_std = original_std

                    filtered_heatmap = apply_diffusion_enhanced_rank_gaussian_filter(
                        stable_input, step_ratio, activation_std
                    )

                    if np.isnan(filtered_heatmap).any() or np.isinf(filtered_heatmap).any():
                        raise ValueError("NaN/Inf detected")

                except Exception as filter_error:
                    raise filter_error
            else:
                filtered_heatmap = img_scores_2d.copy()

            try:
                if np.isnan(filtered_heatmap).any() or np.isinf(filtered_heatmap).any():
                    raise ValueError("Invalid filter result")
            except (ImportError, Exception):
                filtered_heatmap = img_scores_2d.copy()
                step_ratio = 0.5
                activation_std = original_std
                base_kernel = diffusion_aware_kernel_size(step_ratio, activation_std, img_scores_2d.shape[0])
                blur1 = cv2.GaussianBlur(filtered_heatmap, (base_kernel, base_kernel), 0.5)
                blur2 = cv2.GaussianBlur(filtered_heatmap, (base_kernel+2, base_kernel+2), 1.0)
                blur3 = cv2.GaussianBlur(filtered_heatmap, (base_kernel+4, base_kernel+4), 1.5)
                filtered_heatmap = 0.6 * blur1 + 0.3 * blur2 + 0.1 * blur3
                filtered_heatmap = cv2.bilateralFilter(
                    (filtered_heatmap * 255).astype(np.uint8),
                    d=5, sigmaColor=50, sigmaSpace=50
                ).astype(np.float32) / 255.0


            if USE_CONFIDENCE_THRESHOLD:

                mean_val = filtered_heatmap.mean()
                std_val = filtered_heatmap.std()
            else:
                mean_val = filtered_heatmap.mean()
                std_val = filtered_heatmap.std()

            def calculate_adaptive_threshold(heatmap, mean_val, std_val, _dacg_params=None):
                """Compute adaptive threshold based on activation distribution.
                
                _dacg_params (dict, optional): DACG overrides:
                    delta_sigma     : std threshold (default 0.22)
                    delta_mu        : mean threshold (default 0.35)
                    delta_mu_prime  : low-mean threshold (default 0.25)
                    alpha_high_var  :  (90)
                    alpha_high_mean :  (75)
                    alpha_low_mean  :  (80)
                    alpha_default   :  (85)
                """
                if _dacg_params is None:
                    _dacg_params = {}
                delta_sigma    = _dacg_params.get('delta_sigma',    0.22)
                delta_mu       = _dacg_params.get('delta_mu',       0.35)
                delta_mu_prime = _dacg_params.get('delta_mu_prime', 0.25)
                alpha_high_var  = _dacg_params.get('alpha_high_var',  90)
                alpha_high_mean = _dacg_params.get('alpha_high_mean', 75)
                alpha_low_mean  = _dacg_params.get('alpha_low_mean',  80)
                alpha_default   = _dacg_params.get('alpha_default',   85)

                non_zero_values = heatmap[heatmap > 0]

                if len(non_zero_values) == 0:
                    return 1e-6, 0, "all_zero"

                if std_val > delta_sigma:
                    threshold_percentile = alpha_high_var
                    strategy = "high_var"
                elif mean_val > delta_mu:
                    threshold_percentile = alpha_high_mean
                    strategy = "high_mean"
                elif mean_val < delta_mu_prime:
                    threshold_percentile = alpha_low_mean
                    strategy = "low_mean"
                else:
                    threshold_percentile = alpha_default
                    strategy = "default"

                threshold = np.percentile(non_zero_values, threshold_percentile)

                threshold = max(threshold, 1e-6)

                return threshold, threshold_percentile, strategy

            if USE_CONFIDENCE_THRESHOLD:
                confidence_threshold, threshold_percentile, strategy = calculate_adaptive_threshold(
                    filtered_heatmap, mean_val, std_val, _dacg_params=dacg_params
                )

                high_confidence_mask = filtered_heatmap > confidence_threshold


                enhanced_heatmap = filtered_heatmap.copy()

                if not high_confidence_mask.all():
                    low_confidence_mask = ~high_confidence_mask

                    import cv2
                    low_conf_region = enhanced_heatmap[low_confidence_mask]

                    if len(low_conf_region) > 0:
                        low_conf_2d = low_conf_region.reshape(-1, 1)
                        low_conf_filtered = cv2.GaussianBlur(low_conf_2d, (3, 1), 0.3)
                        enhanced_heatmap[low_confidence_mask] = low_conf_filtered.flatten()

            else:
                enhanced_heatmap = filtered_heatmap.copy()

            if USE_BACKGROUND_SUPPRESSION:
                non_zero_enhanced = enhanced_heatmap[enhanced_heatmap > 0]


                if len(non_zero_enhanced) > 0:
                    percentile_threshold = np.percentile(non_zero_enhanced, 50)

                    overall_threshold = np.percentile(enhanced_heatmap, 60)

                    non_zero_mean = non_zero_enhanced.mean()

                    max_threshold = enhanced_heatmap.max() * 0.05

                    background_threshold = (
                        percentile_threshold * 0.15 +
                        overall_threshold * 0.15 +
                        non_zero_mean * 0.5 * 0.4 +
                        max_threshold * 0.3
                    )

                else:
                    background_threshold = 1e-6


                background_mask = enhanced_heatmap < background_threshold

                if background_mask.sum() > 0:
                    background_region = enhanced_heatmap[background_mask]

                    suppression_factor = background_region / safe_threshold if (safe_threshold := max(background_threshold, 1e-8)) else 1.0
                    suppression_factor = np.clip(suppression_factor, 0.0, 1.0)

                    very_low_mask = background_region < background_threshold * 0.2
                    if very_low_mask.sum() > 0:
                        suppression_factor[very_low_mask] = suppression_factor[very_low_mask] * 0.3

                    final_suppression = np.clip(suppression_factor, 0.5, 1.0)

                    enhanced_heatmap[background_mask] = background_region * final_suppression

            original_max = img_scores_2d.max()
            enhanced_heatmap = np.clip(enhanced_heatmap, 0, original_max)


        except Exception as e:
            enhanced_heatmap = heatmap

        img_scores_list.append(enhanced_heatmap.flatten())

        if hasattr(h, 'item'):
            h = h.item()
        elif isinstance(h, np.ndarray):
            h = int(h)
        else:
            h = int(h)

        if hasattr(w, 'item'):
            w = w.item()
        elif isinstance(w, np.ndarray):
            w = int(w)
        else:
            w = int(w)

        if len(enhanced_heatmap.shape) == 1:
            enhanced_heatmap = enhanced_heatmap.reshape(h, w)
        else:
            if enhanced_heatmap.shape != (h, w):
                import cv2
                enhanced_heatmap = cv2.resize(enhanced_heatmap, (w, h))


        if enhanced_heatmap.max() > enhanced_heatmap.min():
            enhanced_heatmap = (enhanced_heatmap - enhanced_heatmap.min()) / (enhanced_heatmap.max() - enhanced_heatmap.min() + 1e-8)

        activation_mean = enhanced_heatmap.mean()
        activation_std = enhanced_heatmap.std()
        activation_skewness = np.mean((enhanced_heatmap - activation_mean) ** 3) / (activation_std ** 3)


        should_apply_eci = True
        # should_apply_eci = (
        # )

        if should_apply_eci:
            try:
                pre_eci_heatmap = enhanced_heatmap.copy()

                eci_enhanced_heatmap = apply_single_image_eci(enhanced_heatmap, tokens, cam_logits)

                pre_eci_range = pre_eci_heatmap.max() - pre_eci_heatmap.min()
                eci_range = eci_enhanced_heatmap.max() - eci_enhanced_heatmap.min()

                if eci_range < pre_eci_range * 0.9:
                    enhanced_heatmap = pre_eci_heatmap
                else:
                    enhanced_heatmap = eci_enhanced_heatmap

            except Exception as e:
                pass
        else:
            pass


        if np.isnan(enhanced_heatmap).any() or np.isinf(enhanced_heatmap).any():
            enhanced_heatmap = heatmap

        return enhanced_heatmap

    except Exception:
        return heatmap

def save_cam_images(original_heatmap, enhanced_heatmap, image, image_name, output_dir="cam_results_all_tokens"):
    """Save CAM comparison images (original vs enhanced).

    Args:
        original_heatmap: raw LaViDA CAM heatmap
        enhanced_heatmap: TAM-enhanced heatmap
        image: 
        image_name: 
        output_dir: 
    """
    import cv2
    import numpy as np
    from PIL import Image

    os.makedirs(output_dir, exist_ok=True)

    try:
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        img_h, img_w = image_array.shape[:2]

        original_resized = cv2.resize(original_heatmap, (img_w, img_h))
        original_colored = cv2.applyColorMap(np.uint8(255 * original_resized), cv2.COLORMAP_JET)
        original_superimposed = cv2.addWeighted(image_array, 0.6, original_colored, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_original_cam.jpg"), original_superimposed)

        enhanced_resized = cv2.resize(enhanced_heatmap, (img_w, img_h))
        enhanced_colored = cv2.applyColorMap(np.uint8(255 * enhanced_resized), cv2.COLORMAP_JET)
        enhanced_superimposed = cv2.addWeighted(image_array, 0.6, enhanced_colored, 0.4, 0)
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_enhanced_cam.jpg"), enhanced_superimposed)

        comparison = np.hstack([original_superimposed, enhanced_superimposed])
        cv2.imwrite(os.path.join(output_dir, f"{image_name}_comparison.jpg"), comparison)

        original_img = image_array.copy()
        original_heatmap_only = cv2.applyColorMap(np.uint8(255 * original_resized), cv2.COLORMAP_JET)
        enhanced_heatmap_only = cv2.applyColorMap(np.uint8(255 * enhanced_resized), cv2.COLORMAP_JET)

        top_row = np.hstack([original_img, original_heatmap_only])
        bottom_row = np.hstack([original_superimposed, enhanced_superimposed])
        summary = np.vstack([top_row, bottom_row])

        cv2.imwrite(os.path.join(output_dir, f"{image_name}_summary.jpg"), summary)


    except Exception:
        pass


def decode_grandf_rle_masks(annotation, image_name):
    """Decode RLE masks from GranD-f annotations.

    Args:
        annotation: GranD-f annotation dict
        image_name: image filename

    Returns:
        combined_mask: 
        object_info: 
    """
    try:
        if 'groundings' not in annotation:
            return None, []

        groundings = annotation['groundings']
        combined_mask = None
        object_info = []


        for obj_idx, (obj_name, obj_data) in enumerate(groundings.items()):
            if 'rle_masks' not in obj_data:
                continue

            rle_masks = obj_data['rle_masks']
            if not rle_masks:
                continue

            if 'size' in rle_masks[0]:
                img_size = rle_masks[0]['size']  # [height, width]
            else:
                continue

            rle_data = rle_masks[0]

            rle = {
                'size': img_size,
                'counts': rle_data['counts']
            }

            try:
                decoded_mask = mask_utils.decode(rle)

                if decoded_mask is not None and decoded_mask.size > 0:
                    object_id = obj_idx + 1
                    decoded_mask = decoded_mask.astype(np.uint8) * object_id

                    if combined_mask is None:
                        combined_mask = decoded_mask
                    else:
                        combined_mask = np.maximum(combined_mask, decoded_mask)

                    object_info.append({
                        'name': obj_name,
                        'id': object_id,
                        'mask_size': img_size
                    })

                else:
                    pass

            except Exception as e:
                continue

        if combined_mask is not None:
            return combined_mask, object_info
        else:
            return None, []

    except Exception as e:
        return None, []

def save_grandf_mask_as_png(combined_mask, image_name, output_dir="temp_grandf_masks"):
    """Save combined segmentation mask as PNG.

    Args:
        combined_mask: binary mask array
        image_name: image filename
        output_dir: 

    Returns:
        mask_path: 
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        mask_filename = f"{Path(image_name).stem}_mask.png"
        mask_path = os.path.join(output_dir, mask_filename)

        cv2.imwrite(mask_path, combined_mask)

        return mask_path

    except Exception as e:
        return None

def load_grandf_ha_data(grandf_path):
    """Load GranD-f HA annotation data.

    Args:
        grandf_path: path to GranD-f dataset

    Returns:
        annotations: 
        images_dir: 
    """
    try:
        annotation_file = os.path.join(grandf_path, "train", "GranDf_HA_GCG_train.json")

        if not os.path.exists(annotation_file):
            return [], None


        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)


        images_dir = os.path.join(grandf_path, "GranDf_HA_images", "train")

        if not os.path.exists(images_dir):
            return annotations, None


        return annotations, images_dir

    except Exception as e:
        return [], None

def compute_iou_direct(heatmap, mask_path):
    """Compute IoU between heatmap and ground truth mask.

    Args:
        heatmap: activation heatmap (numpy array)
        mask_path: path to ground truth mask

    Returns:
        iou: IOU
    """
    try:
        if not os.path.exists(mask_path):
            return 0.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0.0


        heatmap_resized = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))

        heatmap_binary = (heatmap_resized > 0).astype(np.uint8) * 255

        if mask.sum() != 0:
            unique_classes = np.unique(mask)
            unique_classes = unique_classes[unique_classes > 0]


            if len(unique_classes) > 0:
                ious = []
                precisions = []
                recalls = []

                for class_id in unique_classes:
                    gt = (mask == class_id).astype('uint8')

                    tp = float((gt * heatmap_binary > 0).sum())
                    union = ((gt + heatmap_binary / 255) > 0).sum()

                    if union > 0:
                        class_iou = tp / union
                        class_precision = tp / (heatmap_binary > 0).sum() if (heatmap_binary > 0).sum() > 0 else 0
                        class_recall = tp / gt.sum() if gt.sum() > 0 else 0

                        ious.append(class_iou)
                        precisions.append(class_precision)
                        recalls.append(class_recall)

                if len(ious) > 0:
                    max_iou = max(ious)
                    max_precision = precisions[ious.index(max_iou)]
                    max_recall = recalls[ious.index(max_iou)]


                    return max_iou
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return 0.0

    except Exception as e:
        return 0.0

def compute_iou_with_fixed_threshold(heatmap, mask_path, threshold=0.4):
    """Compute IoU using a fixed threshold for binarization.

    Args:
        heatmap: activation heatmap (numpy array)
        mask_path: path to ground truth mask
        threshold: 0.4predict_copy.py

    Returns:
        iou: IOU
    """
    try:
        if not os.path.exists(mask_path):
            return 0.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 0.0


        heatmap_resized = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))

        heatmap_binary = (heatmap_resized > threshold).astype(np.uint8) * 255

        if mask.sum() != 0:
            unique_classes = np.unique(mask)
            unique_classes = unique_classes[unique_classes > 0]


            if len(unique_classes) > 0:
                ious = []
                precisions = []
                recalls = []

                for class_id in unique_classes:
                    gt = (mask == class_id).astype('uint8')

                    tp = float((gt * heatmap_binary > 0).sum())
                    union = ((gt + heatmap_binary / 255) > 0).sum()

                    if union > 0:
                        class_iou = tp / union
                        class_precision = tp / (heatmap_binary > 0).sum() if (heatmap_binary > 0).sum() > 0 else 0
                        class_recall = tp / gt.sum() if gt.sum() > 0 else 0

                        ious.append(class_iou)
                        precisions.append(class_precision)
                        recalls.append(class_recall)

                if len(ious) > 0:
                    max_iou = max(ious)
                    max_precision = precisions[ious.index(max_iou)]
                    max_recall = recalls[ious.index(max_iou)]


                    return max_iou
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return 0.0

    except Exception as e:
        return 0.0

def compute_obj_func_iou(heatmap, mask_path, tokens=None, processor=None):
    """Compute Obj-IoU and Func-IoU following TAM evaluation protocol.

    Args:
        heatmap: activation heatmap (numpy array)
        mask_path: path to ground truth mask
        tokens: token (Func-IOU)
        processor:  (Func-IOU)

    Returns:
        dict: obj_iou, func_iou, precision, recall
    """
    try:
        if not os.path.exists(mask_path):
            return {"obj_iou": 0.0, "func_iou": 0.0, "precision": 0.0, "recall": 0.0}

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {"obj_iou": 0.0, "func_iou": 0.0, "precision": 0.0, "recall": 0.0}


        heatmap_resized = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))

        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)

        otsu_threshold, heatmap_binary = cv2.threshold(
            heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        obj_iou = 0.0
        precision = 0.0
        recall = 0.0
        noun_fg_thresh = []

        if mask.sum() != 0:
            unique_classes = np.unique(mask)
            unique_classes = unique_classes[unique_classes > 0]


            if len(unique_classes) > 0:
                ious = []
                precisions = []
                recalls = []

                for class_id in unique_classes:
                    gt = (mask == class_id).astype('uint8')

                    tp = float((gt * heatmap_binary > 0).sum())
                    union = ((gt + heatmap_binary / 255) > 0).sum()

                    if union > 0:
                        class_iou = tp / union
                        class_precision = tp / (heatmap_binary > 0).sum() if (heatmap_binary > 0).sum() > 0 else 0
                        class_recall = tp / gt.sum() if gt.sum() > 0 else 0

                        ious.append(class_iou)
                        precisions.append(class_precision)
                        recalls.append(class_recall)

                if len(ious) > 0:
                    max_iou = max(ious)
                    max_precision = precisions[ious.index(max_iou)]
                    max_recall = recalls[ious.index(max_iou)]

                    obj_iou = max_iou
                    precision = max_precision
                    recall = max_recall
                    noun_fg_thresh = [otsu_threshold]

                else:
                    pass
            else:
                pass
        else:
            pass

        func_iou = 0.0
        if len(noun_fg_thresh) > 0 and tokens is not None and processor is not None:

            fg_thresh = sum(noun_fg_thresh) / len(noun_fg_thresh)

            neg_iou = float((heatmap_resized < fg_thresh/255.0).sum()) / heatmap_resized.size
            func_iou = neg_iou

        else:
            pass

        return {
            "obj_iou": obj_iou,
            "func_iou": func_iou,
            "precision": precision,
            "recall": recall
        }

    except Exception as e:
        return {"obj_iou": 0.0, "func_iou": 0.0, "precision": 0.0, "recall": 0.0}

def compute_comprehensive_score(obj_iou, precision, recall, heatmap, ground_truth):
    """Compute comprehensive TAM score combining multiple metrics.

    Components:
    1. Obj-IoU: overlap between binarized heatmap and ground truth
    2. Contrast: foreground/background activation ratio

    Func-IOU

    Args:
        obj_iou: IOU
        precision: 
        recall: 
        heatmap:  (numpy array, [0,1])
        ground_truth:  (numpy array)

    Returns:
        dict: 
    """


    foreground_mask = (ground_truth > 0)
    background_mask = (ground_truth == 0)

    fg_mean = heatmap[foreground_mask].mean() if foreground_mask.sum() > 0 else 1.0
    bg_mean = heatmap[background_mask].mean() if background_mask.sum() > 0 else 0.0

    contrast_ratio = fg_mean / (bg_mean + 1e-8)
    contrast_score = min(contrast_ratio / 20.0, 1.0)

    target_activation = heatmap[foreground_mask].sum() if foreground_mask.sum() > 0 else 0
    total_activation = heatmap.sum()
    concentration_ratio = target_activation / (total_activation + 1e-8)

    if precision > 0 and recall > 0:
        detection_f1 = 2 * precision * recall / (precision + recall)
    else:
        detection_f1 = 0

    if obj_iou > 0 and contrast_score > 0 and concentration_ratio > 0:
        tam_score_geometric = (obj_iou * contrast_score * concentration_ratio) ** (1/3)
    else:
        tam_score_geometric = 0.0

    if obj_iou > 0 and contrast_score > 0 and concentration_ratio > 0:
        tam_score_harmonic = 3 / (1/obj_iou + 1/contrast_score + 1/concentration_ratio)
    else:
        tam_score_harmonic = 0.0

    tam_score = tam_score_harmonic

    return {
        "obj_iou": obj_iou,
        "contrast_ratio": contrast_ratio,
        "contrast_score": contrast_score,
        "concentration_ratio": concentration_ratio,

        "fg_mean": fg_mean,
        "bg_mean": bg_mean,
        "detection_f1": detection_f1,            # F1-Score
        "precision": precision,
        "recall": recall,

        "tam_score": tam_score,
        "tam_score_geometric": tam_score_geometric,
        "tam_score_harmonic": tam_score_harmonic
    }

def compute_iou(heatmap, mask_path):
    """Compute Obj-IoU (convenience wrapper)."""
    result = compute_obj_func_iou(heatmap, mask_path)
    return result["obj_iou"]

def validate_and_standardize_image(image_path):
    """Validate and standardize input image (ensure RGB mode).

    Args:
        image_path: path to input image

    Returns:
        image: PIL
    """
    try:
        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        max_size = 1024
        min_size = 224

        width, height = image.size

        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        elif min(width, height) < min_size:
            ratio = min_size / min(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return image

    except Exception as e:
        return None

def find_target_token(tokens, target_word):
    """Find indices of a target word in the token sequence.

    Args:
        tokens: list of token strings
        target_word: word to search for (e.g. "cow", "horse")

    Returns:
        target_indices: tokenNone
    """
    if not target_word:
        return None

    target_word = target_word.lower().strip()
    target_indices = []

    for i, token in enumerate(tokens):
        if not token or not token.strip():
            continue

        clean_token = token.strip().lower()
        clean_token = clean_token.replace(',', '').replace('.', '').replace('!', '').replace('?', '')

        if not clean_token:
            continue

        if len(clean_token) > 0 and len(target_word) > 0:
            if (clean_token == target_word or
                target_word in clean_token):
                target_indices.append(i)

    return target_indices if target_indices else None

def process_single_image(image_path, prompt_text=None, mask_path=None, target_token=None, manual_indices=None, ablation_mode="all_methods"):
    """Generate CAM for a single image using contrastive gradient and TAM enhancement.

    Args:
        image_path: path to input image
        prompt_text: text prompt (optional)
        mask_path:  (IOU)
        target_token: token"cow", "horse" (token)
        ablation_mode:  ("all_methods")

    Returns:
        result: 
    """

    global feature_maps, gradients
    feature_maps.clear()
    gradients.clear()

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    image = validate_and_standardize_image(image_path)
    if image is None:
        return None

    try:
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)
    except Exception:
        return None

    if prompt_text is None:
        prompt_text = "What do you see in this image?"


    conv = copy.deepcopy(conv_templates["llada"])
    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    conv.tokenizer = tokenizer
    prompt_text = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    image_sizes = [image.size]

    t0 = time.time()
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:

            torch.cuda.empty_cache()
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            (cont,hist,cam_logits), info4cam = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.1,
                max_new_tokens=64,
                block_length=64,
                step_ratio=0.5,
                tokenizer=tokenizer,
                prefix_lm=True,
                verbose=True,
                schedule='shift',
            )
            t1 = time.time()
            break

        except Exception as e:
            retry_count += 1

            if retry_count < max_retries:
                time.sleep(2)

                if retry_count == 2:
                    try:
                        scale_factor = 4
                        smaller_image = image.resize((image.size[0]//scale_factor, image.size[1]//scale_factor), Image.Resampling.LANCZOS)
                        image_tensor = process_images([smaller_image], image_processor, model.config)
                        image_tensor = image_tensor.to(model.device, dtype=torch.bfloat16)
                        image_sizes = [smaller_image.size]

                        torch.cuda.empty_cache()
                        gc.collect()
                    except Exception as resize_e:
                        pass
            else:
                return None

    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    sep_words = [tokenizer.batch_decode(i, skip_special_tokens=True) for i in cont][0]

    workspace_dir = "workspace"
    os.makedirs(workspace_dir, exist_ok=True)
    info_save_path = f"{workspace_dir}/{Path(image_path).stem}_decoding-info.txt"
    with open(info_save_path, 'w', encoding='utf-8') as f:
        for line in sep_words:
            f.write(str(line).strip() + '\n')

    text_outputs = [text_output.lstrip('!') for text_output in text_outputs]

    full_text = text_outputs[0] if text_outputs else ""
    full_prompt_for_forward = prompt_text + full_text
    
    
    input_ids_full = tokenizer_image_token(
        full_prompt_for_forward, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors="pt"
    ).unsqueeze(0).to(model.device)
    
    
    model.zero_grad()
    feature_maps.clear()
    gradients.clear()
    torch.cuda.empty_cache()
    
    
    attention_mask = torch.ones_like(input_ids_full, device=model.device)
    labels = torch.full_like(input_ids_full, -100)
    
    try:
        with torch.set_grad_enabled(True):
            outputs = model(
                input_ids=input_ids_full,
                attention_mask=attention_mask,
                labels=labels,
                images=image_tensor,
                image_sizes=image_sizes,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=True
            )
        
        full_logits = outputs.logits  # Shape: [Batch, Sequence_Length, Vocab_Size]
        
        if full_sequence_hook.features is not None:
            hook_features = full_sequence_hook.features
            
            if hook_features.dim() == 3:  # [Batch, Seq_Len, Dim]
                full_feature = hook_features[0]
            else:  # [Seq_Len, Dim]
                full_feature = hook_features
            
        else:
            return None
        
    except Exception:
        return None

    gen_target_idx = 1 if len(sep_words) > 1 else 0
    gen_distractor_idx = 0 if gen_target_idx != 0 else 2
    
    for i, token in enumerate(sep_words):
        if i != gen_target_idx and token and token.strip():
            gen_distractor_idx = i
            break
    
    
    try:
        target_token_text = sep_words[gen_target_idx].strip()
        distractor_token_text = sep_words[gen_distractor_idx].strip()
        
        target_token_ids = tokenizer.encode(target_token_text, add_special_tokens=False)
        distractor_token_ids = tokenizer.encode(distractor_token_text, add_special_tokens=False)
        
        
        target_seq_idx = None
        distractor_seq_idx = None
        
        for tid in target_token_ids:
            matches = (input_ids_full[0] == tid).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                target_seq_idx = matches[-1].item()
                break
        
        for tid in distractor_token_ids:
            matches = (input_ids_full[0] == tid).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                distractor_seq_idx = matches[-1].item()
                break
        
        if target_seq_idx is None:
            target_seq_idx = min(input_ids_full.shape[1] - 1, 100 + gen_target_idx)
        
        if distractor_seq_idx is None:
            distractor_seq_idx = min(input_ids_full.shape[1] - 1, 100 + gen_distractor_idx)
        
        
    except Exception as e:
        target_seq_idx = min(input_ids_full.shape[1] - 1, 101)
        distractor_seq_idx = min(input_ids_full.shape[1] - 1, 100)

    select_logits_idx = np.array([gen_target_idx])


    target_idx = gen_target_idx
    distractor_idx = gen_distractor_idx

    try:
        model.zero_grad()
        full_sequence_hook.gradients = None

        combined_loss = (full_logits[0, target_seq_idx, :].sum()
                         - full_logits[0, distractor_seq_idx, :].sum())
        combined_loss.backward()

        del outputs, full_logits, combined_loss
        torch.cuda.empty_cache()
        gc.collect()

        if full_sequence_hook.gradients is not None:
            hook_grad = full_sequence_hook.gradients
            if hook_grad.dim() == 3:  # [Batch, Seq_Len, Dim]
                combined_grad = hook_grad[0].clone()
            else:
                combined_grad = hook_grad.clone()
            full_sequence_hook.gradients = None
        else:
            return None

        final_contrastive_grad = torch.clamp(combined_grad, min=0)
        del combined_grad

    except RuntimeError as e:
        if "out of memory" in str(e):
            feature_maps.clear()
            gradients.clear()
            if full_sequence_hook is not None:
                full_sequence_hook.clear()
            torch.cuda.empty_cache()
            gc.collect()
            return None
        else:
            return None


    img_feature = None
    grad = None
    bound = [0]
    padf_h, padf_w = (24, 24)
    basefeat_shape = (576, 4096)

    try:
        bound_info, (padfeat_hws, basefeat_shape_info) = info4cam

        bound = bound_info
        padf_h, padf_w = padfeat_hws
        basefeat_shape = basefeat_shape_info
        basefeat_h = np.sqrt(basefeat_shape[0]).astype(np.int8)

    except Exception as e:
        pass


    img_start_idx = bound[0] + basefeat_shape[0]
    img_end_idx = img_start_idx + padf_h * padf_w
    required_len = img_end_idx
    

    if full_feature.shape[0] < img_end_idx:
        img_end_idx = min(img_end_idx, full_feature.shape[0])
        img_start_idx = max(0, img_end_idx - padf_h * padf_w)
    
    if final_contrastive_grad.shape[0] < img_end_idx:
        img_end_idx = min(img_end_idx, final_contrastive_grad.shape[0])
        img_start_idx = max(0, img_end_idx - padf_h * padf_w)

    img_feature = full_feature[img_start_idx:img_end_idx, :]
    
    actual_patches = img_feature.shape[0]
    expected_patches = padf_h * padf_w
    if actual_patches < expected_patches:
        import math
        new_h = int(math.sqrt(actual_patches))
        new_w = actual_patches // new_h
        if new_h * new_w == actual_patches:
            padf_h, padf_w = new_h, new_w
        else:
            return None

    if img_feature.shape[0] != (img_end_idx - img_start_idx):
        actual_img_end = img_start_idx + img_feature.shape[0]
        grad = final_contrastive_grad[img_start_idx:actual_img_end, :]
    else:
        grad = final_contrastive_grad[img_start_idx:img_end_idx, :]
    
    
    if img_feature.shape[0] != grad.shape[0]:
        return None

    try:
        img_feature = rearrange(img_feature, '(h w) d -> d h w ', w=padf_w, h=padf_h)
        grad = rearrange(grad, '(h w) d -> h w d ', w=padf_w, h=padf_h)
    except Exception as e:
        return None

    pooled_gradients = torch.mean(grad, dim=[0, 1])
    activation = img_feature
    for i in range(activation.size(0)):
        activation[i, :, :] *= pooled_gradients[i]
    activation = nn.ReLU()(activation)
    heatmap = torch.mean(activation, dim=0, dtype=torch.float32).detach().cpu().numpy()
    heatmap -= heatmap.min()
    heatmap /= (heatmap.max() + 1e-8)

    heatmap_raw = heatmap.copy()

    if ablation_mode == "baseline":
        threshold = 0.4
        heatmap[heatmap < threshold] = 0
    else:
        pass

    image_h, image_w = np.array(image).shape[:2]
    if hasattr(image_h, 'item'):
        image_h = image_h.item()
    elif isinstance(image_h, np.ndarray):
        image_h = int(image_h)
    else:
        image_h = int(image_h)

    if hasattr(image_w, 'item'):
        image_w = image_w.item()
    elif isinstance(image_w, np.ndarray):
        image_w = int(image_w)
    else:
        image_w = int(image_w)

    heatmap = cv2.resize(heatmap, (image_w, image_h), interpolation=cv2.INTER_LINEAR)

    class SimpleProcessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
            return self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)

    processor = SimpleProcessor(tokenizer)

    tokens = cont[0].tolist()

    enhanced_heatmap = apply_tam_enhancement(
        heatmap=heatmap_raw if ablation_mode != "baseline" else heatmap,
        image=np.array(image),
        info4cam=info4cam,
        tokens=tokens,
        logits=cam_logits,
        processor=processor,
        target_token_idx=select_logits_idx[0] if len(select_logits_idx) > 0 else 2,
        img_scores_list=[],
        cam_logits=cam_logits,
        ablation_mode=ablation_mode
    )

    save_dir = f"cam_results_{ablation_mode}"
    os.makedirs(save_dir, exist_ok=True)

    token_info = f"_token{select_logits_idx[0] if len(select_logits_idx) > 0 else 'unknown'}"
    image_name = Path(image_path).stem + token_info
    save_cam_images(
        original_heatmap=heatmap,
        enhanced_heatmap=enhanced_heatmap,
        image=image,
        image_name=image_name,
        output_dir=save_dir
    )

    original_obj_iou = 0.0
    enhanced_obj_iou = 0.0
    original_func_iou = 0.0
    enhanced_func_iou = 0.0
    original_precision = 0.0
    enhanced_precision = 0.0
    original_recall = 0.0
    enhanced_recall = 0.0

    if mask_path and os.path.exists(mask_path):

        original_result = compute_obj_func_iou(heatmap_raw, mask_path, tokens, processor)
        original_obj_iou = original_result["obj_iou"]
        original_func_iou = original_result["func_iou"]
        original_precision = original_result["precision"]
        original_recall = original_result["recall"]

        enhanced_result = compute_obj_func_iou(enhanced_heatmap, mask_path, tokens, processor)
        enhanced_obj_iou = enhanced_result["obj_iou"]
        enhanced_func_iou = enhanced_result["func_iou"]
        enhanced_precision = enhanced_result["precision"]
        enhanced_recall = enhanced_result["recall"]

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros_like(heatmap_raw)

        heatmap_raw_resized = cv2.resize(heatmap_raw, (mask.shape[1], mask.shape[0]))
        enhanced_heatmap_resized = cv2.resize(enhanced_heatmap, (mask.shape[1], mask.shape[0]))

        original_scores = compute_comprehensive_score(
            original_obj_iou, original_precision, original_recall,
            heatmap_raw_resized, mask
        )
        enhanced_scores = compute_comprehensive_score(
            enhanced_obj_iou, enhanced_precision, enhanced_recall,
            enhanced_heatmap_resized, mask
        )
    else:
        if mask_path:
            pass
        else:
            pass

    summary_path = os.path.join(save_dir, f'{Path(image_path).stem}_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"LaViDA Contrastive Gradient CAM \n")
        f.write(f"=" * 50 + "\n")
        f.write(f": {Path(image_path).name}\n")
        f.write(f"CAM: Contrastive Gradient CAM\n")
        f.write(f"Token (Target): '{sep_words[target_idx]}' ({target_idx})\n")
        f.write(f"Token (Distractor): '{sep_words[distractor_idx]}' ({distractor_idx})\n")
        f.write(f": G_final = ReLU(G_target - G_distractor)\n")
        f.write(f": {t1 - t0:.2f}\n")
        f.write(f": {text_outputs[0] if text_outputs else 'N/A'}\n")
        f.write(f"\nDiffusion-CAM ():\n")
        f.write(f"  LaViDA CAM:\n")
        f.write(f"    Obj-IOU: {original_obj_iou:.4f}\n")
        f.write(f"    Precision: {original_precision:.4f}\n")
        f.write(f"    Recall: {original_recall:.4f}\n")
        f.write(f"    /: {original_scores['contrast_ratio']:.2f}x (fg={original_scores['fg_mean']:.3f}, bg={original_scores['bg_mean']:.3f})\n")
        f.write(f"    : {original_scores['concentration_ratio']:.2%}\n")
        f.write(f"    F1-Score: {original_scores['detection_f1']:.4f}\n")
        f.write(f"    F3-Score (): {original_scores['tam_score_harmonic']:.4f}\n")
        f.write(f"    F3-Score (): {original_scores['tam_score_geometric']:.4f}\n")
        f.write(f"  Diffusion-CAM:\n")
        f.write(f"    Obj-IOU: {enhanced_obj_iou:.4f}\n")
        f.write(f"    Precision: {enhanced_precision:.4f}\n")
        f.write(f"    Recall: {enhanced_recall:.4f}\n")
        f.write(f"    /: {enhanced_scores['contrast_ratio']:.2f}x (fg={enhanced_scores['fg_mean']:.3f}, bg={enhanced_scores['bg_mean']:.3f})\n")
        f.write(f"    : {enhanced_scores['concentration_ratio']:.2%}\n")
        f.write(f"    F1-Score: {enhanced_scores['detection_f1']:.4f}\n")
        f.write(f"    F3-Score (): {enhanced_scores['tam_score_harmonic']:.4f}\n")
        f.write(f"    F3-Score (): {enhanced_scores['tam_score_geometric']:.4f}\n")
        f.write(f"  :\n")
        f.write(f"    Obj-IOU: {enhanced_obj_iou - original_obj_iou:+.4f} ({((enhanced_obj_iou - original_obj_iou) / original_obj_iou * 100) if original_obj_iou > 0 else 0:+.1f}%)\n")
        f.write(f"    : {enhanced_scores['contrast_ratio'] - original_scores['contrast_ratio']:+.2f}x ({((enhanced_scores['contrast_ratio'] - original_scores['contrast_ratio']) / original_scores['contrast_ratio'] * 100) if original_scores['contrast_ratio'] > 0 else 0:+.1f}%)\n")
        f.write(f"    : {enhanced_scores['concentration_ratio'] - original_scores['concentration_ratio']:+.2%}\n")
        f.write(f"    F3-Score (): {enhanced_scores['tam_score_harmonic'] - original_scores['tam_score_harmonic']:+.4f} ({((enhanced_scores['tam_score_harmonic'] - original_scores['tam_score_harmonic']) / original_scores['tam_score_harmonic'] * 100) if original_scores['tam_score_harmonic'] > 0 else 0:+.1f}%)\n")
        f.write(f"    F3-Score (): {enhanced_scores['tam_score_geometric'] - original_scores['tam_score_geometric']:+.4f} ({((enhanced_scores['tam_score_geometric'] - original_scores['tam_score_geometric']) / original_scores['tam_score_geometric'] * 100) if original_scores['tam_score_geometric'] > 0 else 0:+.1f}%)\n")
        f.write(f"\n:\n")
        f.write(f"  CAM: {image_name}_original_cam.jpg\n")
        f.write(f"  CAM: {image_name}_enhanced_cam.jpg\n")
        f.write(f"  : {image_name}_comparison.jpg\n")
        f.write(f"  : {image_name}_summary.jpg\n")
        f.write(f"  : {Path(summary_path).name}\n")


    result = {
        'image_path': image_path,
        'target_token': target_token,
        'selected_tokens': [sep_words[i] for i in select_logits_idx] if 'select_logits_idx' in locals() else None,
        'original_heatmap': heatmap,
        'heatmap_raw': heatmap_raw,               # for DACG sensitivity re-runs
        'enhanced_heatmap': enhanced_heatmap,
        '_tam_args': {                              # intermediate data for re-applying
            'image_array': np.array(image),
            'info4cam': info4cam,
            'tokens': tokens,
            'cam_logits': cam_logits,
            'processor': processor,
            'target_token_idx': select_logits_idx[0] if len(select_logits_idx) > 0 else 2,
        },
        'original_obj_iou': original_obj_iou,
        'enhanced_obj_iou': enhanced_obj_iou,
        'original_func_iou': original_func_iou,
        'enhanced_func_iou': enhanced_func_iou,
        'original_precision': original_precision,
        'enhanced_precision': enhanced_precision,
        'original_recall': original_recall,
        'enhanced_recall': enhanced_recall,
        'original_scores': original_scores,
        'enhanced_scores': enhanced_scores,
        'obj_iou_improvement': enhanced_obj_iou - original_obj_iou,
        'func_iou_improvement': enhanced_func_iou - original_func_iou,
        'comprehensive_improvement': enhanced_scores['tam_score'] - original_scores['tam_score'],
        'generation_time': t1 - t0,
        'save_dir': save_dir,
        'saved_files': {
            'original_cam': f"{image_name}_original_cam.jpg",
            'enhanced_cam': f"{image_name}_enhanced_cam.jpg",
            'comparison': f"{image_name}_comparison.jpg",
            'summary': f"{image_name}_summary.jpg",
            'summary_txt': summary_path
        }
    }

    return result

def load_selected_images(image_list_file):
    """Load image IDs from a text file.

    Args:
        image_list_file: path to file with image IDs (one per line)

    Returns:
        list: 
    """
    image_ids = []

    if not os.path.exists(image_list_file):
        return image_ids

    with open(image_list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.endswith('.jpg'):
                line = line[:-4]

            try:
                image_id = int(line)
                image_id_str = f"{image_id:012d}"
                image_ids.append(image_id_str)
            except ValueError:
                pass

    return image_ids

def test_selected_images(image_list_file, target_token=None, ablation_mode="all_methods"):
    """Run test."""

    image_ids = load_selected_images(image_list_file)
    if not image_ids:
        return

    coco_path = os.environ.get("COCO_DATASET_PATH", "")
    if not coco_path or not os.path.exists(coco_path):
        return

    try:
        input_data = prepare_input(coco_path)

        found_images = []
        for image_id in image_ids:
            for i, (img_path, prompt, captions, mask_path, category) in enumerate(input_data):
                img_filename = Path(img_path).stem
                if img_filename == image_id:
                    if os.path.exists(mask_path):
                        found_images.append((i, img_path, prompt, mask_path, category))
                        break
                    else:
                        break
            else:
                pass

        if not found_images:
            return


        success_count = 0
        fixed_prompt = "Describe this image in one sentence."
        for idx, (i, img_path, prompt, mask_path, category) in enumerate(found_images):
            result = process_single_image(img_path, fixed_prompt, mask_path, target_token, ablation_mode=ablation_mode)
            if result:
                success_count += 1
            else:
                pass


    except Exception as e:
        pass

def test_single_image(target_token=None, manual_indices=None):
    """"""
    if manual_indices is not None:
        pass
    elif target_token:
        pass
    else:
        pass

    local_image_paths = [
        os.environ.get("TEST_IMAGE_PATH", ""),
        "image/8.jpg",
        "images/test.jpg",
    ]

    image_path = None
    for path in local_image_paths:
        if os.path.exists(path):
            image_path = path
            break

    if image_path is None:
        return

    simple_prompt = "What do you see in this image?"
    result = process_single_image(image_path, simple_prompt, mask_path=None, target_token=target_token, manual_indices=manual_indices)
    if result:
        if target_token:
            pass
    else:
        pass

def test_dynamic_kernel_improvement():
    """"""

    test_cases = [
        {"step_ratio": 0.5, "activation_std": 0.1, "image_size": 256, "desc": ""},
        {"step_ratio": 0.5, "activation_std": 0.3, "image_size": 512, "desc": ""},
        {"step_ratio": 0.5, "activation_std": 0.5, "image_size": 1024, "desc": ""},
        {"step_ratio": 0.25, "activation_std": 0.2, "image_size": 512, "desc": ""},
    ]

    for case in test_cases:
        kernel_size = diffusion_aware_kernel_size(
            case["step_ratio"],
            case["activation_std"],
            case["image_size"]
        )
        denoising_steps = int(1 / case["step_ratio"])

def test_single_token_examples():
    """Test CAM for specific tokens."""

    test_tokens = ["cow", "horse", "person", "car", "dog"]

    for token in test_tokens:
        pass


def load_model_only():
    """Load LaViDa and register FullSequenceHook on the target block."""
    global model, tokenizer, image_processor, conv

    pretrained = os.environ.get("LAVIDA_MODEL_PATH", "lavida-llada-v1.0-instruct")
    vision_tower = os.environ.get("LAVIDA_VISION_TOWER", "siglip-so400m-patch14-384")
    model_name = "llava_llada"
    device_map = "cuda:0"

    conv_template = "llada"
    question = DEFAULT_IMAGE_TOKEN + "\nIs there anyone in the picture?"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    vision_kwargs = dict(
        mm_vision_tower=vision_tower,
        mm_resampler_type=None,
        mm_projector_type='mlp2x_gelu',
        mm_hidden_size=1152,
        use_mm_proj=True
    )
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map, vision_kwargs=vision_kwargs, torch_dtype='bfloat16', trust_remote_code=True
    )
    conv.tokenizer = tokenizer

    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)

    global full_sequence_hook
    target_layer = model.model.transformer.blocks[10]  #----------------change layer 0-30
    full_sequence_hook = FullSequenceHook(target_layer)

def main():
    """"""
    load_model_only()

    coco_path = os.environ.get("COCO_DATASET_PATH", "")
    grandf_path = os.environ.get("GRANDF_DATASET_PATH", "")

    if os.path.exists(coco_path):
        try:
            input_data = prepare_input(coco_path)

            images_with_masks = []
            for i, (img_path, prompt, captions, mask_path, category) in enumerate(input_data):
                if os.path.exists(mask_path):
                    images_with_masks.append(i)


            for idx, i in enumerate(images_with_masks[:20]):
                img_path, prompt, captions, mask_path, category = input_data[i]

            selected_indices = images_with_masks[478:520]

            # selected_indices = []
            # for i, (img_path, _, _, _, _) in enumerate(input_data):
            #     if Path(img_path).name in selected_image_names:
            #         selected_indices.append(i)

            # import random
            # selected_indices = random.sample(range(len(input_data)), min(10, len(input_data)))

            valid_indices = [i for i in selected_indices if i < len(input_data)]
            test_samples = len(valid_indices)

            if test_samples == 0:
                valid_indices = [0, 1, 2]
                test_samples = min(3, len(input_data))


            success_count = 0
            processed_count = 0
            skipped_count = 0
            obj_iou_improvements = []
            func_iou_improvements = []
            comprehensive_improvements = []
            original_obj_ious = []
            enhanced_obj_ious = []
            original_func_ious = []
            enhanced_func_ious = []
            original_comprehensive_scores = []
            enhanced_comprehensive_scores = []

            for idx, i in enumerate(valid_indices):
                img_path, prompt, captions, mask_path, category = input_data[i]

                if not os.path.exists(mask_path):
                    skipped_count += 1
                    continue

                processed_count += 1
                result = process_single_image(img_path, prompt, mask_path)
                if result and 'original_obj_iou' in result and 'enhanced_obj_iou' in result:
                    original_obj_iou = result['original_obj_iou']
                    enhanced_obj_iou = result['enhanced_obj_iou']
                    original_func_iou = result['original_func_iou']
                    enhanced_func_iou = result['enhanced_func_iou']
                    original_comprehensive_score = result['original_scores']['tam_score']
                    enhanced_comprehensive_score = result['enhanced_scores']['tam_score']

                    obj_iou_improvement = enhanced_obj_iou - original_obj_iou
                    func_iou_improvement = enhanced_func_iou - original_func_iou
                    comprehensive_improvement = enhanced_comprehensive_score - original_comprehensive_score

                    original_obj_ious.append(original_obj_iou)
                    enhanced_obj_ious.append(enhanced_obj_iou)
                    original_func_ious.append(original_func_iou)
                    enhanced_func_ious.append(enhanced_func_iou)
                    original_comprehensive_scores.append(original_comprehensive_score)
                    enhanced_comprehensive_scores.append(enhanced_comprehensive_score)
                    obj_iou_improvements.append(obj_iou_improvement)
                    func_iou_improvements.append(func_iou_improvement)
                    comprehensive_improvements.append(comprehensive_improvement)

                    success_count += 1
                else:
                    pass

            if obj_iou_improvements:
                avg_original_obj_iou = np.mean(original_obj_ious)
                avg_enhanced_obj_iou = np.mean(enhanced_obj_ious)
                avg_obj_improvement = np.mean(obj_iou_improvements)
                positive_obj_improvements = sum(1 for x in obj_iou_improvements if x > 0)
                obj_improvement_rate = positive_obj_improvements / len(obj_iou_improvements) * 100

                avg_original_func_iou = np.mean(original_func_ious)
                avg_enhanced_func_iou = np.mean(enhanced_func_ious)
                avg_func_improvement = np.mean(func_iou_improvements)
                positive_func_improvements = sum(1 for x in func_iou_improvements if x > 0)
                func_improvement_rate = positive_func_improvements / len(func_iou_improvements) * 100

                avg_original_comprehensive = np.mean(original_comprehensive_scores)
                avg_enhanced_comprehensive = np.mean(enhanced_comprehensive_scores)
                avg_comprehensive_improvement = np.mean(comprehensive_improvements)
                positive_comprehensive_improvements = sum(1 for x in comprehensive_improvements if x > 0)
                comprehensive_improvement_rate = positive_comprehensive_improvements / len(comprehensive_improvements) * 100


            else:
                pass

            if success_count == 0:
                test_single_image()

        except Exception as e:
            test_single_image()

    elif os.path.exists(grandf_path):
        try:
            annotations, images_dir = load_grandf_ha_data(grandf_path)

            if not annotations or not images_dir:
                test_single_image()
                return

            test_samples = min(10, len(annotations))

            success_count = 0
            processed_count = 0
            skipped_count = 0
            obj_iou_improvements = []
            func_iou_improvements = []
            comprehensive_improvements = []
            original_obj_ious = []
            enhanced_obj_ious = []
            original_func_ious = []
            enhanced_func_ious = []
            original_comprehensive_scores = []
            enhanced_comprehensive_scores = []

            for idx in range(test_samples):
                annotation = annotations[idx]
                image_name = annotation.get('image', '')

                if not image_name:
                    skipped_count += 1
                    continue


                img_path = os.path.join(images_dir, image_name)
                if not os.path.exists(img_path):
                    skipped_count += 1
                    continue

                combined_mask, object_info = decode_grandf_rle_masks(annotation, image_name)
                if combined_mask is None:
                    skipped_count += 1
                    continue

                mask_path = save_grandf_mask_as_png(combined_mask, image_name)
                if not mask_path:
                    skipped_count += 1
                    continue

                prompt = f"Describe what you see in this image. Focus on {object_info[0]['name'] if object_info else 'the main objects'}."

                result = process_single_image(img_path, prompt, mask_path)

                if result and 'original_obj_iou' in result and 'enhanced_obj_iou' in result:
                    original_obj_iou = result['original_obj_iou']
                    enhanced_obj_iou = result['enhanced_obj_iou']
                    original_func_iou = result['original_func_iou']
                    enhanced_func_iou = result['enhanced_func_iou']
                    original_comprehensive_score = result['original_scores']['tam_score']
                    enhanced_comprehensive_score = result['enhanced_scores']['tam_score']

                    original_obj_ious.append(original_obj_iou)
                    enhanced_obj_ious.append(enhanced_obj_iou)
                    original_func_ious.append(original_func_iou)
                    enhanced_func_ious.append(enhanced_func_iou)
                    original_comprehensive_scores.append(original_comprehensive_score)
                    enhanced_comprehensive_scores.append(enhanced_comprehensive_score)

                    obj_improvement = enhanced_obj_iou - original_obj_iou
                    func_improvement = enhanced_func_iou - original_func_iou
                    comprehensive_improvement = enhanced_comprehensive_score - original_comprehensive_score

                    obj_iou_improvements.append(obj_improvement)
                    func_iou_improvements.append(func_improvement)
                    comprehensive_improvements.append(comprehensive_improvement)

                    success_count += 1

                else:
                    pass

                processed_count += 1

                try:
                    if os.path.exists(mask_path):
                        os.remove(mask_path)
                except:
                    pass

            if success_count > 0:

                avg_original_obj_iou = np.mean(original_obj_ious)
                avg_enhanced_obj_iou = np.mean(enhanced_obj_ious)
                avg_obj_improvement = np.mean(obj_iou_improvements)
                positive_obj_improvements = sum(1 for x in obj_iou_improvements if x > 0)
                obj_improvement_rate = positive_obj_improvements / len(obj_iou_improvements) * 100


                avg_original_func_iou = np.mean(original_func_ious)
                avg_enhanced_func_iou = np.mean(enhanced_func_ious)
                avg_func_improvement = np.mean(func_iou_improvements)
                positive_func_improvements = sum(1 for x in func_iou_improvements if x > 0)
                func_improvement_rate = positive_func_improvements / len(func_iou_improvements) * 100


                avg_original_comprehensive = np.mean(original_comprehensive_scores)
                avg_enhanced_comprehensive = np.mean(enhanced_comprehensive_scores)
                avg_comprehensive_improvement = np.mean(comprehensive_improvements)
                positive_comprehensive_improvements = sum(1 for x in comprehensive_improvements if x > 0)
                comprehensive_improvement_rate = positive_comprehensive_improvements / len(comprehensive_improvements) * 100


            else:
                pass

            temp_dir = "temp_grandf_masks"
            if os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass

        except Exception:
            test_single_image()
    else:

        test_dynamic_kernel_improvement()

        test_single_image()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--selected_images" and len(sys.argv) > 2:
            image_list_file = sys.argv[2]
            target_token = None
            ablation_mode = "all_methods"

            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--target_token" and i + 1 < len(sys.argv):
                    target_token = sys.argv[i + 1]
                    i += 2
                elif sys.argv[i] == "--ablation_mode" and i + 1 < len(sys.argv):
                    ablation_mode = sys.argv[i + 1]
                    i += 2
                else:
                    i += 1

            if target_token:
                pass

            valid_modes = ["baseline", "gaussian_only", "confidence_only", "background_only", "all_methods"]
            if ablation_mode not in valid_modes:
                sys.exit(1)

            load_model_only()

            test_selected_images(image_list_file, target_token, ablation_mode)
        else:
            pass
    else:
        main()
