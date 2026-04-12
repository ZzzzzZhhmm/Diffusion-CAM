"""GradCAM baseline: generate() + info4cam patch features, hook on a transformer block (optional VLM checkpoint)."""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import time
import copy

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIRD = os.path.join(_REPO, "third_party")
_METHOD = os.path.join(_REPO, "method")
for _p in (_METHOD, _THIRD, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def load_checkpoint_model():
    """Load optional VLM checkpoint (e.g. LaViDa weights); paths from LAVIDA_MODEL_PATH and LAVIDA_VISION_TOWER."""
    pretrained = os.environ.get("LAVIDA_MODEL_PATH", "lavida-llada-v1.0-instruct")
    vision_tower = os.environ.get("LAVIDA_VISION_TOWER", "siglip-so400m-patch14-384")
    model_name = "llava_llada"
    device_map = "cuda:0"
    vision_kwargs = dict(
        mm_vision_tower=vision_tower,
        mm_resampler_type=None,
        mm_projector_type='mlp2x_gelu',
        mm_hidden_size=1152,
        use_mm_proj=True
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained, None, model_name, device_map=device_map,
        vision_kwargs=vision_kwargs, torch_dtype='bfloat16', trust_remote_code=True
    )
    model.eval()
    model.tie_weights()
    model.to(torch.bfloat16)
    return tokenizer, model, image_processor


def generate_gradcam(image_path, prompt, target_layer_idx=10, target_token_indices=None,
                     model=None, tokenizer=None, image_processor=None):
    """Run GradCAM and return heatmap, elapsed time, token ids, text outputs."""
    if model is None or tokenizer is None or image_processor is None:
        tokenizer, model, image_processor = load_checkpoint_model()

    conv_template = "llada"
    question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    conv.tokenizer = tokenizer
    prompt_question = conv.get_prompt()

    image = Image.open(image_path).convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device='cuda') for _image in image_tensor]

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to('cuda')
    image_sizes = [image.size]

    feature_maps = []
    gradients = []

    def save_feature_maps_hook(module, input, output):
        def _save_grad(grad):
            if 4096 in grad.shape:
                gradients.append(grad)
        if isinstance(output, tuple):
            for o in output:
                if isinstance(o, tuple):
                    for i in o:
                        if not i.requires_grad:
                            i.requires_grad = True
                        if isinstance(i, torch.Tensor) and i.requires_grad:
                            feature_maps.append(i)
                            i.retain_grad()
                            i.register_hook(_save_grad)
                elif isinstance(o, torch.Tensor):
                    if not o.requires_grad:
                        o.requires_grad = True
                    feature_maps.append(o)
                    o.retain_grad()
                    o.register_hook(_save_grad)
        elif isinstance(output, torch.Tensor):
            if not output.requires_grad:
                output.requires_grad = True
            output.retain_grad()
            feature_maps.append(output)
            output.register_hook(_save_grad)

    target_layer = model.model.transformer.blocks[target_layer_idx]
    hook_handle = target_layer.register_forward_hook(save_feature_maps_hook)

    try:
        t0 = time.time()
        generate_output = model.generate(
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
            verbose=False,
            schedule='shift',
        )
        if len(generate_output) == 2:
            output_tuple, info4cam = generate_output
            if len(output_tuple) == 3:
                cont, hist, cam_logits = output_tuple
            elif len(output_tuple) == 2:
                cont, cam_logits = output_tuple
            else:
                raise ValueError(f"Unexpected output_tuple length: {len(output_tuple)}")
        else:
            raise ValueError(f"Unexpected generate output length: {len(generate_output)}")

        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        sep_words = [tokenizer.batch_decode(i, skip_special_tokens=True) for i in cont][0]

        if isinstance(target_token_indices, str):
            from predict import find_target_token
            indices = find_target_token(sep_words, target_token_indices)
            if indices:
                select_logits_idx = np.array(indices)
            else:
                select_logits_idx = np.array([2])
        elif target_token_indices is None:
            select_logits_idx = np.array([2])
        else:
            select_logits_idx = np.array(target_token_indices) - 1

        target_logits = cam_logits[-1][0][select_logits_idx]
        target_logits.sum().backward(retain_graph=True)

        if len(feature_maps) == 0 or len(gradients) < 2:
            raise ValueError(
                f"Missing features/gradients: feature_maps={len(feature_maps)}, gradients={len(gradients)}"
            )

        bound, (padfeat_hws, basefeat_shape) = info4cam
        padf_h, padf_w = padfeat_hws
        img_feature = feature_maps[0][0][
            bound[0] + basefeat_shape[0]:bound[0] + basefeat_shape[0] + padf_h * padf_w, :
        ]
        grad = gradients[1][0][
            bound[0] + basefeat_shape[0]:bound[0] + basefeat_shape[0] + padf_h * padf_w, :
        ]

        from einops import rearrange
        img_feature = rearrange(img_feature, '(h w) d -> d h w', w=padf_w, h=padf_h)
        grad = rearrange(grad, '(h w) d -> h w d', w=padf_w, h=padf_h)

        pooled_gradients = torch.mean(grad, dim=[0, 1])
        activation = img_feature
        for i in range(activation.size(0)):
            activation[i, :, :] *= pooled_gradients[i]
        activation = F.relu(activation)
        heatmap = torch.mean(activation, dim=0, dtype=torch.float32).detach().cpu().numpy()
        heatmap -= heatmap.min()
        heatmap /= (heatmap.max() + 1e-8)

        image_h, image_w = np.array(image).shape[:2]
        heatmap = cv2.resize(heatmap, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        t1 = time.time()
        return heatmap, t1 - t0, cont[0].tolist(), text_outputs

    except Exception:
        return None, 0, None, None
    finally:
        hook_handle.remove()


def process_single_image_gradcam(image_path, prompt, mask_path=None, target_layer_idx=10,
                                 model=None, tokenizer=None, image_processor=None, target_token=None):
    """GradCAM for one image; optional mask for IoU metrics."""
    target_token_to_use = target_token if target_token else None
    gradcam_heatmap, generation_time, tokens, text_outputs = generate_gradcam(
        image_path, prompt, target_layer_idx, target_token_to_use,
        model=model, tokenizer=tokenizer, image_processor=image_processor
    )
    if gradcam_heatmap is None:
        return None

    iou_result = None
    if mask_path and os.path.exists(mask_path):
        from predict import compute_obj_func_iou

        class SimpleProcessor:
            def __init__(self, tok):
                self.tokenizer = tok

            def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
                return self.tokenizer.batch_decode(
                    ids, skip_special_tokens=skip_special_tokens,
                    clean_up_tokenization_spaces=clean_up_tokenization_spaces
                )

        processor = SimpleProcessor(tokenizer)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            heatmap_resized = cv2.resize(gradcam_heatmap, (mask.shape[1], mask.shape[0]))
            heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
            otsu_threshold, _ = cv2.threshold(heatmap_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, heatmap_binary = cv2.threshold(heatmap_uint8, otsu_threshold, 255, cv2.THRESH_BINARY)

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
                        obj_iou = max_iou
                        precision = precisions[ious.index(max_iou)]
                        recall = recalls[ious.index(max_iou)]
                        noun_fg_thresh = [otsu_threshold]

            func_iou = 0.0
            if len(noun_fg_thresh) > 0:
                fg_thresh = sum(noun_fg_thresh) / len(noun_fg_thresh)
                neg_iou = float((heatmap_resized < fg_thresh / 255.0).sum()) / heatmap_resized.size
                func_iou = neg_iou

            iou_result = {
                'obj_iou': obj_iou,
                'func_iou': func_iou,
                'precision': precision,
                'recall': recall
            }

    return {
        'image_path': image_path,
        'gradcam_heatmap': gradcam_heatmap,
        'iou_result': iou_result,
        'generation_time': generation_time,
        'target_layer': target_layer_idx
    }


def test_gradcam_on_coco():
    """Optional COCO smoke test; set COCO_DATASET_PATH to a prepared coco_dataset folder."""
    from diffusion_cam.dataset_prep import prepare_input
    coco = os.environ.get("COCO_DATASET_PATH")
    if not coco or not os.path.isdir(coco):
        return
    input_data = prepare_input(coco)
    if not input_data:
        return
    test_samples = min(3, len(input_data))
    for i in range(test_samples):
        img_path, prompt, _, mask_path, _ = input_data[i]
        if mask_path and not os.path.exists(mask_path):
            continue
        process_single_image_gradcam(img_path, prompt, mask_path)


if __name__ == "__main__":
    test_gradcam_on_coco()
