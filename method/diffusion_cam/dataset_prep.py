# SPDX-License-Identifier: Apache-2.0
# Dataset listing helpers adapted from the TAM evaluation utilities.
# See NOTICE.txt in this directory for attribution.

import json
import os


def prepare_input(dataset_path, processed_input=""):
    """Build lists of (image_path, prompt, captions, seg_path, category) for COCO / GranDf-style layouts."""
    input_data = []
    if processed_input != "":
        with open(os.path.join(dataset_path, processed_input), encoding="utf-8") as f:
            return json.load(f)

    if "coco" in dataset_path:
        with open(os.path.join(dataset_path, "annotations/instances_val2014.json"), encoding="utf-8") as f:
            seg_anno = json.load(f)
        with open(os.path.join(dataset_path, "annotations/captions_val2014.json"), encoding="utf-8") as f:
            cap_anno = json.load(f)
        defualt_prompt = "Write a one-sentence caption for this image:"
        category = {
            "person": 1,
            "bicycle": 2,
            "car": 3,
            "motorcycle": 4,
            "airplane": 5,
            "bus": 6,
            "train": 7,
            "truck": 8,
            "boat": 9,
            "traffic light": 10,
            "fire hydrant": 11,
            "stop sign": 13,
            "parking meter": 14,
            "bench": 15,
            "bird": 16,
            "cat": 17,
            "dog": 18,
            "horse": 19,
            "sheep": 20,
            "cow": 21,
            "elephant": 22,
            "bear": 23,
            "zebra": 24,
            "giraffe": 25,
            "backpack": 27,
            "umbrella": 28,
            "handbag": 31,
            "tie": 32,
            "suitcase": 33,
            "frisbee": 34,
            "skis": 35,
            "snowboard": 36,
            "ball": 37,
            "kite": 38,
            "baseball bat": 39,
            "baseball glove": 40,
            "skateboard": 41,
            "surfboard": 42,
            "tennis racket": 43,
            "bottle": 44,
            "glass": 46,
            "cup": 47,
            "fork": 48,
            "knife": 49,
            "spoon": 50,
            "bowl": 51,
            "banana": 52,
            "apple": 53,
            "sandwich": 54,
            "orange": 55,
            "broccoli": 56,
            "carrot": 57,
            "hot dog": 58,
            "pizza": 59,
            "donut": 60,
            "cake": 61,
            "chair": 62,
            "couch": 63,
            "potted plant": 64,
            "bed": 65,
            "dining table": 67,
            "toilet": 70,
            "tv": 72,
            "laptop": 73,
            "mouse": 74,
            "remote": 75,
            "keyboard": 76,
            "cell phone": 77,
            "microwave": 78,
            "oven": 79,
            "toaster": 80,
            "sink": 81,
            "refrigerator": 82,
            "book": 84,
            "clock": 85,
            "vase": 86,
            "scissors": 87,
            "teddy bear": 88,
            "hair drier": 89,
            "toothbrush": 90,
        }
        cap_dic = {}
        for _ in cap_anno["annotations"]:
            if _["image_id"] not in cap_dic:
                cap_dic[_["image_id"]] = [_["caption"]]
            else:
                cap_dic[_["image_id"]].append(_["caption"])

        for _ in seg_anno["images"]:
            fn = str(_["id"]).zfill(12)
            input_data.append(
                [
                    os.path.join(dataset_path, "image", fn + ".jpg"),
                    defualt_prompt,
                    cap_dic[_["id"]],
                    os.path.join(dataset_path, "seg_label", fn + ".png"),
                    category,
                ]
            )

    elif "GranDf" in dataset_path or "OpenPSG" in dataset_path:
        with open(os.path.join(dataset_path, "anno.json"), encoding="utf-8") as f:
            data = json.load(f)
        if "GranDf" in dataset_path:
            defualt_prompt = "Write a description for this image using around two sentences:"
        else:
            defualt_prompt = "Write a description for this image using around three sentences:"
        for _ in data:
            input_data.append(
                [
                    os.path.join(dataset_path, _[0]),
                    defualt_prompt,
                    [_[1]],
                    os.path.join(dataset_path, _[2]),
                    _[3],
                ]
            )

    return input_data
