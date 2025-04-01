#!/usr/bin/env python3

# ================================================================================
# File       : annotation.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Provides functions to generate and save bounding box and mask
#              annotations
# Date       : 2025-04-01
# ================================================================================

import json
import os

import numpy as np
from PIL import Image


def save_bbox_annotation(
    output_dir, filename, class_id, x, y, w, h, image_size, format="yolo"
):
    """
    Save bounding box in YOLO or COCO format

    Parameters:
        x, y         : Top-left corner (absolute)
        w, h         : Width and height
        image_size   : (width, height) of the full image
        format       : "yolo" or "coco"
    """
    img_w, img_h = image_size

    if format == "yolo":
        # YOLO format: center-based, normalized
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        rel_w = w / img_w
        rel_h = h / img_h
        line = f"{class_id} {x_center:.6f} {y_center:.6f} {rel_w:.6f} {rel_h:.6f}\n"

        os.makedirs(output_dir, exist_ok=True)
        label_path = os.path.join(output_dir, filename.replace(".png", ".txt"))
        with open(label_path, "w") as f:
            f.write(line)

    elif format == "coco":
        # COCO format: top-left based with width and height
        ann = {"image": filename, "bbox": [x, y, w, h], "category_id": class_id}
        os.makedirs(output_dir, exist_ok=True)
        label_path = os.path.join(output_dir, filename.replace(".png", ".json"))
        with open(label_path, "w") as f:
            json.dump(ann, f, indent=2)


def save_mask_annotation(mask: np.ndarray, output_path):
    """
    Save binary mask image as PNG

    Parameters:
        mask: 0-255 uint8 binary mask (numpy array)
        output_path: file path to save
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_img.save(output_path)


def save_coco_annotations(output_dir, filename, image_size, annotations):
    """
    Save COCO-style annotations (multiple bboxes) for one image

    Parameters:
        output_dir : Output directory
        filename   : Target image filename
        image_size : (width, height)
        annotations: List of dicts with {class_id, bbox=[x, y, w, h]}
    """
    coco_ann = {
        "image": {
            "file_name": filename,
            "width": image_size[0],
            "height": image_size[1],
        },
        "annotations": [
            {
                "category_id": ann["class_id"],
                "bbox": ann["bbox"],
                "area": ann["bbox"][2] * ann["bbox"][3],
                "iscrowd": 0,
            }
            for ann in annotations
        ],
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, filename.replace(".png", ".json"))
    with open(json_path, "w") as f:
        json.dump(coco_ann, f, indent=2)
