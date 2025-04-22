#!/usr/bin/env python3

# ================================================================================
# File       : synth.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Orchestrates object synthesis onto backgrounds with shadow,
#              blending, and annotation
# Date       : 2025-04-01
# ================================================================================

import os
import random

import numpy as np
from imgproc.adjust import adjust_color_auto
from imgproc.geometry import get_random_scale_and_position
from imgproc.render import blend_object_onto_background, blur_mask
from imgproc.shadow import apply_shadow_to_image, generate_shadow_mask
from PIL import Image


def load_random_background(bg_dir, size):
    files = [f for f in os.listdir(bg_dir) if f.endswith((".jpg", ".png"))]
    path = os.path.join(bg_dir, random.choice(files))
    return Image.open(path).convert("RGB").resize(size)


def compose_object_on_background(obj_path, bg_dir, output_path, canvas_size=(512, 512)):
    obj_img = Image.open(obj_path).convert("RGBA")
    bg_img = load_random_background(bg_dir, canvas_size)

    obj_img = adjust_color_auto(obj_img, bg_img)

    # Determine size and position
    (new_w, new_h), (pos_x, pos_y) = get_random_scale_and_position(
        bg_img.size, obj_img.size
    )
    obj_img = obj_img.resize((new_w, new_h), Image.LANCZOS)

    mask = np.array(obj_img.split()[-1])
    blurred_mask = blur_mask(mask)
    obj_np = np.array(obj_img.convert("RGB"))
    bg_np = np.array(bg_img)

    # Add shadow
    shadow_mask = generate_shadow_mask(blurred_mask)
    bg_np = apply_shadow_to_image(bg_np, shadow_mask)

    # Synthesis
    composed = blend_object_onto_background(bg_np, obj_np, blurred_mask, pos_x, pos_y)
    Image.fromarray(composed.astype(np.uint8)).save(output_path)
    print(f"[main]: Saved result to {output_path}")


def compose_multiple_objects_on_background(
    obj_paths, bg_dir, output_path, class_ids, canvas_size=(512, 512)
):
    from imgproc.adjust import adjust_color_auto
    from imgproc.annotation import save_bbox_annotation, save_mask_annotation
    from imgproc.geometry import get_random_scale_and_position
    from imgproc.render import blend_object_onto_background, blur_mask
    from imgproc.shadow import apply_shadow_to_image, generate_shadow_mask

    bg_img = load_random_background(bg_dir, canvas_size)
    bg_np = np.array(bg_img)

    annotations = []
    masks_out = []

    for obj_path, class_id in zip(obj_paths, class_ids):
        obj_img = Image.open(obj_path).convert("RGBA")
        obj_img = adjust_color_auto(obj_img, bg_img)

        (new_w, new_h), (pos_x, pos_y) = get_random_scale_and_position(
            bg_img.size, obj_img.size
        )
        obj_img = obj_img.resize((new_w, new_h), Image.LANCZOS)

        mask = np.array(obj_img.split()[-1])
        blurred_mask = blur_mask(mask)
        obj_np = np.array(obj_img.convert("RGB"))

        shadow_mask = generate_shadow_mask(blurred_mask)
        bg_np = apply_shadow_to_image(bg_np, shadow_mask)

        bg_np = blend_object_onto_background(bg_np, obj_np, blurred_mask, pos_x, pos_y)

        # Annotation information
        annotations.append({"class_id": class_id, "bbox": [pos_x, pos_y, new_w, new_h]})

        masks_out.append(blurred_mask)

    # Save process
    Image.fromarray(bg_np.astype(np.uint8)).save(output_path)
    base = os.path.splitext(os.path.basename(output_path))[0]

    # Save mask（possible synth to one file）
    for i, m in enumerate(masks_out):
        save_mask_annotation(m, f"{os.path.dirname(output_path)}/{base}_mask{i}.png")

    # Save COCO format annotation
    from imgproc.annotation import save_coco_annotations

    save_coco_annotations(
        output_dir=os.path.dirname(output_path),
        filename=os.path.basename(output_path),
        image_size=bg_img.size,
        annotations=annotations,
    )

    print(f"[main]: Saved result to {output_path}")
