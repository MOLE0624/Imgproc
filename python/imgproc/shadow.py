#!/usr/bin/env python3

# ================================================================================
# File       : shadow.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Generates and applies soft shadows based on object masks
#              mapping
# Date       : 2025-04-01
# ================================================================================

import cv2
import numpy as np


def generate_shadow_mask(
    mask: np.ndarray, offset=(5, 5), intensity=0.4, blur_radius=7
) -> np.ndarray:
    shadow_mask = np.roll(mask, offset[1], axis=0)
    shadow_mask = np.roll(shadow_mask, offset[0], axis=1)
    shadow_mask = cv2.GaussianBlur(
        shadow_mask, (blur_radius * 2 + 1, blur_radius * 2 + 1), 0
    )
    return (shadow_mask * intensity).astype(np.uint8)


def apply_shadow_to_image(image: np.ndarray, shadow_mask: np.ndarray) -> np.ndarray:
    shadow_rgb = np.stack([shadow_mask] * 3, axis=2)
    shadowed = image.astype(float) * (1 - shadow_rgb / 255.0)
    return np.clip(shadowed, 0, 255).astype(np.uint8)
