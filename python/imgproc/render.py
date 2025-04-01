#!/usr/bin/env python3

# ================================================================================
# File       : render.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Handles image composition including blending and mask-based
#              rendering
# Date       : 2025-04-01
# ================================================================================

import cv2
import numpy as np


def blur_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    return cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)


def blend_object_onto_background(
    bg: np.ndarray, obj: np.ndarray, mask: np.ndarray, pos_x: int, pos_y: int
) -> np.ndarray:
    h, w = obj.shape[:2]
    alpha = (mask / 255.0).reshape(h, w, 1)

    roi = bg[pos_y : pos_y + h, pos_x : pos_x + w]
    blended = alpha * obj + (1 - alpha) * roi
    bg[pos_y : pos_y + h, pos_x : pos_x + w] = blended
    return bg
