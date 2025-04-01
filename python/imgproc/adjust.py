#!/usr/bin/env python3

# ================================================================================
# File       : adjust.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Implements automatic brightness and contrast adjustment based on
#              background image
# Date       : 2025-04-01
# ================================================================================

import cv2
import numpy as np
from PIL import ImageEnhance


def adjust_color_auto(obj_img_pil, bg_img_pil):
    obj_np = np.array(obj_img_pil.convert("RGB"))
    bg_np = np.array(bg_img_pil.convert("RGB"))

    obj_gray = cv2.cvtColor(obj_np, cv2.COLOR_RGB2GRAY)
    bg_gray = cv2.cvtColor(bg_np, cv2.COLOR_RGB2GRAY)

    brightness = np.clip(np.mean(bg_gray) / (np.mean(obj_gray) + 1e-5), 0.8, 1.2)
    contrast = np.clip(np.std(bg_gray) / (np.std(obj_gray) + 1e-5), 0.8, 1.2)

    img = ImageEnhance.Brightness(obj_img_pil).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img
