import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance

"""
Noise
Contrast
Compression
Photorealistic Rain
Photorealistic Haze
Motion-Blur
Defocus-Blur
Backlight illumination
"""


def add_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def adjust_contrast(img, factor=1.5):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Contrast(pil_img)
    enhanced_img = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)


def jpeg_compression(img, quality=20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(encimg, 1)


def apply_motion_blur(img, kernel_size=15, angle=0):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)


def apply_defocus_blur(img, ksize=11):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_backlight(img, strength=0.7):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
    light = np.zeros_like(img)
    light[:, :] = (255, 255, 255)
    blended = cv2.addWeighted(img, 1 - strength, light, strength, 0)
    return np.where(mask[..., None] == 255, blended, img)


def add_rain(img):
    rain_layer = np.zeros_like(img)
    for _ in range(500):
        x, y = random.randint(0, img.shape[1]), random.randint(0, img.shape[0])
        length = random.randint(10, 20)
        cv2.line(rain_layer, (x, y), (x, y + length), (200, 200, 200), 1)
    return cv2.addWeighted(img, 0.8, rain_layer, 0.2, 0)


def add_haze(img, strength=0.5):
    white = np.full_like(img, 255)
    return cv2.addWeighted(img, 1 - strength, white, strength, 0)


def apply_global_distortions(image_path, save_path):
    img = cv2.imread(image_path)
    img = add_noise(img)
    img = adjust_contrast(img)
    img = jpeg_compression(img)
    img = add_rain(img)
    img = add_haze(img)
    img = apply_motion_blur(img)
    img = apply_defocus_blur(img)
    img = add_backlight(img)
    cv2.imwrite(save_path, img)


# Example usage
if __name__ == "__main__":
    apply_global_distortions("input.jpg", "distorted.jpg")
