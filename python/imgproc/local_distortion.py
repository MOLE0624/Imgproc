import cv2
import numpy as np

"""
Motion-Blur
Defocus-Blur
Backlight illumination
"""


def apply_local_mask(img, mask, effect_fn, *args, **kwargs):
    effected = effect_fn(img.copy(), *args, **kwargs)
    return np.where(mask[..., None] == 255, effected, img)


def motion_blur(img, kernel_size=15, angle=0):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
    kernel = kernel / kernel_size
    return cv2.filter2D(img, -1, kernel)


def defocus_blur(img, ksize=11):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def backlight_illumination(img, center=None, strength=0.7):
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, min(h, w) // 4, 255, -1)
    light = np.full_like(img, 255)
    blended = cv2.addWeighted(img, 1 - strength, light, strength, 0)
    return np.where(mask[..., None] == 255, blended, img)


def apply_local_distortions(image_path, save_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Create local masks
    motion_mask = np.zeros((h, w), dtype=np.uint8)
    defocus_mask = np.zeros((h, w), dtype=np.uint8)
    backlight_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.rectangle(motion_mask, (50, 50), (w // 2, h // 2), 255, -1)
    cv2.circle(defocus_mask, (w // 2, h // 2), 100, 255, -1)
    cv2.ellipse(backlight_mask, (3 * w // 4, h // 3), (80, 80), 0, 0, 360, 255, -1)

    img = apply_local_mask(img, motion_mask, motion_blur, kernel_size=25, angle=45)
    img = apply_local_mask(img, defocus_mask, defocus_blur, ksize=21)
    img = apply_local_mask(img, backlight_mask, backlight_illumination, strength=0.5)

    cv2.imwrite(save_path, img)


# Example usage
if __name__ == "__main__":
    apply_local_distortions("input.jpg", "local_distorted.jpg")
