#!/usr/bin/env python3

# ================================================================================
# File       : test_resize.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Tests for Resize using nearest neighbor interpolation
# Date       : 2025-03-30
# ================================================================================

import time
import unittest

import numpy as np
from imgproc.interp import ResizeMethod
from imgproc.resize import Resize


def create_test_image(height, width, channels=3):
    """Creates a test image with a simple gradient pattern"""
    image = np.zeros((height, width, channels), dtype=np.uint8)
    for c in range(channels):
        image[..., c] = np.linspace(0, 255, width, dtype=np.uint8)
    return image


class TestResizeNearestNeighbor(unittest.TestCase):

    def run_resize_test(self, description, input_shape, output_shape):
        img = create_test_image(*input_shape)
        resizer = Resize(ResizeMethod.NEAR)
        resized = resizer.resize(img, *output_shape)

        expected_shape = (output_shape[0], output_shape[1], input_shape[2])
        self.assertEqual(
            resized.shape, expected_shape, f"{description}: shape mismatch"
        )

        self.assertTrue(
            resized.min() >= 0 and resized.max() <= 255,
            f"{description}: pixel value out of range",
        )

    def test_downsampling_rgb(self):
        self.run_resize_test(
            "Downsampling RGB", input_shape=(64, 64, 3), output_shape=(16, 16)
        )

    def test_upsampling_rgb(self):
        self.run_resize_test(
            "Upsampling RGB", input_shape=(16, 16, 3), output_shape=(64, 64)
        )

    def test_identity_rgb(self):
        self.run_resize_test(
            "Identity RGB", input_shape=(32, 32, 3), output_shape=(32, 32)
        )

    def test_non_square_rgb(self):
        self.run_resize_test(
            "Non-square RGB", input_shape=(32, 64, 3), output_shape=(64, 32)
        )

    def test_upsampling_grayscale(self):
        self.run_resize_test(
            "Upsampling Grayscale", input_shape=(16, 16, 1), output_shape=(32, 32)
        )

    def test_performance_large_image(self):
        img = create_test_image(1024, 1024, 3)
        resizer = Resize(ResizeMethod.NEAR)

        # Warm-up (compilation time excluded)
        _ = resizer.resize(img, 512, 512)

        start = time.time()
        _ = resizer.resize(img, 512, 512)
        elapsed_ms = (time.time() - start) * 1000  # milliseconds

        print(f"\nPerformance Test (1024x1024 → 512x512): {elapsed_ms:.2f} ms")
        self.assertLess(
            elapsed_ms, 100.0, "Resize took too long"
        )  # adjust threshold as needed


if __name__ == "__main__":
    unittest.main()
