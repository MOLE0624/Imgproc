#!/usr/bin/env python3

# ================================================================================
# File       : resize.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a resize operation
# Date       : 2025-03-30
# ================================================================================

import jax.numpy as jnp
import numpy as np
from interp import ResizeMethod
from interp.linear import Bilinear
from interp.nearest import NearestNeighbor


class Resized:
    def __init__(self, jax_array: jnp.ndarray):
        self._jax_array = jax_array

    def np(self) -> np.ndarray:
        return np.array(self._jax_array.block_until_ready()).astype(np.uint8)

    def jax(self) -> jnp.ndarray:
        return self._jax_array

    def __getitem__(self, key):
        return self._jax_array[key]

    @property
    def shape(self):
        return self._jax_array.shape


class Resize:
    def __init__(self, method: ResizeMethod):
        if method == ResizeMethod.NEAR:
            self.interpolation = NearestNeighbor()
        elif method == ResizeMethod.LINEAR:
            self.interpolation = Bilinear()
        else:
            raise NotImplementedError(
                f"{method.value} interpolation not yet implemented"
            )

        self._warm_cache = set()

    def _warmup(self, image: np.ndarray, new_height: int, new_width: int):
        input_shape = image.shape
        output_shape = (new_height, new_width)
        key = (input_shape, output_shape)

        if key not in self._warm_cache:
            _ = self._resize(image, new_height, new_width)
            self._warm_cache.add(key)

    def _resize(self, image: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
        original_height, original_width = image.shape[:2]

        grid_y, grid_x = jnp.meshgrid(
            jnp.arange(new_height), jnp.arange(new_width), indexing="ij"
        )
        src_x = (grid_x / new_width) * original_width
        src_y = (grid_y / new_height) * original_height
        coords = jnp.stack([src_x.ravel(), src_y.ravel()], axis=1)

        interpolated = self.interpolation.interpolate(image, coords)
        return interpolated.reshape((new_height, new_width, *image.shape[2:]))

    def resize(self, image: np.ndarray, new_height: int, new_width: int) -> Resized:
        self._warmup(image, new_height, new_width)
        return Resized(self._resize(image, new_height, new_width))


if __name__ == "__main__":
    import cv2

    img = cv2.imread("../assets/image/pearlmole.png")
    resizer = Resize(ResizeMethod.NEAR)
    resized_img = resizer.resize(img, 100, 100).np()

    cv2.imwrite("resize_100_100.jpg", resized_img)
