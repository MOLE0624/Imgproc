#!/usr/bin/env python3

# ================================================================================
# File       : cubic.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a bicubic interpolation operation
# Date       : 2025-03-31
# ================================================================================

import jax
import jax.numpy as jnp
import numpy as np
from interp import Interpolation


class Bicubic(Interpolation):
    @staticmethod
    def _cubic_kernel(x: jnp.ndarray) -> jnp.ndarray:
        """Keys' cubic convolution kernel (a = -0.5)"""
        absx = jnp.abs(x)
        absx2 = absx**2
        absx3 = absx**3

        a = -0.5

        return jnp.where(
            absx <= 1,
            (a + 2) * absx3 - (a + 3) * absx2 + 1,
            jnp.where(
                (absx > 1) & (absx < 2),
                a * absx3 - 5 * a * absx2 + 8 * a * absx - 4 * a,
                0.0,
            ),
        )

    @staticmethod
    @jax.jit
    def _interpolate(image: jnp.ndarray, coords: jnp.ndarray):
        def get_pixel(coord):
            x, y = coord
            h, w = image.shape[:2]

            x0 = jnp.floor(x).astype(int)
            y0 = jnp.floor(y).astype(int)

            # Sample 4x4 grid around (x, y)
            dx = jnp.arange(-1, 3)
            dy = jnp.arange(-1, 3)

            grid_x = jnp.clip(x0 + dx, 0, w - 1)
            grid_y = jnp.clip(y0 + dy, 0, h - 1)

            # Compute weights
            wx = Bicubic._cubic_kernel(x - grid_x)
            wy = Bicubic._cubic_kernel(y - grid_y)

            wx = wx / jnp.sum(wx)
            wy = wy / jnp.sum(wy)

            # Collect 4x4 patch
            def sample_row(j):
                return image[grid_y[j], grid_x]

            patch = jnp.stack([sample_row(j) for j in range(4)])

            return jnp.tensordot(wy, jnp.tensordot(patch, wx, axes=(1, 0)), axes=(0, 0))

        return jax.vmap(get_pixel)(coords)

    def interpolate(self, image: np.ndarray, coords: np.ndarray) -> jnp.ndarray:
        image = jnp.array(image)
        coords = jnp.array(coords)
        return self._interpolate(image, coords).block_until_ready()
