#!/usr/bin/env python3

# ================================================================================
# File       : linear.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a bilinear interpolation operation
# Date       : 2025-03-31
# ================================================================================

import jax
import jax.numpy as jnp
import numpy as np
from interp import Interpolation


class Bilinear(Interpolation):
    @staticmethod
    @jax.jit
    def _interpolate(image: jnp.ndarray, coords: jnp.ndarray):
        def get_pixel(coord):
            x, y = coord
            h, w = image.shape[:2]

            x0 = jnp.floor(x).astype(jnp.int32)
            x1 = jnp.clip(x0 + 1, 0, w - 1)
            x0 = jnp.clip(x0, 0, w - 1)

            y0 = jnp.floor(y).astype(jnp.int32)
            y1 = jnp.clip(y0 + 1, 0, h - 1)
            y0 = jnp.clip(y0, 0, h - 1)

            wa = (x1 - x) * (y1 - y)
            wb = (x - x0) * (y1 - y)
            wc = (x1 - x) * (y - y0)
            wd = (x - x0) * (y - y0)

            Ia = image[y0, x0]
            Ib = image[y0, x1]
            Ic = image[y1, x0]
            Id = image[y1, x1]

            return wa * Ia + wb * Ib + wc * Ic + wd * Id

        return jax.vmap(get_pixel)(coords)

    def interpolate(self, image: np.ndarray, coords: np.ndarray) -> jnp.ndarray:
        image = jnp.array(image)
        coords = jnp.array(coords)
        return self._interpolate(image, coords).block_until_ready()
