#!/usr/bin/env python3

# ================================================================================
# File       : nearest.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements a nearest neighbor interpolation operation
# Date       : 2025-03-30
# ================================================================================

import jax
import jax.numpy as jnp
import numpy as np

from . import Interpolation


class NearestNeighbor(Interpolation):
    @staticmethod
    @jax.jit
    def _interpolate(image: jnp.ndarray, coords: jnp.ndarray):
        def get_pixel(coord):
            x, y = coord
            xi = jnp.clip(jnp.round(x).astype(int), 0, image.shape[1] - 1)
            yi = jnp.clip(jnp.round(y).astype(int), 0, image.shape[0] - 1)
            return image[yi, xi]

        return jax.vmap(get_pixel)(coords)

    def interpolate(self, image: np.ndarray, coords: np.ndarray) -> jnp.ndarray:
        image = jnp.array(image)
        coords = jnp.array(coords)
        return self._interpolate(image, coords).block_until_ready()
