#!/usr/bin/env python3

# ================================================================================
# File       : area.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: This script implements an inter-area (area-based) interpolation
#              operation
# Date       : 2025-04-02
# ================================================================================

import jax
import jax.numpy as jnp
import numpy as np

from . import Interpolation


class InterArea(Interpolation):
    @staticmethod
    @jax.jit
    def _interpolate(image: jnp.ndarray, coords: jnp.ndarray):
        h, w = image.shape[:2]

        def get_pixel(coord):
            x, y = coord

            x0 = jnp.floor(x).astype(jnp.int32)
            y0 = jnp.floor(y).astype(jnp.int32)

            x0 = jnp.clip(x0, 0, w - 2)
            y0 = jnp.clip(y0, 0, h - 2)

            region = jax.lax.dynamic_slice(
                image,
                start_indices=(y0, x0, 0) if image.ndim == 3 else (y0, x0),
                slice_sizes=(2, 2, image.shape[2]) if image.ndim == 3 else (2, 2),
            )

            return (
                jnp.mean(region, axis=(0, 1)) if region.ndim == 3 else jnp.mean(region)
            )

        return jax.vmap(get_pixel)(coords)

    def interpolate(self, image: np.ndarray, coords: np.ndarray) -> jnp.ndarray:
        image = jnp.array(image)
        coords = jnp.array(coords)
        return self._interpolate(image, coords).block_until_ready()
