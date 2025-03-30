#!/usr/bin/env python3

# ================================================================================
# File       : interp / __init__.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Defines interpolation methods and the base Interpolation interface
# Date       : 2025-03-30
# ================================================================================

from abc import ABC, abstractmethod
from enum import Enum

import jax.numpy as jnp
import numpy as np


class ResizeMethod(Enum):
    NEAR = "nearest"
    LINE = "linear"
    CUBE = "cubic"
    LANC = "lanczos"


class Interpolation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def interpolate(self, image: np.ndarray, coords: np.ndarray) -> jnp.ndarray:
        pass
