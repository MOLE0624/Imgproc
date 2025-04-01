#!/usr/bin/env python3

# ================================================================================
# File       : geometry.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Implements geometric transformations including bbox rotation and
#              coordinate mapping
# Date       : 2025-04-01
# ================================================================================

import math
import random
from enum import Enum, auto
from typing import List, Tuple


class SnapMode(Enum):
    ROUND = auto()
    FLOOR = auto()
    CEIL = auto()


# +-------------------------------+--------+-------+
# | Step                          | Float  | Int   |
# +-------------------------------+--------+-------+
# | Image transformations         |   ✅   |  ❌   |
# | Coordinate math (rotation)    |   ✅   |  ❌   |
# | Intermediate visualizations   |   ✅   |  ❌   |
# | Save to file (YOLO, COCO)     |   ❌   |  ✅   |
# | Draw rectangle on image (cv2) |   ❌   |  ✅   |
# +-------------------------------+--------+-------+


def get_random_scale_and_position(
    bg_size: Tuple[int, int],
    obj_size: Tuple[int, int],
    scale_range: Tuple[float, float] = (0.4, 0.9),
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    bg_w, bg_h = bg_size
    obj_w, obj_h = obj_size

    scale = random.uniform(*scale_range)
    new_w = int(obj_w * scale)
    new_h = int(obj_h * scale)

    max_x = max(0, bg_w - new_w)
    max_y = max(0, bg_h - new_h)

    pos_x = random.randint(0, max_x)
    pos_y = random.randint(0, max_y)

    return (new_w, new_h), (pos_x, pos_y)


def rotate_point(
    x: float, y: float, cx: float, cy: float, angle_rad: float
) -> Tuple[float, float]:
    """Rotate a point (x, y) around center (cx, cy) by angle (radians)."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x -= cx
    y -= cy
    x_new = cos_a * x - sin_a * y + cx
    y_new = sin_a * x + cos_a * y + cy
    return x_new, y_new


def transform_points_in_rotated_bbox(
    bbox: Tuple[float, float, float, float],
    points: List[Tuple[float, float]],
    angle_deg: float,
) -> Tuple[Tuple[float, float, float, float], List[Tuple[float, float]]]:
    """
    Rotate a bounding box and map local coordinates to new local coordinates
    inside the rotated (axis-aligned) bounding box.
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    angle_rad = math.radians(angle_deg)

    abs_points = [(x + px, y + py) for (px, py) in points]
    rotated_abs = [rotate_point(px, py, cx, cy, angle_rad) for px, py in abs_points]

    xs, ys = zip(*rotated_abs)
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    new_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    new_local = [(px - x_min, py - y_min) for px, py in rotated_abs]

    return new_bbox, new_local


def rotated_bbox_to_axis_aligned(
    corner_points: List[Tuple[float, float]], inside_points: List[Tuple[float, float]]
) -> Tuple[
    Tuple[float, float, float, float],  # new_bbox
    List[Tuple[float, float]],  # local_inside_points
    List[Tuple[float, float]],  # local_corners
]:
    """
    Convert rotated shape defined by corners into an axis-aligned bbox.
    """
    xs, ys = zip(*corner_points)
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)

    new_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    local_inside_points = [(x - x_min, y - y_min) for x, y in inside_points]
    local_corners = [(x - x_min, y - y_min) for x, y in corner_points]

    return new_bbox, local_inside_points, local_corners


def snap_points(
    points: List[Tuple[float, float]], mode: SnapMode = SnapMode.ROUND
) -> List[Tuple[int, int]]:
    """
    Snap a list of float (x, y) points to integer coordinates.

    Args:
        points: List of (x, y) float coordinates
        mode: Rounding strategy: "round", "floor", or "ceil"

    Returns:
        List of (x, y) integer coordinates
    """
    if mode == SnapMode.ROUND:
        return [(round(x), round(y)) for x, y in points]
    elif mode == SnapMode.FLOOR:
        return [(int(x), int(y)) for x, y in points]
    elif mode == SnapMode.CEIL:
        from math import ceil

        return [(ceil(x), ceil(y)) for x, y in points]
    else:
        raise ValueError(f"Unknown snap mode: {mode}")


def snap_bbox(
    bbox: tuple[float, float, float, float],
    mode: SnapMode = SnapMode.ROUND,
) -> tuple[int, int, int, int]:
    """
    Snap (x, y, w, h) float bbox to integer pixels using rounding strategy.

    Args:
        bbox: (x, y, w, h) with float values
        mode: "round" (default), "floor", or "ceil"

    Returns:
        bbox as (x, y, w, h) with int values
    """
    x, y, w, h = bbox
    if mode == SnapMode.ROUND:
        return round(x), round(y), round(w), round(h)
    elif mode == SnapMode.FLOOR:
        return int(x), int(y), int(w), int(h)
    elif mode == SnapMode.CEIL:
        from math import ceil

        return ceil(x), ceil(y), ceil(w), ceil(h)
    else:
        raise ValueError(f"Unknown snap mode: {mode}")
