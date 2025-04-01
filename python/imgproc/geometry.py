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


def get_random_scale_and_position(bg_size, obj_size, scale_range=(0.4, 0.9)):
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


def rotate_point(x, y, cx, cy, angle_rad):
    """Rotate a point (x, y) around center (cx, cy) by angle (radians)."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x -= cx
    y -= cy
    x_new = cos_a * x - sin_a * y + cx
    y_new = sin_a * x + cos_a * y + cy
    return x_new, y_new


def transform_points_in_rotated_bbox(bbox, points, angle_deg):
    """
    Rotate a bounding box and map local coordinates to new local coordinates
    inside the rotated (axis-aligned) bounding box.

    Parameters:
        bbox (tuple): Original bounding box in absolute coordinates (x, y, w, h)
        points (list of tuples): Points inside the original bbox, in local coordinates (0,0)-(w,h)
        angle_deg (float): Rotation angle in degrees (counterclockwise)

    Returns:
        new_bbox (tuple): Axis-aligned bounding box that encloses the rotated bbox (x, y, w, h)
        new_local_points (list of tuples): Points transformed into the local coordinates of new_bbox
    """
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    angle_rad = math.radians(angle_deg)

    # Convert local points to absolute coordinates
    abs_points = [(x + px, y + py) for (px, py) in points]

    # Rotate all points around the bbox center
    rotated_abs = [rotate_point(px, py, cx, cy, angle_rad) for px, py in abs_points]

    # Compute new axis-aligned bounding box
    xs, ys = zip(*rotated_abs)
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)
    new_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    # Convert rotated absolute points to local coordinates in new bbox
    new_local = [(px - x_min, py - y_min) for px, py in rotated_abs]

    return new_bbox, new_local


def rotated_bbox_to_axis_aligned(corner_points, inside_points):
    """
    Convert rotated shape defined by corners into an axis-aligned bbox.

    Parameters:
        corner_points (list of (x, y)): Corners of the rotated rectangle (in order)
        inside_points (list of (x, y)): Any additional points (e.g., center, landmarks)

    Returns:
        new_bbox (x, y, w, h): Axis-aligned bbox enclosing the corners
        local_inside_points (list): Inside points in local coords (relative to new_bbox)
        local_corners (list): Corner points in local coords (relative to new_bbox)
    """
    xs, ys = zip(*corner_points)
    x_min, y_min = min(xs), min(ys)
    x_max, y_max = max(xs), max(ys)

    new_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    local_inside_points = [(x - x_min, y - y_min) for x, y in inside_points]
    local_corners = [(x - x_min, y - y_min) for x, y in corner_points]

    return new_bbox, local_inside_points, local_corners
