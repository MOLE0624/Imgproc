#!/usr/bin/env python3

# ================================================================================
# File       : test_geometry.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Visual test for rotated bounding box transformation and coordinate
#              mapping
# Date       : 2025-04-01
# ================================================================================

import math
import unittest

import matplotlib.pyplot as plt

from imgproc.geometry import (rotate_point, rotated_bbox_to_axis_aligned,
                              transform_points_in_rotated_bbox)


class TestGeometryVisual(unittest.TestCase):
    def test_visualize_rotated_bbox_transform(self):
        bbox = (100, 100, 100, 100)
        points = [
            (0, 0),
            (100, 0),
            (100, 100),
            (0, 100),
            (50, 50),
            (30, 70),
            (80, 80),
            (12, 24),
        ]
        angle_deg = 30
        angle_rad = math.radians(angle_deg)

        x, y, w, h = bbox
        cx = x + w / 2
        cy = y + h / 2

        # Convert local bbox points to absolute positions
        abs_points = [(x + px, y + py) for px, py in points]

        # Rotate all points around the bbox center
        expected_rotated_abs = [
            rotate_point(px, py, cx, cy, angle_rad) for px, py in abs_points
        ]

        # Transform using the main geometry function
        new_bbox, new_local_points = transform_points_in_rotated_bbox(
            bbox, points, angle_deg
        )
        x_new, y_new, _, _ = new_bbox
        recovered_abs = [(x_new + px, y_new + py) for (px, py) in new_local_points]

        # Validate that the recovered coordinates closely match the expected rotated ones
        for i, (exp, rec) in enumerate(zip(expected_rotated_abs, recovered_abs)):
            with self.subTest(i=i):
                self.assertAlmostEqual(exp[0], rec[0], delta=1.0)
                self.assertAlmostEqual(exp[1], rec[1], delta=1.0)

        # === Visualization ===
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"Rotated BBox Transform (Angle: {angle_deg}°)")
        ax.set_aspect("equal")
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        ax.invert_yaxis()

        # Draw the original axis-aligned bbox (gray dashed)
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                w,
                h,
                fill=False,
                color="gray",
                linestyle="--",
                label="Original BBox",
            )
        )

        # Plot original (pre-rotation) points (gray dots)
        xs0, ys0 = zip(*abs_points)
        ax.plot(xs0, ys0, "o", color="gray", label="Original Points")

        # Define the 4 corners of the original bbox, then rotate them
        corners_local = [(0, 0), (w, 0), (w, h), (0, h)]
        rotated_corners = [
            rotate_point(x + px, y + py, cx, cy, angle_rad) for px, py in corners_local
        ]

        # Draw the rotated bbox outline (orange)
        for i in range(4):
            x0, y0 = rotated_corners[i]
            x1, y1 = rotated_corners[(i + 1) % 4]
            ax.plot([x0, x1], [y0, y1], color="orange", linewidth=1.5)
        ax.plot([], [], color="orange", label="Rotated BBox")  # dummy for legend

        # Plot expected rotated points (blue dots)
        xs1, ys1 = zip(*expected_rotated_abs)
        ax.plot(xs1, ys1, "bo", label="Expected Rotated Points")

        # Plot recovered transformed points (red X)
        xs2, ys2 = zip(*recovered_abs)
        ax.plot(xs2, ys2, "rx", label="Recovered Points")

        # Compute the axis-aligned bbox from rotated corners
        rotated_bbox_from_points, local_inner_pts, local_corners = (
            rotated_bbox_to_axis_aligned(
                corner_points=rotated_corners, inside_points=recovered_abs
            )
        )

        rx, ry, rw, rh = rotated_bbox_from_points
        ax.add_patch(
            plt.Rectangle(
                (rx, ry),
                rw,
                rh,
                fill=False,
                edgecolor="green",
                linestyle="--",
                label="Axis-aligned from Rotated",
            )
        )

        # Plot inner points (from recovered_abs) as local → absolute (cyan dots)
        abs_from_local_inner = [(rx + px, ry + py) for (px, py) in local_inner_pts]
        xs3, ys3 = zip(*abs_from_local_inner)
        ax.plot(xs3, ys3, "c.", label="Recovered Points in New BBox")

        # Plot rotated corner points in new bbox (cyan triangles)
        abs_from_local_corners = [(rx + px, ry + py) for (px, py) in local_corners]
        xs4, ys4 = zip(*abs_from_local_corners)
        ax.plot(xs4, ys4, "c^", label="Rotated Corners in New BBox")

        # Label indices on recovered (red) points
        for i, (px, py) in enumerate(zip(xs2, ys2)):
            ax.text(px + 2, py - 2, f"{i}", fontsize=8)

        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main()
