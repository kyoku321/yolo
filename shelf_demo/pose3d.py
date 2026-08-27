"""2D box + aligned depth map + intrinsics -> 3D target point (camera frame).

The robust part is depth selection: a YOLO box contains background and the
product's rounded corners, both of which poison a naive "centre pixel" read.
We shrink the box toward its centre (TARGET3D_ROI_SHRINK), drop zero and
beyond-range depths, and take the MEDIAN of what remains. That median sits
on the product's front face for boxes/bottles on a shelf.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config
from .detector import Box

MIN_VALID_PX = 8     # fewer valid depth pixels than this => refuse to 3D-locate


@dataclass
class Target3D:
    """One product localised in the camera frame (metres)."""
    point_cam: np.ndarray      # (3,) xyz, metres, camera optical frame
    z_m: float                 # median depth used
    valid_px: int              # how many depth pixels voted (quality metric)
    box: Box


def deproject(px: float, py: float, z: float, K) -> np.ndarray:
    """Back-project pixel (px, py) at depth z (metres) into the camera frame.

    K is the 3x3 intrinsics matrix of the ALIGNED (color) image.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.array([(px - cx) * z / fx, (py - cy) * z / fy, z])


def roi_median_depth(depth_m: np.ndarray, box: Box,
                     shrink: float | None = None,
                     max_m: float | None = None) -> tuple[float, int]:
    """Median valid depth (metres, count) inside the shrunk-on-centre ROI.

    Invalid = 0 (no depth) or > max_m. Returns (nan, 0) when there is
    nothing usable.
    """
    shrink = config.TARGET3D_ROI_SHRINK if shrink is None else float(shrink)
    max_m = config.RS_DEPTH_M_MAX if max_m is None else float(max_m)
    h, w = depth_m.shape
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    mx, my = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
    hw, hh = bw * shrink / 2.0, bh * shrink / 2.0
    x1 = int(np.clip(round(mx - hw), 0, w - 1))
    x2 = int(np.clip(round(mx + hw), 0, w - 1))
    y1 = int(np.clip(round(my - hh), 0, h - 1))
    y2 = int(np.clip(round(my + hh), 0, h - 1))
    if x2 < x1 or y2 < y1:
        return float("nan"), 0
    roi = depth_m[y1:y2 + 1, x1:x2 + 1]
    valid = roi[(roi > 0.0) & (roi <= max_m)]
    if valid.size == 0:
        return float("nan"), 0
    return float(np.median(valid)), int(valid.size)


def target_point(box: Box, depth_m: np.ndarray, K) -> Target3D | None:
    """3D location of one detected product, or None if depth is unusable."""
    z, n = roi_median_depth(depth_m, box)
    if not (z == z) or n < MIN_VALID_PX:   # NaN check
        return None
    cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
    return Target3D(point_cam=deproject(cx, cy, z, K),
                    z_m=z, valid_px=n, box=box)
