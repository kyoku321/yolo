"""Intel RealSense RGB-D capture for the eye-in-hand stack.

`pyrealsense2` is a HEAVY, optional dependency (Linux wheels; on Jetson it
ships with librealsense), so it is imported lazily inside RealSenseCamera —
transforms/pose3d/calibration and the web demo work without it.

Frame carries depth ALREADY in metres (float32) and aligned to the color
image, so every downstream consumer (pose3d, rgbd_live) is camera-agnostic.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class Frame:
    """One aligned RGB-D sample."""
    rgb: np.ndarray          # (H, W, 3) uint8, RGB (PIL-ready)
    depth_m: np.ndarray      # (H, W) float32, metres, aligned to rgb, 0=invalid
    K: np.ndarray            # (3,3) intrinsics of the color stream
    ts: float = field(default_factory=time.monotonic)


class RealSenseCamera:
    """Context-manager wrapper around pyrealsense2 (aligned depth->color)."""

    def __init__(self, width: int | None = None, height: int | None = None,
                 fps: int | None = None) -> None:
        self.width = width or config.RS_WIDTH
        self.height = height or config.RS_HEIGHT
        self.fps = fps or config.RS_FPS
        self._pipe = None
        self._align = None
        self._depth_scale = None
        self.K: np.ndarray | None = None     # filled on start()

    def __enter__(self) -> "RealSenseCamera":
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                "pyrealsense2 is not installed. On Linux x86: "
                "`pip install pyrealsense2`; on Jetson use the librealsense "
                "bundled with the NVIDIA SDK."
            ) from e
        self._rs = rs
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height,
                          rs.format.z16, self.fps)
        self._pipe = rs.pipeline()
        profile = self._pipe.start(cfg)
        dev = profile.get_device()
        self._depth_scale = (dev.first_depth_sensor().get_depth_scale()
                             if dev else 0.001)
        self._align = rs.align(rs.stream.color)

        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        return self

    def __exit__(self, *exc) -> None:
        if self._pipe is not None:
            self._pipe.stop()
            self._pipe = None

    def read(self, timeout_ms: int = 5000) -> Frame:
        """Grab one aligned frame; depth converted to metres."""
        if self._pipe is None:
            raise RuntimeError("camera not started - use `with RealSenseCamera()`")
        fs = self._align.process(self._pipe.wait_for_frames(timeout_ms))
        color = np.asanyarray(fs.get_color_frame().get_data())      # BGR
        rgb = color[:, :, ::-1].copy()                              # -> RGB
        raw = np.asanyarray(fs.get_depth_frame().get_data())
        depth_m = raw.astype(np.float32) * np.float32(self._depth_scale)
        return Frame(rgb=rgb, depth_m=depth_m, K=self.K.copy())
