"""Intel RealSense RGB-D capture for the eye-in-hand stack.

`pyrealsense2` is a HEAVY, optional dependency (Linux wheels; on Jetson it
ships with librealsense), so it is imported lazily inside RealSenseCamera —
transforms/pose3d/calibration and the web demo work without it.
On macOS install the community wheel `pip install pyrealsense2-macosx`;
SDK 2.50+ often needs `sudo` to access the device there.

Frame carries depth ALREADY in metres (float32) and aligned to the color
image, so every downstream consumer (pose3d, rgbd_live) is camera-agnostic.

The pipeline is started with a warm-up read and retries up to 3 times:
the first start after (re)plug frequently starts but delivers no frames
(macOS USB quirk observed on D435i). read() restarts the pipeline once on
a mid-run frame loss, so a USB hiccup does not kill the servo loop.
"""
from __future__ import annotations

import sys
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
        self._pipe = rs.pipeline()
        self._align = rs.align(rs.stream.color)
        self._start_pipeline()
        return self

    def _start_pipeline(self, tries: int = 3) -> None:
        """Start the pipeline and verify frames actually flow (warm-up).

        A FRESH pipeline object is created per attempt and a failed attempt
        is abandoned (GC releases it) — calling stop() on a pipeline that
        started but never delivered frames segfaults the process on macOS
        (librealsense 2.54.x, D435i observed).
        """
        rs = self._rs
        last_err = ""
        for attempt in range(1, tries + 1):
            pipe = rs.pipeline()
            profile = None
            self._pipe = None
            try:
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, self.width, self.height,
                                  rs.format.bgr8, self.fps)
                cfg.enable_stream(rs.stream.depth, self.width, self.height,
                                  rs.format.z16, self.fps)
                profile = pipe.start(cfg)
                fs = self._align.process(pipe.wait_for_frames(5000))
                if (fs.get_color_frame() is not None
                        and fs.get_depth_frame() is not None):
                    self._pipe = pipe       # commit the working pipeline
                    break
                last_err = "warm-up frame missing color/depth"
            except RuntimeError as e:
                last_err = str(e)
            # Do NOT stop a pipeline that never delivered frames: on macOS
            # that segfaults. Abandon the object (reassigned next attempt)
            # and start a fresh one.
            pipe = None
            if attempt < tries:
                time.sleep(0.5)
        if self._pipe is None:
            raise RuntimeError(
                f"RealSense pipeline gave up after {tries} attempts "
                f"(last: {last_err}); try unplug+replug or another direct "
                "USB 3 port")
        if attempt > 1:
            print(f"note: RealSense pipeline warm after attempt {attempt}",
                  file=sys.stderr)
        dev = profile.get_device()
        self._depth_scale = (dev.first_depth_sensor().get_depth_scale()
                             if dev else 0.001)
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        self.K = np.array([[intr.fx, 0.0, intr.ppx],
                           [0.0, intr.fy, intr.ppy],
                           [0.0, 0.0, 1.0]], dtype=np.float64)

    def __exit__(self, *exc) -> None:
        # Only stop a pipeline that was committed (delivered frames); that
        # stop is safe. A never-started pipeline object is left to GC.
        if self._pipe is not None:
            try:
                self._pipe.stop()
            except Exception:   # noqa: BLE001 - teardown must never raise
                pass
            self._pipe = None

    def read(self, timeout_ms: int = 5000) -> Frame:
        """Grab one aligned frame; depth converted to metres."""
        if self._pipe is None:
            raise RuntimeError("camera not started - use `with RealSenseCamera()`")
        try:
            fs = self._align.process(self._pipe.wait_for_frames(timeout_ms))
        except RuntimeError:
            # Mid-run frame loss (USB hiccup): abandon this pipeline object
            # (never stop() it: a stalled pipeline segfaults on macOS)
            # and start a fresh one, then retry this read.
            print("note: RealSense frame loss - restarting pipeline",
                  file=sys.stderr)
            self._pipe = None
            self._start_pipeline()
            fs = self._align.process(self._pipe.wait_for_frames(timeout_ms))
        color = np.asanyarray(fs.get_color_frame().get_data())      # BGR
        rgb = color[:, :, ::-1].copy()                              # -> RGB
        raw = np.asanyarray(fs.get_depth_frame().get_data())
        depth_m = raw.astype(np.float32) * np.float32(self._depth_scale)
        return Frame(rgb=rgb, depth_m=depth_m, K=self.K.copy())
