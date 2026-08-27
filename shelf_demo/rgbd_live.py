"""RGB-D grasp session: 2D recognition (live.py) + 3D localisation (pose3d).

This is the on-board counterpart of the Live tab's /live/api/grasp route:
same stability logic, but fed by RealSense Frames on the robot instead of
browser JPEGs, and the grasp answer comes back in METRES:

    point_cam_m   target in the camera optical frame (always available)
    point_base_m  target in the ARM BASE frame via fk-at-capture-time and
                  the calibrated T_ee_cam  (needs calibration + arm FK)

Key design point: the arm FK and the frame form ONE sample — always pass
fk=arm.fk() taken as close to the camera read as possible. During the later
servo loop the arm is (near-)still when photographing, so a plain read is
fine; hardware-timestamp alignment is a follow-up (see design doc, risks).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import live as L
from .camera import Frame
from .pipeline import ShelfPipeline
from .pose3d import target_point
from .robot import ArmBase
from .transforms import transform_points


@dataclass
class _FrameContext:
    """What grasp3d() needs to freeze per accepted frame."""
    depth_m: np.ndarray
    K: np.ndarray
    ts: float
    T_base_ee: np.ndarray | None   # None if the arm was not queried/known


class RGBDGraspSession:
    def __init__(self, pipeline: ShelfPipeline, arm: ArmBase | None = None,
                 T_ee_cam: np.ndarray | None = None, **live_kwargs) -> None:
        self.rec = L.LiveRecognizer(pipeline, **live_kwargs)
        self.arm = arm
        self.T_ee_cam = T_ee_cam
        self._ctx: _FrameContext | None = None

    # ---- perception --------------------------------------------------------
    def process_frame(self, frame: Frame,
                      fk=None) -> L.LiveResult:
        """Run one RGB-D frame through recognition.

        fk: 4x4 T_base_ee captured at frame time; falls back to self.arm.fk()
        (which is the same thing for an ArmBase), else None (2D-only mode).
        """
        if fk is None and self.arm is not None:
            fk = self.arm.fk()
        self._ctx = _FrameContext(
            depth_m=frame.depth_m, K=frame.K, ts=frame.ts,
            T_base_ee=None if fk is None else np.asarray(fk, dtype=float))
        image = Image.fromarray(frame.rgb)
        return self.rec.process(image)

    # ---- the 3D grasp answer -----------------------------------------------
    def grasp3d(self, name: str) -> dict:
        """Extend LiveRecognizer.grasp_status() with metric 3D coordinates."""
        out = self.rec.grasp_status(name)
        ctx = self._ctx
        out.update({"units": "m",
                    "point_cam_m": None, "point_base_m": None,
                    "z_m": None, "valid_px": 0})
        if not out.get("ready"):
            return out
        if ctx is None:
            out["ready"] = False
            out["reason"] = "no RGB-D frame processed yet"
            return out
        frame_age_ms = (time.monotonic() - ctx.ts) * 1000.0
        if frame_age_ms > L.GRASP_MAX_AGE_MS:
            out["ready"] = False
            out["reason"] = (f"last RGB-D frame is {frame_age_ms:.0f} ms old - "
                             "robot loop not running?")
            return out

        from .detector import Box  # local import: detector pulls config only
        x1, y1, x2, y2 = out["box"]
        tgt = target_point(Box(x1, y1, x2, y2, 1.0), ctx.depth_m, ctx.K)
        if tgt is None:
            out["ready"] = False
            out["reason"] = ("depth unusable on the stable box "
                             "(too few valid pixels / out of range)")
            return out
        out["z_m"] = round(tgt.z_m, 4)
        out["valid_px"] = tgt.valid_px
        p_cam = tgt.point_cam
        out["point_cam_m"] = [round(float(v), 4) for v in p_cam]

        if self.T_ee_cam is not None and ctx.T_base_ee is not None:
            T = ctx.T_base_ee @ np.asarray(self.T_ee_cam, dtype=float)
            out["point_base_m"] = [round(float(v), 4)
                                   for v in transform_points(T, p_cam)]
        else:
            out["reason"] = ("3D OK in camera frame; arm-base frame needs "
                             "hand-eye calibration + live FK")
        return out
