"""End-to-end test of the RGB-D grasp chain - synthetic data, no HW/models.

Builds a virtual scene (a flat 0.6 m wall with one "Cola" box), runs the
REAL GraspSession chain (LiveRecognizer stability window -> pose3d target ->
eye-in-hand transform) and checks the metric answer:

  P_cam  = deproject(box centre, median depth 0.6)
  P_base = T_base_ee @ T_ee_cam @ P_cam   (both transforms set by hand)

Also covers the failure paths: no depth on the box, unknown name, stale
frame, and that the arm FK was queried once per frame (not cached).

Run:  python scripts/test_rgbd_grasp.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shelf_demo import config, live as L
from shelf_demo.camera import Frame
from shelf_demo.detector import Box
from shelf_demo.pipeline import Recognition
from shelf_demo.rgbd_live import RGBDGraspSession
from shelf_demo.robot import MockArm
from shelf_demo import transforms as T

# Virtual camera: 640x480, fx=fy=600, cx=320,cy=240; wall at z = 0.6 m
W, H, Z_WALL = 640, 480, 0.6
K = np.array([[600.0, 0.0, 320.0],
              [0.0, 600.0, 240.0],
              [0.0, 0.0, 1.0]])
BOX = (100, 100, 200, 200)       # the Cola facing, in pixels
CX, CY = 150.0, 150.0            # its centre

# Expected geometry (derived by hand):
#   p_cam  = ((cx-px)*z/fx, (cy-py)*z/fy, z) = (-0.17, -0.09, 0.6)
EXP_P_CAM = np.array([-0.17, -0.09, Z_WALL])
T_EE_CAM = np.eye(4)                                     # hand-eye: identity
EE_POS = np.array([0.3, 0.0, 0.5])
T_BASE_EE = T.compose([0, 0, 0], EE_POS)                 # fk: translated
EXP_P_BASE = EXP_P_CAM + EE_POS


class _FakeDetector:
    def detect(self, image, conf=None, imgsz=None):
        return [Box(*BOX, 0.95)]


class _FakePipeline:
    def __init__(self):
        self.detector = _FakeDetector()

    def match_boxes(self, image, boxes, threshold=None):
        return [Recognition(b, 1, "Cola", 0.9, True) for b in boxes]


def make_frame(depth_m: float | None = 0.6) -> Frame:
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = (np.full((H, W), depth_m, dtype=np.float32)
             if depth_m is not None
             else np.zeros((H, W), dtype=np.float32))
    return Frame(rgb=rgb, depth_m=depth, K=K.copy())


def main() -> int:
    ok = True

    def check(label: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label
              + ("" if cond else f"  [{extra}]"))

    arm = MockArm(T_BASE_EE)
    sess = RGBDGraspSession(_FakePipeline(), arm=arm, T_ee_cam=T_EE_CAM)

    # CONFIRM_FRAMES frames to confirm the SKU, then GRASP_WINDOW stable
    # frames; +2 of margin above theory
    n_feed = config.CONFIRM_FRAMES + L.GRASP_WINDOW + 2
    for _ in range(n_feed):
        sess.process_frame(make_frame(0.6))

    res = sess.grasp3d("Cola")
    check("grasp3d ready after confirm + stable frames",
          res["ready"] is True, res.get("reason", ""))
    check("arm FK queried exactly once per frame (not cached)",
          arm.fk_calls == n_feed, str(arm.fk_calls))

    p_cam = np.array(res["point_cam_m"] or [9, 9, 9])
    check("point_cam matches deprojection (+-1 cm)",
          np.allclose(p_cam, EXP_P_CAM, atol=0.01),
          f"{p_cam} vs {EXP_P_CAM}")
    p_base = np.array(res["point_base_m"] or [9, 9, 9])
    check("point_base = fk @ cam (+-1 cm)",
          np.allclose(p_base, EXP_P_BASE, atol=0.01),
          f"{p_base} vs {EXP_P_BASE}")
    check("z_m quality fields filled",
          res["z_m"] == 0.6 and res["valid_px"] > 0,
          str(res))

    # unknown product --------------------------------------------------------
    res = sess.grasp3d("Sprite")
    check("unknown name not ready", res["ready"] is False)

    # dead depth (all zeros) -------------------------------------------------
    sess2 = RGBDGraspSession(_FakePipeline(), arm=MockArm(T_BASE_EE),
                             T_ee_cam=T_EE_CAM)
    for _ in range(n_feed):
        sess2.process_frame(make_frame(None))
    res = sess2.grasp3d("Cola")
    check("zero-depth frame not graspable", res["ready"] is False)
    check("  reason mentions depth", "depth" in res.get("reason", ""),
          res.get("reason", ""))

    # stale frame ------------------------------------------------------------
    sess._ctx.ts = time.monotonic() - 10.0     # last frame 10 s ago
    res = sess.grasp3d("Cola")
    check("stale frame not graspable", res["ready"] is False,
          res.get("reason", ""))

    # no calibration / no fk: camera-frame answer still works -----------------
    sess3 = RGBDGraspSession(_FakePipeline())   # no arm, no T_ee_cam
    for _ in range(n_feed):
        sess3.process_frame(make_frame(0.6))
    res = sess3.grasp3d("Cola")
    check("no-calib mode: point_cam OK, point_base None",
          res["ready"] is True and res["point_base_m"] is None)

    print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
