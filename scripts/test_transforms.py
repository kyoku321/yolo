"""Unit tests for SE3 math (shelf_demo/transforms.py) - no models, no HW.

Covers:
  1. rotvec <-> rotation matrix round-trip
  2. compose/invert: T_inv @ T == I
  3. transform_points: single point and batch
  4. a known rotation (90 deg about z) behaves as expected
  5. the eye-in-hand chain P_base = T_base_ee @ T_ee_cam @ P_cam inverts back
  6. target_in_base matches the manual chain

Run:  python scripts/test_transforms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shelf_demo import transforms as T


def main() -> int:
    ok = True

    def check(label: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(("PASS " if cond else "FAIL ") + label
              + ("" if cond else f"  [{extra}]"))

    rng = np.random.default_rng(7)

    # 1. rotvec round-trip ---------------------------------------------------
    for _ in range(10):
        # rotation vector with angle < pi (R_to_rotvec returns the shortest
        # representative, so wider angles come back angle-flipped by design)
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * rng.uniform(0.01, np.pi - 1e-3)
        v2 = T.R_to_rotvec(T.rotvec_to_R(v))
        check(f"rotvec round-trip {v.round(2)}",
              np.linalg.norm(v - v2) < 1e-9, str(v - v2))

    # 2. compose / invert ----------------------------------------------------
    T_ab = T.compose(rng.normal(size=3), rng.normal(size=3) - 0.5)
    I = T.invert(T_ab) @ T_ab
    check("invert(T) @ T == I", np.max(np.abs(I - np.eye(4))) < 1e-10)

    # 3. transform_points single & batch ------------------------------------
    pts = rng.normal(size=(5, 3))
    one = T.transform_points(T_ab, pts[2])
    batch = T.transform_points(T_ab, pts)
    check("transform_points batch matches single", np.allclose(batch[2], one))
    check("transform_points batch shape", batch.shape == (5, 3))

    # 4. known rotation: +90deg about z maps x-axis -> y-axis ---------------
    Rz = T.rotvec_to_R([0, 0, np.pi / 2])
    x = Rz @ np.array([1.0, 0.0, 0.0])
    check("Rz(+90) maps [1,0,0] -> [0,1,0]",
          np.allclose(x, [0, 1, 0], atol=1e-12), str(x))

    # 5+6. eye-in-hand chain round trip --------------------------------------
    T_ee_cam = T.compose([0.1, -0.5, 0.3], [0.06, 0.02, 0.12])   # wrist cam
    T_base_ee = T.compose([0.4, 0.2, -0.1], [0.35, -0.15, 0.6])  # fk
    p_cam = np.array([0.12, -0.03, 0.62])                        # from depth
    p_base = T.target_in_base(T_base_ee, T_ee_cam, p_cam)
    p_base_manual = T.transform_points(T_base_ee @ T_ee_cam, p_cam)
    check("target_in_base == manual chain", np.allclose(p_base, p_base_manual))
    p_cam_back = T.transform_points(
        T.invert(T_ee_cam) @ T.invert(T_base_ee), p_base)
    check("chain round-trip back to camera frame",
          np.allclose(p_cam_back, p_cam, atol=1e-12), str(p_cam_back - p_cam))

    print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
