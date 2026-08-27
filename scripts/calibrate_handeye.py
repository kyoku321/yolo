"""Hand-eye calibration tool (eye-in-hand).

Three modes:

  simulate   Self-check the solver with synthetic data - no camera, no arm:
                 python scripts/calibrate_handeye.py simulate

  collect    Real hardware. Needs a printed ChArUco board resting on the
             table (fixed in the arm base frame!) and a RealSense on the
             wrist. Each sample also needs the arm FK at that instant, read
             from --fk-file (your arm bridge writes T_base_ee JSON there:
             {"rotation": [rotvec], "translation": [x,y,z]}):
                 python scripts/calibrate_handeye.py collect \
                     --fk-file data/fk.json

             Press ENTER to capture a sample, type 's' + ENTER to stop.
             Aim for >= 10 poses, with >15 deg rotation differences between
             them (a pure-translation calibration cannot solve rotation).

  solve      Solve T_ee_cam from collected samples and write
             data/handeye.json (path = config.HANDEYE_PATH):
                 python scripts/calibrate_handeye.py solve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shelf_demo import config
from shelf_demo.calibration import (HandEyeCalibrator, fk_json_to_T,
                                    load_handeye, save_handeye)
from shelf_demo.transforms import compose, invert, rotvec_to_R


# ---- ChArUco board pose (collect mode) --------------------------------------

def make_board(sx: int = 5, sy: int = 7, square_m: float = 0.04,
               marker_m: float = 0.02):
    """CharucoBoard + detector for the default print (5x7, 40mm squares)."""
    import cv2
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((sx, sy), square_m, marker_m, dic)
    return board, cv2.aruco.CharucoDetector(board)


def board_pose(rgb: np.ndarray, K: np.ndarray, board, detector):
    """ChArUco pose from an RGB image -> 4x4 T_cam_board, or None."""
    import cv2
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    out = detector.detectBoard(gray)
    charuco_corners, charuco_ids = out[0], out[1]
    if charuco_ids is None or len(charuco_ids) < 6:
        return None
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None)
    if not ok:
        return None
    return compose(rvec.reshape(3), tvec.reshape(3))


# ---- modes -------------------------------------------------------------------

def cmd_simulate(args) -> int:
    rng = np.random.default_rng(args.seed)
    T_ee_cam_gt = compose([0.15, -0.20, 0.10], [0.05, 0.02, 0.10])
    T_base_board = compose([0.0, 0.0, 0.05], [0.40, -0.10, 0.35])

    cal = HandEyeCalibrator()
    for _ in range(args.samples):
        T_base_ee = compose(rng.normal(size=3) * 0.8,
                            rng.uniform([-0.2, -0.2, 0.2],
                                        [0.6, 0.3, 0.7]))
        T_base_cam = T_base_ee @ T_ee_cam_gt
        T_cam_board = invert(T_base_cam) @ T_base_board
        # measurement noise: rotation rotvec jitter + translation jitter
        r_noisy = (rng.normal(size=3) * args.noise_rad
                   + _rotvec_of(T_cam_board))
        t_noisy = T_cam_board[:3, 3] + rng.normal(size=3) * args.noise_m
        cal.add_sample(T_base_ee, compose(r_noisy, t_noisy))

    X, residual = cal.solve("park")
    T_err = float(np.linalg.norm(X[:3, 3] - T_ee_cam_gt[:3, 3]))
    from shelf_demo.transforms import R_to_rotvec
    dR = np.linalg.norm(R_to_rotvec(
        X[:3, :3] @ T_ee_cam_gt[:3, :3].T)) * 180.0 / np.pi
    print(f"samples={args.samples}  noise(rad,m)=({args.noise_rad},"
          f"{args.noise_m})")
    print(f"AX=XB residual : {residual:.6f}")
    print(f"solved vs GT   : rotation error {dR:.3f} deg, "
          f"translation error {T_err * 1000:.2f} mm")
    ok = dR < 0.5 and T_err < 0.002 and residual < 0.02
    print("ALL PASS" if ok else "SOLVER ERROR TOO LARGE")
    return 0 if ok else 1


def _rotvec_of(T: np.ndarray) -> np.ndarray:
    from shelf_demo.transforms import R_to_rotvec
    return R_to_rotvec(T[:3, :3])


def cmd_collect(args) -> int:
    from shelf_demo.camera import RealSenseCamera  # lazy: needs pyrealsense2

    board, detector = make_board(square_m=args.square_m,
                                 marker_m=args.marker_m)
    out_path = Path(args.out)
    cal = (HandEyeCalibrator.load_samples(out_path)
           if args.append and out_path.exists() else HandEyeCalibrator())
    fk_path = Path(args.fk_file)

    print("Move the arm, press ENTER to capture (board must be visible), "
          "'s' + ENTER to stop.")
    with RealSenseCamera() as cam:
        frame = cam.read()
        K = frame.K
        while True:
            cmd = input(f"[{len(cal.samples)} samples] ").strip().lower()
            if cmd == "s":
                break
            if not fk_path.exists():
                print(f"  !! {fk_path} missing - start your arm FK bridge first")
                continue
            T_base_ee = fk_json_to_T(json.loads(fk_path.read_text()))
            frame = cam.read()
            T_cam_board = board_pose(frame.rgb, K, board, detector)
            if T_cam_board is None:
                print("  !! board not found in view - move and retry")
                continue
            n = cal.add_sample(T_base_ee, T_cam_board)
            print(f"  sample {n} recorded")

    if len(cal.samples) >= 3:
        cal.save_samples(out_path)
        print(f"saved {len(cal.samples)} samples to {out_path}")
        print("next: python scripts/calibrate_handeye.py solve "
              f"--samples {out_path}")
    else:
        print(f"only {len(cal.samples)} samples - need >= 3, nothing saved")
    return 0


def cmd_solve(args) -> int:
    cal = HandEyeCalibrator.load_samples(args.samples)
    X, residual = cal.solve("park")
    print(f"{len(cal.samples)} samples, AX=XB residual {residual:.6f}")
    if residual > 0.03:
        print("!! residual high - check sample pose diversity / board size /"
              " FK sync; more samples with bigger rotation gaps help")
    out = Path(args.out or config.HANDEYE_PATH)
    save_handeye(out, X, meta={"samples": len(cal.samples),
                               "residual": residual,
                               "method": "park"})
    print(f"T_ee_cam saved to {out}:")
    print(np.round(X, 5))
    # sanity: reload and display
    load_handeye(out)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    sp = sub.add_parser("simulate", help="solver self-check, no hardware")
    sp.add_argument("--samples", type=int, default=14)
    sp.add_argument("--noise-rad", type=float, default=0.002)
    sp.add_argument("--noise-m", type=float, default=0.001)
    sp.add_argument("--seed", type=int, default=3)
    sp.set_defaults(fn=cmd_simulate)

    sp = sub.add_parser("collect", help="capture sample pairs (real camera)")
    sp.add_argument("--fk-file", default=str(config.DATA_DIR / "fk.json"))
    sp.add_argument("--out", default=str(config.DATA_DIR / "handeye_samples.json"))
    sp.add_argument("--append", action="store_true")
    sp.add_argument("--square-m", type=float, default=0.040)
    sp.add_argument("--marker-m", type=float, default=0.020)
    sp.set_defaults(fn=cmd_collect)

    sp = sub.add_parser("solve", help="solve T_ee_cam from samples")
    sp.add_argument("--samples", default=str(config.DATA_DIR / "handeye_samples.json"))
    sp.add_argument("--out", default="")
    sp.set_defaults(fn=cmd_solve)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
