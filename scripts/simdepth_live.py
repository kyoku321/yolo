"""Eye-in-hand test on macOS WITHOUT a depth camera (synthetic depth plane).

`pyrealsense2` has no macOS wheel, so the real RGB-D loop
(scripts/rs_live.py) can only run on the Linux robot host. This harness
feeds the EXACT same production code path as rs_live.py, but the depth map
is a flat plane at `--depth` metres with synthetic intrinsics (--hfov):

    any UVC/FaceTime RGB camera
        -> YOLO detect -> IoU track -> CLIP match        (real, unchanged)
        -> pose3d deprojection on synthetic depth        (real geometry code)
        -> RGBDGraspSession.grasp3d stability window     (real, unchanged)

It validates recognition + tracking + 3D deprojection + grasp readiness on
real video, and shows 3D coordinates (metres, camera frame) in the overlay
window and on stdout — the thing the web Live tab cannot show (the browser
sends JPEGs, no depth).

macOS note: the TERMINAL app needs Camera permission (System Settings ->
Privacy & Security -> Camera), or cv2.VideoCapture will refuse to open.

Examples:
    python scripts/simdepth_live.py --target "KIRIN LAGER" --depth 0.35
    python scripts/simdepth_live.py --camera 1 --hfov 87 --save-dir data/simcap
    python scripts/simdepth_live.py --no-window      # headless, JSON lines

Keys (window mode): q = quit, s = save annotated frame (with --save-dir).
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from shelf_demo import config
from shelf_demo.camera import Frame
from shelf_demo.draw import _font, draw_recognitions
from shelf_demo.pose3d import target_point
from shelf_demo.rgbd_live import RGBDGraspSession


def synthetic_K(w: int, h: int, hfov_deg: float) -> np.ndarray:
    fx = fy = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0.0, w / 2.0],
                     [0.0, fy, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def catalog_names() -> list[str]:
    con = sqlite3.connect(config.DB_PATH)
    rows = con.execute("SELECT name FROM products ORDER BY sku_id")
    return [r[0] for r in rows]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0,
                   help="OpenCV device index (macOS avfoundation order)")
    p.add_argument("--target", default="",
                   help="product name to 3D-locate (grasp3d)")
    p.add_argument("--depth", type=float, default=0.35,
                   help="synthetic depth plane distance in metres")
    p.add_argument("--hfov", type=float, default=87.0,
                   help="horizontal FOV (deg) for synthetic intrinsics")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--save-dir", default="",
                   help="dump annotated frames here (s key or every 30th)")
    p.add_argument("--no-window", action="store_true",
                   help="headless: no imshow, JSON lines only")
    args = p.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}.")
        print("macOS: grant Camera permission to your terminal app "
              "(System Settings -> Privacy & Security -> Camera), then retry.")
        return 1

    names = catalog_names()
    print(f"camera {args.camera} open; catalog names: {names}")
    if args.target and args.target not in names:
        print(f"WARNING: target {args.target!r} not in catalog (names above)")

    print("loading models ...")
    from shelf_demo.pipeline import ShelfPipeline
    session = RGBDGraspSession(ShelfPipeline(), imgsz=args.imgsz)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    K = synthetic_K(W, H, args.hfov)
    depth = np.full((H, W), args.depth, np.float32)
    print(f"frame {W}x{H}  synthetic K fx=fy={K[0,0]:.1f}  "
          f"depth plane={args.depth} m  hfov={args.hfov} deg")
    print("pointing at the shelf - q to quit")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    font = _font(16)
    last_print = 0.0
    n = 0
    t0 = time.monotonic()
    while True:
        ok, bgr = cap.read()
        if not ok:
            print("camera read failed - stopping")
            break
        if bgr.shape[:2] != (H, W):      # size changed: rebuild geometry
            H, W = bgr.shape[:2]
            K = synthetic_K(W, H, args.hfov)
            depth = np.full((H, W), args.depth, np.float32)

        rgb = bgr[:, :, ::-1]
        result = session.process_frame(
            Frame(rgb=rgb, depth_m=depth, K=K, ts=time.monotonic()))

        canvas = draw_recognitions(Image.fromarray(rgb),
                                   result.recognitions)
        draw = ImageDraw.Draw(canvas)
        for r in result.recognitions:
            if not r.is_match:
                continue
            tgt = target_point(r.box, depth, K)
            if tgt is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in r.box.xyxy)
            txt = (f"cam(x={tgt.point_cam[0]:+.3f} "
                   f"y={tgt.point_cam[1]:+.3f} z={tgt.z_m:.3f} m)")
            draw.text((x1, y2 + 2), txt, fill=(255, 255, 0), font=font)

        st = None
        if args.target:
            st = session.grasp3d(args.target)
            msg = (f"GRASP READY {args.target}  "
                   f"cam={st['point_cam_m']}  base={st['point_base_m']}  "
                   f"valid={st['valid_px']}px") if st["ready"] else \
                  f"target: {st.get('reason', '?')}"
            color = (0, 255, 0) if st["ready"] else (0, 160, 255)
            draw.rectangle([0, 0, W - 1, 26], fill=(40, 40, 40))
            draw.text((8, 5), msg, fill=color, font=font)

        bgr_out = np.array(canvas)[:, :, ::-1]
        if not args.no_window:
            cv2.imshow("simdepth eye-in-hand test (q: quit, s: save)", bgr_out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and save_dir:
                fn = save_dir / f"sim_{int(time.time())}_{n:05d}.jpg"
                cv2.imwrite(str(fn), bgr_out)
                print(f"saved {fn}")
        if save_dir and n % 30 == 0:
            fn = save_dir / f"sim_{int(time.time())}_{n:05d}.jpg"
            cv2.imwrite(str(fn), bgr_out)

        now = time.monotonic()
        if args.target and st is not None and now - last_print >= 1.0:
            last_print = now
            print(json.dumps({
                "t": round(now - t0, 1),
                "ready": st["ready"],
                "reason": st.get("reason"),
                "box": st.get("box"),
                "point_cam_m": st.get("point_cam_m"),
                "point_base_m": st.get("point_base_m"),
                "z_m": st.get("z_m"),
                "valid_px": st.get("valid_px"),
                "frames_stable": st.get("frames_stable"),
            }, ensure_ascii=False), flush=True)
        n += 1

    cap.release()
    if not args.no_window:
        cv2.destroyAllWindows()
    print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
