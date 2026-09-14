"""Quick RealSense health check: enumerate, stream, verify depth.

Run this FIRST whenever the D435i behaves oddly. On macOS the camera must
not be held by another client (browser Live tab, Teams, Photo Booth...) —
the SDK then fails with "No device connected" / "failed to set power
state".

    python scripts/rs_check.py            # 1280x720, 15 frames, saves images
    python scripts/rs_check.py --no-save
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=15)
    p.add_argument("--cv2-warmup", action="store_true",
                   help="open/release the UVC RGB stream with OpenCV first "
                        "(wakes the device on macOS)")
    p.add_argument("--no-save", action="store_true",
                   help="do not write data/rs_check/{rgb,depth}.png|jpg")
    args = p.parse_args()

    import pyrealsense2 as rs

    if args.cv2_warmup:
        import cv2
        cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            for _ in range(5):
                cap.read()
            cap.release()
            print("cv2 warmup done", flush=True)
        time.sleep(1)

    ctx = rs.context()
    devs = None
    for attempt in range(5):
        try:
            devs = ctx.query_devices()
            print(f"enumerate attempt {attempt}: {len(devs)} device(s)", flush=True)
            break
        except Exception as e:
            print(f"enumerate attempt {attempt}: {e}", flush=True)
            time.sleep(2)
    if not devs:
        print("FAIL: no device. Close any app using the camera "
              "(browser Live tab!) and retry.", flush=True)
        return 1

    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height,
                      rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height,
                      rs.format.z16, args.fps)
    profile = None
    for attempt in range(5):
        pipe = rs.pipeline(ctx)
        try:
            profile = pipe.start(cfg)
            print(f"pipeline started (attempt {attempt})", flush=True)
            break
        except Exception as e:
            print(f"start attempt {attempt}: {e}", flush=True)
            time.sleep(3)
    if profile is None:
        print("FAIL: pipeline will not start. Is the camera held by another "
              "app? Try closing the browser Live tab.", flush=True)
        return 1

    align = rs.align(rs.stream.color)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    print(f"depth_scale={scale}  fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f} "
          f"{intr.width}x{intr.height}", flush=True)

    ok = 0
    t0 = time.time()
    for i in range(args.frames):
        fs = align.process(pipe.wait_for_frames(5000))
        if fs is None:
            print(f"frame {i}: None", flush=True)
            continue
        raw = np.asanyarray(fs.get_depth_frame().get_data())
        d = raw.astype(np.float32) * np.float32(scale)
        v = d[d > 0]
        if v.size:
            ok += 1
            print(f"frame {i}: valid={v.size} ({100*v.size/d.size:.1f}%) "
                  f"min={v.min():.3f} m  median={np.median(v):.3f} m  "
                  f"max={v.max():.3f} m", flush=True)
        if not args.no_save and i == args.frames - 1:
            import cv2
            out = Path("data/rs_check")
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "rgb.jpg"),
                        np.asanyarray(fs.get_color_frame().get_data()))
            cv2.imwrite(str(out / "depth.png"),
                        (np.clip(d, 0, 1.5) / 1.5 * 65535).astype(np.uint16))
            print(f"saved {out/'rgb.jpg'} and {out/'depth.png'}", flush=True)
    pipe.stop()
    print(f"DONE: {ok}/{args.frames} frames with depth in "
          f"{time.time()-t0:.1f}s", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
