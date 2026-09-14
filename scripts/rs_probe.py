"""RealSense health probe: no models, no recognition — just the hardware path.

First thing to run when the D435i is (re)plugged:

    python scripts/rs_probe.py

Checks: device enumeration, pipeline start, aligned RGB+depth frames,
depth scale, intrinsics, and per-frame depth statistics. Exit 0 = hardware
path is healthy and scripts/rs_live.py can be trusted.

macOS: needs `pip install pyrealsense2-macosx` (community wheel, arm64).
Linux: `pip install pyrealsense2`.

If the pipeline fails with "failed to set power state" / "No device
connected": stop the browser Live tab (it holds the UVC stream), then
unplug + replug the camera (direct USB 3 port, not through a hub).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def start_and_read(rs, width: int, height: int, fps: int,
                   frames: int, tries: int = 3):
    """Start the pipeline (up to `tries` full restarts) and capture frames.

    Returns (ok_count, scale, intr, last_error). macOS USB quirk: the first
    start after plug often delivers no frames; a restart usually fixes it.
    """
    last_err = ""
    for attempt in range(1, tries + 1):
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        t0 = time.monotonic()
        try:
            profile = pipe.start(cfg)
        except Exception as e:   # noqa: BLE001
            last_err = f"pipeline start failed: {e}"
            print(f"  attempt {attempt}: {last_err}")
            time.sleep(1.0)
            continue
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
        ok = 0
        t_last = time.monotonic()
        err = ""
        for i in range(frames):
            try:
                fs = align.process(pipe.wait_for_frames(5000))
            except Exception as e:   # noqa: BLE001
                err = str(e)
                break
            color, depth = fs.get_color_frame(), fs.get_depth_frame()
            if not (color and depth):
                continue
            ok += 1
            dm = np.asanyarray(depth.get_data()).astype(np.float32) * scale
            valid = dm[dm > 0]
            now = time.monotonic()
            if ok == 1:
                print(f"  attempt {attempt}: first aligned frame "
                      f"{color.width}x{color.height} "
                      f"({(time.monotonic() - t0) * 1000:.0f} ms after start) "
                      f"depth valid {valid.size}/{dm.size} px")
            if ok % max(1, frames // 3) == 0 or ok == frames:
                print(f"    frame {ok:3d}/{frames}: valid={valid.size:6d} px "
                      f"z median={np.median(valid):.3f} m "
                      f"p10={np.percentile(valid, 10):.3f} "
                      f"p90={np.percentile(valid, 90):.3f} "
                      f"~{1000.0 / max(now - t_last, 1e-6):.0f} fps")
                t_last = now
        try:
            pipe.stop()
        except Exception:   # noqa: BLE001
            pass
        if ok >= max(10, frames // 2):
            return ok, scale, intr, ""
        last_err = err or f"only {ok}/{frames} frames"
        print(f"  attempt {attempt}: {last_err} - retrying")
        time.sleep(1.0)
    return 0, None, None, last_err


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=150,
                   help="frames to capture (default 150 = ~5 s @30fps)")
    args = p.parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError as e:
        print(f"ERROR: pyrealsense2 not importable: {e}")
        print("macOS: pip install pyrealsense2-macosx")
        return 1

    ctx = rs.context()
    devices = ctx.query_devices()
    if not len(devices):
        print("ERROR: no RealSense device visible to the SDK.")
        print("  1) stop the browser Live tab (it holds the UVC stream)")
        print("  2) unplug + replug the D435i, direct USB 3 port")
        return 1
    for d in devices:
        print(f"device: {d.get_info(rs.camera_info.name)} "
              f"(serial {d.get_info(rs.camera_info.serial_number)})")

    # Full resolution first; if frames starve, retry at 640x480 to tell
    # "USB bandwidth/hub problem" apart from "device problem".
    ok, scale, intr, err = start_and_read(
        rs, args.width, args.height, args.fps, args.frames)
    if ok == 0 and (args.width, args.height) != (640, 480):
        print(f"full {args.width}x{args.height} failed ({err}) - "
              "falling back to 640x480 to test USB bandwidth")
        ok, scale, intr, err = start_and_read(rs, 640, 480, args.fps,
                                              args.frames)

    if ok == 0:
        print(f"FAIL: no frames at any resolution. Last error: {err}")
        print("  try: another direct USB 3 port (no hub), different cable")
        return 1
    print(f"depth_scale={scale}  K: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
    print(f"OK: {ok} aligned RGB-D frames captured. Hardware path healthy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
