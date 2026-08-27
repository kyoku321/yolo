"""On-board eye-in-hand loop (RealSense -> recognise -> 3D grasp answer).

Runs WITHOUT the web UI: camera frames feed RGBDGraspSession directly and
a target product's 3D position prints once per second. This is the entry
point the servo controller will sit behind later.

  # 2D-only smoke (no hand-eye file yet):
  python scripts/rs_live.py --target "Cola"

  # full 3D in the ARM BASE frame (after calibrate_handeye.py):
  python scripts/rs_live.py --target "Cola" --fk-file data/fk.json

--fk-file: JSON written continuously by your arm bridge
({"rotation": rotvec, "translation": [x,y,z]} for T_base_ee). Without it
(and without handeye.json) you still get point_cam_m.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from shelf_demo import config
from shelf_demo.calibration import fk_json_to_T, load_handeye
from shelf_demo.camera import RealSenseCamera
from shelf_demo.pipeline import ShelfPipeline
from shelf_demo.rgbd_live import RGBDGraspSession


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="", help="product name to 3D-locate")
    p.add_argument("--fk-file", default="",
                   help="JSON with live T_base_ee written by your arm bridge")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between grasp3d prints")
    args = p.parse_args()

    try:
        T_ee_cam = load_handeye()
        print(f"hand-eye loaded from {config.HANDEYE_PATH}")
    except FileNotFoundError as e:
        print(f"note: {e} -> camera-frame output only")
        T_ee_cam = None

    print("loading models ...")
    pipeline = ShelfPipeline()
    session = RGBDGraspSession(pipeline, T_ee_cam=T_ee_cam,
                               imgsz=args.imgsz)
    fk_path = Path(args.fk_file) if args.fk_file else None

    t_last = 0.0
    with RealSenseCamera() as cam:
        print(f"streaming {cam.width}x{cam.height}@{cam.fps}fps - Ctrl-C to quit")
        try:
            while True:
                frame = cam.read()
                fk = None
                if fk_path and fk_path.exists():
                    fk = fk_json_to_T(json.loads(fk_path.read_text()))
                result = session.process_frame(frame, fk=fk)
                now = time.monotonic()
                if result.n_new:
                    names = sorted({r.name for r in result.recognitions
                                    if r.is_match})
                    print(f"[{now % 1000:9.1f}] recognised {result.n_new} new "
                          f"object(s); scene: {names}")
                if args.target and now - t_last >= args.interval:
                    t_last = now
                    st = session.grasp3d(args.target)
                    if st["ready"]:
                        print(f"[{now % 1000:9.1f}] {args.target}: "
                              f"cam={st['point_cam_m']} "
                              f"base={st['point_base_m']} "
                              f"({st['valid_px']}px)")
                    else:
                        print(f"[{now % 1000:9.1f}] {args.target}: "
                              f"not ready - {st.get('reason', '?')}")
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
