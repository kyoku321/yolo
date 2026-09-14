"""Segfault localizer for the macOS + sudo + D435i path.

Run with:  sudo .venv/bin/python scripts/diag_sudo.py

Each step prints a marker. If the process segfaults, faulthandler prints a
"Current thread" Python stack showing the EXACT line that crashed — paste
that whole output.
"""
from __future__ import annotations

import faulthandler
import pathlib
import sys
import time

faulthandler.enable()


def step(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    step("step1: import pyrealsense2 (no torch yet)")
    import pyrealsense2 as rs
    step(f"pyrealsense2 {getattr(rs, '__version__', '?')}, "
         f"lib path: {getattr(rs, '__file__', '?')}")

    step("step2: context + enumerate devices")
    ctx = rs.context()
    devs = ctx.query_devices()
    step(f"devices visible: {len(devs)}")
    for d in devs:
        step(f"  - {d.get_info(rs.camera_info.name)} "
             f"(serial {d.get_info(rs.camera_info.serial_number)})")
    if not len(devs):
        step("NO DEVICE: unplug the D435i, wait 5 s, replug into a direct "
             "USB-C port, then rerun")
        return 1

    step("step3: import torch + load models (coexistence check)")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import torch
    step(f"torch {torch.__version__}, mps={torch.backends.mps.is_available()}")
    from shelf_demo.pipeline import ShelfPipeline
    pipeline = ShelfPipeline()
    step("models OK")

    step("step4: RealSenseCamera start (warm-up + retries inside)")
    from shelf_demo.camera import RealSenseCamera
    cam = RealSenseCamera()
    with cam:
        step("step5: first read")
        f = cam.read()
        step(f"frame OK {f.rgb.shape}")
        step("step6: 30 reads (sustained streaming)")
        for i in range(30):
            f = cam.read()
            if (i + 1) % 10 == 0:
                step(f"  read {i + 1}/30 ok")
        step("step7: context exit (pipeline.stop on healthy pipeline)")
    step("step8: clean exit - camera side fully healthy")

    step("step9: one inference on the captured frame")
    out = pipeline.recognize(f.rgb)
    step(f"recognized: {[(n.name, round(n.score, 2)) for n in out]}")
    step("ALL STEPS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
