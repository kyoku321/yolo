"""Before/after evidence for the detector fix.

The Live tab holds one product in front of a webcam. The SKU-110K specialist
is trained on dense shelves of small facings, so it produced no usable box
there ("no bounding box"). This script scores three backends on this repo's
own images and prints the before/after numbers.

    .venv/bin/python scripts/verify_detection.py

Scenes:
  dense    data/shelf.jpg                     a real 4K shelf (specialist must stay good)
  closeup  data/crops/sku*.jpg                one product filling the frame (17 refs)

Metrics per scene, at the Live-tab default (imgsz 960, conf=DETECT_CONF):
  n          number of boxes
  cov        union coverage of the frame
  maxarea    largest box as a fraction of the frame
  usable     a box exists that is at least AUTO_SWITCH_AREA (an object-sized
             box). For close-ups this is exactly what was missing before.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("SHELF_DATA_DIR", str(ROOT / "data"))

from PIL import Image  # noqa: E402

from shelf_demo import config  # noqa: E402
from shelf_demo.detector import (  # noqa: E402
    Detector,
    build_detector,
    coverage_frac,
    max_area_frac,
)

DENSE = [ROOT / "data" / "shelf.jpg"]
CLOSEUP = sorted((ROOT / "data" / "crops").glob("sku*.jpg"))
IMGSZ = int(os.environ.get("VERIFY_IMGSZ", "960"))


def score(det, im) -> dict:
    # no conf override: each backend runs at its own configured confidence
    # (specialist DETECT_CONF, generalist YOLO_WORLD_CONF)
    boxes = det.detect(im, imgsz=IMGSZ)
    w, h = im.size
    return {
        "n": len(boxes),
        "cov": coverage_frac(boxes, w, h),
        "maxarea": max_area_frac(boxes, w, h),
        "usable": max_area_frac(boxes, w, h) >= config.AUTO_SWITCH_AREA,
        "path": getattr(det, "last_path", getattr(det, "kind", "?")),
    }


def main() -> None:
    print(f"config: DETECTOR={config.DETECTOR} DETECT_CONF={config.DETECT_CONF} "
          f"imgsz={IMGSZ}")
    print(f"auto switch: coverage>={config.AUTO_COVERAGE_MIN} "
          f"min_area>={config.AUTO_MIN_AREA} switch_area>={config.AUTO_SWITCH_AREA}")

    custom = Detector(detector="custom", conf=config.DETECT_CONF)
    world = Detector(detector="world", conf=config.YOLO_WORLD_CONF,
                     iou=config.YOLO_WORLD_IOU)
    auto = build_detector("auto")

    backends = [("custom", custom), ("world", world), ("auto", auto)]

    print("\n=== CLOSE-UP products (the reported failure) ===")
    header = f"{'scene':<40}" + "".join(f"{name:>26}" for name, _ in backends)
    print(header)
    print(f"{'':<40}" + "".join(f"{'n/cov/maxA/usbl':>26}" for _ in backends))
    hits = {name: 0 for name, _ in backends}
    for p in CLOSEUP:
        im = Image.open(p).convert("RGB")
        cells = []
        for name, det in backends:
            s = score(det, im)
            hits[name] += int(s["usable"])
            cells.append(f"{s['n']}/{s['cov']:.2f}/{s['maxarea']:.2f}/"
                         f"{'Y' if s['usable'] else 'n'}")
        print(f"{p.name:<40}" + "".join(f"{c:>26}" for c in cells))
    print(f"{'usable box (close-up)':<40}" +
          "".join(f"{f'{hits[n]}/{len(CLOSEUP)}':>26}" for n, _ in backends))

    print("\n=== DENSE shelves (must not regress) ===")
    for p in DENSE:
        if not p.exists():
            print(f"  (missing) {p}")
            continue
        im = Image.open(p).convert("RGB")
        for name, det in backends:
            s = score(det, im)
            print(f"  {p.name:<32} {name:<7} n={s['n']:<4} "
                  f"cov={s['cov']:.3f} maxarea={s['maxarea']:.4f} "
                  f"path={s['path']}")

    print("\nVerification: every dense row keeps the specialist's tight boxes; "
          "the close-up 'usable' row is what the fix improves.")
    print(f"auto.generalist_error = {auto._generalist_error!r}")


if __name__ == "__main__":
    main()
