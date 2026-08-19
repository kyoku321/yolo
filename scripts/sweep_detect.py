"""Sweep detection params (imgsz / conf / NMS iou) on one shelf photo to
diagnose 'one object -> multiple boxes' problems.

For every combo it prints box count + a duplicate indicator (share of boxes
that overlap another box with IoU > 0.3) and saves annotated zoom crops of
three shelf regions to data/sweep/ for eyeballing.

Usage:
    python scripts/sweep_detect.py data/shelf.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shelf_demo.draw import _font  # noqa: E402

COMBOS = [
    # (imgsz, conf, nms_iou)
    (1280, 0.20, 0.50),   # current default
    (1280, 0.20, 0.30),   # stricter NMS
    (1280, 0.30, 0.30),   # stricter NMS + higher conf
    (1600, 0.20, 0.30),
    (1920, 0.20, 0.30),
]

# zoom windows (x1, y1, x2, y2) on the original 4096x3072 photo
CROPS = {
    "hangers": (500, 250, 2200, 1300),
    "galbo": (500, 1150, 2600, 2100),
    "bottom_trays": (900, 2000, 3100, 3072),
}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main() -> None:
    from ultralytics import YOLO

    image_path = sys.argv[1] if len(sys.argv) > 1 else "data/shelf.jpg"
    image = Image.open(image_path).convert("RGB")
    out_dir = Path("data/sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("weights/sku110k_best.pt")

    print(f"{'imgsz':>5} {'conf':>4} {'iou':>4} | {'boxes':>5} {'dup%':>5}")
    print("-" * 40)
    for imgsz, conf, nms_iou in COMBOS:
        r = model.predict(
            source=image, imgsz=imgsz, conf=conf, iou=nms_iou,
            max_det=1000, agnostic_nms=True, device="mps", verbose=False,
        )[0]
        boxes = [tuple(b.xyxy[0].tolist()) for b in r.boxes]

        # duplicate indicator: share of boxes overlapping another box
        dup = sum(
            1 for i, a in enumerate(boxes)
            if any(iou(a, b) > 0.3 for j, b in enumerate(boxes) if i != j)
        )
        tag = f"s{imgsz}_c{conf}_i{nms_iou}"
        print(f"{imgsz:>5} {conf:>4} {nms_iou:>4} | {len(boxes):>5} "
              f"{100 * dup / max(1, len(boxes)):>4.0f}%")

        for region, (cx1, cy1, cx2, cy2) in CROPS.items():
            crop = image.crop((cx1, cy1, cx2, cy2)).copy()
            draw = ImageDraw.Draw(crop)
            font = _font(16)
            for (x1, y1, x2, y2) in boxes:
                if x2 < cx1 or x1 > cx2 or y2 < cy1 or y1 > cy2:
                    continue
                draw.rectangle(
                    [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
                    outline=(34, 197, 94), width=3,
                )
                conf_i = next(
                    (float(b.conf[0]) for b in r.boxes
                     if tuple(b.xyxy[0].tolist()) == (x1, y1, x2, y2)), 0.0
                )
                draw.text((x1 - cx1 + 3, y1 - cy1 + 3), f"{conf_i:.2f}",
                          fill=(34, 197, 94), font=font)
            crop.save(out_dir / f"{tag}_{region}.jpg", quality=85)

    print(f"\ncrops saved to {out_dir}/")


if __name__ == "__main__":
    main()
