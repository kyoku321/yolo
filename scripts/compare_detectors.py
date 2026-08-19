"""Compare detection quality: custom SKU-110K weights vs zero-shot YOLO-World.

Runs both detectors on the same shelf photo and saves one annotated image
per detector so you can eyeball box tightness:

    data/compare_custom.jpg   <- weights/sku110k_best.pt (trained detector)
    data/compare_world.jpg    <- YOLO-World open-vocab (zero-shot prompts)

Also prints box counts and box-size statistics. The trained detector should
find MORE products with TIGHTER boxes; YOLO-World tends to draw loose,
overlapping boxes around whole shelf regions.

Usage:
    python scripts/compare_detectors.py data/shelf.jpg
    python scripts/compare_detectors.py data/shelf.jpg --conf-custom 0.2 --conf-world 0.05
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shelf_demo import config                  # noqa: E402
from shelf_demo.detector import Box, Detector  # noqa: E402
from shelf_demo.draw import _font              # noqa: E402


def annotate(image: Image.Image, boxes: list[Box], color: tuple[int, int, int],
             out_path: Path) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line = max(2, image.size[0] // 1000)
    font = _font(max(14, image.size[0] // 180))
    for b in boxes:
        draw.rectangle(b.xyxy, outline=color, width=line)
        draw.text((b.x1 + 3, b.y1 + 3), f"{b.conf:.2f}", fill=color, font=font)
    canvas.save(out_path, quality=88)


def stats(name: str, boxes: list[Box], img_w: int, img_h: int) -> None:
    if not boxes:
        print(f"  {name:<8}  0 boxes")
        return
    areas = [(b.x2 - b.x1) * (b.y2 - b.y1) for b in boxes]
    widths = [b.x2 - b.x1 for b in boxes]
    heights = [b.y2 - b.y1 for b in boxes]
    img_area = img_w * img_h
    print(f"  {name:<8}  {len(boxes):>3} boxes | "
          f"median box {statistics.median(widths):.0f}x"
          f"{statistics.median(heights):.0f}px "
          f"({100 * statistics.median(areas) / img_area:.3f}% of image) | "
          f"conf median {statistics.median(b.conf for b in boxes):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default=str(config.ROOT / "weights" / "sku110k_best.pt"))
    ap.add_argument("--conf-custom", type=float, default=0.2,
                    help="conf for the trained detector (it is well-calibrated)")
    ap.add_argument("--conf-world", type=float, default=config.DETECT_CONF,
                    help="conf for YOLO-World (open-vocab scores run low)")
    ap.add_argument("--imgsz", type=int, default=config.DETECT_IMGSZ)
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    print(f"Image: {args.image}  {w}x{h}\n")

    config.ensure_dirs()

    print("[1/2] custom (SKU-110K trained) ...")
    det_custom = Detector(detector="custom", weights=args.weights)
    boxes_custom = det_custom.detect(image, conf=args.conf_custom, imgsz=args.imgsz)
    out_custom = config.DATA_DIR / "compare_custom.jpg"
    annotate(image, boxes_custom, (34, 197, 94), out_custom)

    print("[2/2] world (YOLO-World zero-shot) ...")
    det_world = Detector(detector="world")
    boxes_world = det_world.detect(image, conf=args.conf_world, imgsz=args.imgsz)
    out_world = config.DATA_DIR / "compare_world.jpg"
    annotate(image, boxes_world, (245, 158, 11), out_world)

    print(f"\nResults at imgsz={args.imgsz} "
          f"(custom conf>={args.conf_custom}, world conf>={args.conf_world}):")
    stats("custom", boxes_custom, w, h)
    stats("world", boxes_world, w, h)
    print(f"\nAnnotated images:\n  {out_custom}\n  {out_world}")
    print("Green = trained SKU-110K detector, orange = YOLO-World zero-shot.")


if __name__ == "__main__":
    main()
