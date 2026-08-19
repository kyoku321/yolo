"""Debug the pipeline stage by stage on a single shelf image.

It answers the two separate questions that "No products detected" conflates:
  1. DETECTION — did YOLO find any boxes at all?
  2. RETRIEVAL — for the boxes found, how close is the best catalog match?

Usage:
    python scripts/debug_detect.py path/to/shelf.jpg
    python scripts/debug_detect.py path/to/shelf.jpg --detector world --conf 0.05
    python scripts/debug_detect.py path/to/shelf.jpg --detector coco

Writes an annotated image (every raw detection, no threshold) to
data/debug_boxes.jpg so you can see exactly what the detector sees.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shelf_demo import config                       # noqa: E402
from shelf_demo.detector import Detector            # noqa: E402
from shelf_demo.embedder import Embedder            # noqa: E402
from shelf_demo.database import Catalog             # noqa: E402
from shelf_demo.draw import _font                   # noqa: E402
from PIL import ImageDraw                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--detector", default=None,
                    help="coco | world | custom (default: config.DETECTOR)")
    ap.add_argument("--conf", type=float, default=config.DETECT_CONF)
    ap.add_argument("--imgsz", type=int, default=config.DETECT_IMGSZ,
                    help="inference size; raise to 1920+ for 4K dense shelves")
    args = ap.parse_args()

    image = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image}  size={image.size}")

    det = Detector(detector=args.detector)
    print(f"Detector kind = {det.kind}  weights = {det.weights}")
    if det.prompts:
        print(f"Open-vocab prompts = {det.prompts}")

    # --- stage 1: detection ------------------------------------------------
    boxes = det.detect(image, conf=args.conf, imgsz=args.imgsz)
    print(f"\n[STAGE 1] DETECTION: {len(boxes)} boxes "
          f"at imgsz={args.imgsz}, conf>={args.conf}")
    if not boxes:
        print("  -> 0 boxes. This is your problem: the detector finds nothing.")
        print("     Fixes: raise --imgsz (e.g. 1920 for 4K photos), lower")
        print("     --conf (e.g. 0.02), or train a custom SKU-110K detector.")
        return

    # annotate every raw box (independent of retrieval)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(max(14, image.size[0] // 90))
    for i, b in enumerate(boxes):
        draw.rectangle(b.xyxy, outline=(34, 197, 94), width=2)
        draw.text((b.x1 + 2, b.y1 + 2), str(i), fill=(34, 197, 94), font=font)
    config.ensure_dirs()
    out = config.DATA_DIR / "debug_boxes.jpg"
    canvas.save(out)
    print(f"  -> annotated detections saved to {out}")

    # --- stage 2: retrieval ------------------------------------------------
    cat = Catalog()
    if cat.n_vectors == 0:
        print("\n[STAGE 2] RETRIEVAL: catalog is EMPTY — register a product "
              "first, then re-run to see match scores.")
        return

    emb = Embedder()
    crops = [b.crop(image) for b in boxes]
    embs = emb.embed(crops)
    hits = cat.search(embs, topk=1)

    print(f"\n[STAGE 2] RETRIEVAL vs {cat.n_vectors} catalog vectors "
          f"(match threshold = {config.MATCH_THRESHOLD}):")
    scored = []
    for i, hit in enumerate(hits):
        if hit:
            sku_id, score = hit[0]
            p = cat.get(sku_id)
            scored.append((score, i, p.name if p else "?"))
    scored.sort(reverse=True)
    for score, i, name in scored[:15]:
        flag = "MATCH" if score >= config.MATCH_THRESHOLD else "     "
        print(f"  box {i:>3}  best={score:.3f}  [{flag}]  {name}")
    best = scored[0][0] if scored else 0.0
    print(f"\n  highest similarity = {best:.3f}")
    if best < config.MATCH_THRESHOLD:
        print("  -> detection works but nothing clears the threshold. Lower")
        print(f"     MATCH_THRESHOLD (try {best-0.05:.2f}) or add more/better")
        print("     reference photos. Zero-shot CLIP is weak on fine SKUs;")
        print("     fine-tuning the embedder is the real fix (see README).")


if __name__ == "__main__":
    main()
