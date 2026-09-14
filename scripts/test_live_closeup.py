"""End-to-end Live-path check for the close-up case that was broken.

Feeds the registered ALKALINE reference photo (a product filling the frame,
exactly how it is held in front of the OBSBOT camera) through the real
pipeline + LiveRecognizer, and reports:

  * how many boxes the detector produced (before the fix: 0 in the Live tab),
  * which track path ran (specialist vs open-vocab fallback),
  * the fused SKU label after the temporal vote (needs data/catalog.sqlite).

    .venv/bin/python scripts/test_live_closeup.py
    .venv/bin/python scripts/test_live_closeup.py data/crops/xxx.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from shelf_demo import config  # noqa: E402
from shelf_demo.live import LiveRecognizer  # noqa: E402
from shelf_demo.pipeline import ShelfPipeline  # noqa: E402

DEFAULT = ROOT / "data" / "crops" / "sku0026_20260901002504_0.jpg"  # ALKALINE
FRAMES = 6
IMGSZ = 960   # the Live tab's default


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    image = Image.open(path).convert("RGB")

    pipe = ShelfPipeline()
    det = pipe.detector
    print(f"image          : {path.name} {image.size}")
    print(f"detector       : {det.kind}  conf={config.DETECT_CONF}  imgsz={IMGSZ}")
    print(f"catalog        : {pipe.catalog.n_vectors} vectors")

    rec = LiveRecognizer(pipe, imgsz=IMGSZ)
    result = None
    for i in range(FRAMES):
        result = rec.process(image)
    assert result is not None

    print(f"detector path  : {getattr(det, 'last_path', det.kind)}")
    print(f"boxes          : {len(result.recognitions)}")
    for r in result.recognitions:
        print(f"   track {r.track_id}: {r.box.xyxy}  {r.label}")
    matched = [r for r in result.recognitions if r.is_match]
    print(f"matched SKUs   : {[r.name for r in matched]}")
    print("\nOK: a bounding box is produced on a close-up product."
          if result.recognitions else
          "\nFAIL: no bounding box on a close-up product.")


if __name__ == "__main__":
    main()
