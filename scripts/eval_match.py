"""Retrieval benchmark: does a context margin fix crop-vs-reference scoring?

The catalog stores each product as the REGISTERED PHOTO (a full product shot),
while the query is a detector BOX CROP. A tight crop drops the context the
reference has, which pushes the true match's cosine down (ALKALINE scored
0.592 vs a 0.65 threshold, runner-up only 0.360).

This measures, over every reference photo in data/crops, the true-match score
and top-1 accuracy for:
    tight box      (scale 1.00)
    padded box     (scale 1.10 / 1.20)
    max over scales (test-time augmentation: never worse than the best scale)

    .venv/bin/python scripts/eval_match.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from shelf_demo import config  # noqa: E402
from shelf_demo.detector import build_detector, padded_crop  # noqa: E402
from shelf_demo.embedder import Embedder  # noqa: E402

CROPS = sorted((ROOT / "data" / "crops").glob("sku*.jpg"))
SCALES = [1.0, 1.10, 1.20]
IMGSZ = 960


def pad_crop(image: Image.Image, box, scale: float) -> Image.Image:
    return padded_crop(image, box, scale)


def main() -> None:
    # Reference set = full registered photos, in a fixed order.
    names = [p.name for p in CROPS]
    refs = [Image.open(p).convert("RGB") for p in CROPS]

    emb = Embedder()
    det = build_detector("auto")

    print(f"references: {len(refs)}   scales: {SCALES}   imgsz={IMGSZ}")
    print("embedding reference photos ...")
    R = emb.embed(refs)                      # (N, D)

    scores = {s: [] for s in SCALES}         # true-match score per scale
    top1 = {s: 0 for s in SCALES}
    top1_max = 0
    missing = 0
    rows = []
    for i, (name, img) in enumerate(zip(names, refs)):
        boxes = det.detect(img, imgsz=IMGSZ)
        if not boxes:
            missing += 1
            continue
        box = max(boxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
        sims = []
        for s in SCALES:
            q = emb.embed([pad_crop(img, box, s)])   # (1, D)
            sim = (q @ R.T)[0]                       # (N,)
            sims.append(sim)
            scores[s].append(float(sim[i]))
            top1[s] += int(int(np.argmax(sim)) == i)
        sim_max = np.max(np.stack(sims), axis=0)
        top1_max += int(int(np.argmax(sim_max)) == i)
        rows.append((name, box, [float(np.max(s)) for s in sims],
                     float(np.max(sim_max)),
                     [names[int(np.argmax(s))] for s in sims]))

    n = len(rows)
    print(f"\nscored {n}/{len(refs)} photos ({missing} had no detection)")
    print(f"\n{'photo':<40}{'box':<22}" + "".join(f"{s:>9}" for s in SCALES)
          + f"{'max':>9}")
    for name, box, s1, smax, who in rows:
        mark = "" if smax == max(s1) else "?"
        print(f"{name[:39]:<40}{str(box.xyxy):<22}"
              + "".join(f"{v:>9.3f}" for v in s1) + f"{smax:>9.3f}{mark}")

    print(f"\n{'mean true-match score':<40}"
          + "".join(f"{np.mean(scores[s]):>9.3f}" for s in SCALES)
          + f"{np.mean([r[3] for r in rows]):>9.3f}")
    print(f"{'median true-match score':<40}"
          + "".join(f"{np.median(scores[s]):>9.3f}" for s in SCALES)
          + f"{np.median([r[3] for r in rows]):>9.3f}")
    print(f"{'top-1 accuracy':<40}"
          + "".join(f"{top1[s]}/{n:<7}" for s in SCALES) + f"{top1_max}/{n}")
    print(f"\ncurrent MATCH_THRESHOLD={config.MATCH_THRESHOLD}")
    mx = np.array([r[3] for r in rows])
    for thr in (0.55, 0.60, 0.65):
        print(f"  with max-over-scales: {int((mx >= thr).sum())}/{n} photos "
              f"clear threshold {thr}")


if __name__ == "__main__":
    main()
