"""Train YOLO26n / YOLO8n / YOLO11n on SKU-110K with identical hyperparameters
and compare them on the same validation & test splits.

Usage:
    python scripts/train_compare.py --epochs 50 --imgsz 640 --batch 32

Output:
    runs/detect/compare_<model>/   per-model training artifacts
    runs/detect/compare_summary.csv  final comparison table
"""
from __future__ import annotations

import argparse
import csv
import glob
import time
from pathlib import Path


def find_best_weights(proj: str) -> Path | None:
    cands = glob.glob(f"runs/**/{proj}/**/best.pt", recursive=True)
    return Path(cands[0]) if cands else None


def train_one(model: str, args, device: str) -> Path:
    from ultralytics import YOLO

    name = model.replace(".pt", "")  # e.g. yolo26n / yolov8n / yolo11n
    proj = f"compare_{name}"
    if args.reuse:
        existing = find_best_weights(proj)
        if existing:
            print(f"\n{'='*70}\nREUSING {model}  ->  {existing}\n{'='*70}", flush=True)
            return existing
        print(f"\n{'='*70}\nNO EXISTING WEIGHTS FOR {model}, TRAINING\n{'='*70}", flush=True)

    print(f"\n{'='*70}\nTRAINING {model}  ->  {proj}\n{'='*70}", flush=True)

    yolo = YOLO(model)
    t0 = time.time()
    yolo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        seed=0,
        cos_lr=True,
        project=proj,
        name="weights",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    dt = time.time() - t0
    print(f"[{name}] training finished in {dt/3600:.2f} h", flush=True)
    best = find_best_weights(proj)
    if best is None:
        raise FileNotFoundError(f"best.pt not found after training {model}")
    return best


def test_one(weights: Path, split: str, args, device: str) -> dict:
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    print(f"[{weights.parent.parent.name}] testing on {split} split ...", flush=True)
    res = yolo.val(
        data=args.data,
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        verbose=False,
        plots=False,
    )
    m = res.box
    return {
        "precision": m.mp,
        "recall": m.mr,
        "map50": m.map50,
        "map50_95": m.map,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["yolov8n.pt", "yolo11n.pt", "yolo26n.pt"])
    ap.add_argument("--data", default="SKU-110K.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-test", action="store_true",
                    help="skip the final test-split evaluation")
    ap.add_argument("--reuse", action="store_true",
                    help="skip training if best.pt already exists for the model")
    args = ap.parse_args()

    results = []
    for model in args.models:
        weights = train_one(model, args, args.device)
        row = {"model": model.replace(".pt", "")}
        if args.skip_test:
            results.append(row)
            continue
        for split in ("val", "test"):
            m = test_one(weights, split, args, args.device)
            row[f"{split}_p"] = round(m["precision"], 4)
            row[f"{split}_r"] = round(m["recall"], 4)
            row[f"{split}_map50"] = round(m["map50"], 4)
            row[f"{split}_map50_95"] = round(m["map50_95"], 4)
        results.append(row)
        # incremental save so we keep results even if a later run crashes
        save(results)

    save(results)
    print_table(results)


def save(results: list[dict]) -> None:
    out = Path("runs/detect/compare_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if results:
        keys = []
        for r in results:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
    print(f"summary saved -> {out}", flush=True)


def print_table(results: list[dict]) -> None:
    cols = ["model", "val_map50", "val_map50_95", "val_p", "val_r",
            "test_map50", "test_map50_95", "test_p", "test_r"]
    cols = [c for c in cols if any(c in r for r in results)]
    header = f"{'model':<12}" + "".join(f"{c:>16}" for c in cols[1:])
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r['model']:<12}"
        for c in cols[1:]:
            v = r.get(c, "")
            line += f"{v:>16}"
        print(line)
    print("=" * len(header))


if __name__ == "__main__":
    main()
