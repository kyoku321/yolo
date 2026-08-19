"""Train a plain YOLO shelf detector on SKU-110K.

SKU-110K is a single-class ("object") dense-shelf dataset. Training a normal
YOLO on it yields a tight, dense *product* detector — exactly what the
open-vocab YOLO-World lacks. Ultralytics auto-downloads the dataset (~13 GB)
into ./datasets on first run.

Examples:
    # quick demo-quality run (subset of data, few epochs)
    python scripts/train_sku110k.py --model yolo26n.pt --epochs 15 --fraction 0.15

    # full run (best quality, slow)
    python scripts/train_sku110k.py --model yolo26n.pt --epochs 50

Result weights land at: runs/detect/sku110k/weights/best.pt
Then point the app at them:
    export DETECTOR=custom
    export YOLO_WEIGHTS=runs/detect/sku110k/weights/best.pt
    python app.py
"""
from __future__ import annotations

import argparse


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt",
                    help="base weights to fine-tune from "
                         "(yolo26n/s/m ... or legacy yolov8n/s/m ...)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="fraction of the TRAIN set to use per epoch "
                         "(e.g. 0.15 = 15%%, much faster for a demo)")
    ap.add_argument("--device", default=None,
                    help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stop patience (epochs w/o improvement)")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Training {args.model} on SKU-110K | device={device} "
          f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
          f"fraction={args.fraction}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data="SKU-110K.yaml",     # auto-downloaded by Ultralytics (~13 GB)
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        device=device,
        patience=args.patience,
        project="runs/detect",
        name="sku110k",
        exist_ok=True,
    )
    print("\nDone. Best weights: runs/detect/sku110k/weights/best.pt")
    print("Use them with:  export DETECTOR=custom "
          "YOLO_WEIGHTS=runs/detect/sku110k/weights/best.pt")


if __name__ == "__main__":
    main()
