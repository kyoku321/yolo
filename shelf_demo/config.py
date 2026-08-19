"""Central configuration. Every value can be overridden with an env var."""
from __future__ import annotations

import os
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("SHELF_DATA_DIR", str(ROOT / "data")))
INDEX_PATH = DATA_DIR / "vectors.npz"   # numpy-backed vector store
DB_PATH = DATA_DIR / "catalog.sqlite"
CROP_DIR = DATA_DIR / "crops"          # saved reference crops (for gallery)

# ---- Detection (YOLO) ------------------------------------------------------
# DETECTOR selects the detection backend:
#   "coco"  -> plain YOLO on COCO classes (yolov8n.pt). Auto-downloads and runs
#              out of the box, but CANNOT detect generic retail products
#              (snacks, packages) — it only knows COCO's 80 classes. Fine for a
#              first launch, useless on a real shelf.
#   "world" -> YOLO-World open-vocabulary detector. Detects arbitrary objects
#              from text prompts (YOLO_WORLD_PROMPTS) with NO training. This is
#              the recommended zero-setup choice for a shelf demo.
#   "custom"-> your own weights at YOLO_WEIGHTS, e.g. a model trained on
#              SKU-110K (see scripts/train_sku110k.py). Best quality for dense
#              shelves.
# If the SKU-110K-trained weights exist (produced by scripts/train_sku110k.py),
# default to them: far tighter boxes on dense shelves than zero-shot
# YOLO-World. Override anytime with DETECTOR / YOLO_WEIGHTS env vars.
_SKU110K_WEIGHTS = ROOT / "weights" / "sku110k_best.pt"
_SKU110K_AVAILABLE = _SKU110K_WEIGHTS.exists()
DETECTOR = _env("DETECTOR", "custom" if _SKU110K_AVAILABLE else "world")
YOLO_WEIGHTS = _env(
    "YOLO_WEIGHTS",
    str(_SKU110K_WEIGHTS) if _SKU110K_AVAILABLE else "yolov8n.pt",
)   # used by "coco"/"custom"

# YOLO-World open-vocabulary settings.
YOLO_WORLD_WEIGHTS = _env("YOLO_WORLD_WEIGHTS", "yolov8s-worldv2.pt")
YOLO_WORLD_PROMPTS = _env(
    "YOLO_WORLD_PROMPTS",
    "product,package,snack bag,pouch,box,bottle,can,carton,chocolate bar,"
    "bag of chips,candy",
)
# Dense retail shelves need a LARGE inference size (products become tiny when a
# 4K photo is squashed to 640) and a LOW confidence (open-vocab prompts score
# low). These defaults are the single biggest reason detection "finds nothing".
DETECT_IMGSZ = int(_env("DETECT_IMGSZ", "1280"))       # try 1920 for 4K photos
# Default conf follows the detector: trained SKU-110K weights are well
# calibrated (0.2 keeps the UI clean); YOLO-World scores low, needs 0.05.
DETECT_CONF = float(_env(
    "DETECT_CONF", "0.2" if DETECTOR == "custom" else "0.05"
))
# NMS IoU: 0.5 is standard, but the trained single-class model emits box
# pairs offset by ~half a box on the same product (IoU ~0.4-0.5) — 0.5 keeps
# both and you see two boxes on one item. 0.3 suppresses them without
# merging genuinely adjacent facings.
DETECT_IOU = float(_env(
    "DETECT_IOU", "0.3" if DETECTOR == "custom" else "0.5"
))
DETECT_MAX_DET = int(_env("DETECT_MAX_DET", "1000"))   # shelves are dense
# If True, every detected box is treated as a product region regardless of its
# COCO class. Correct behaviour for a single-class SKU-110K model too.
DETECT_CLASS_AGNOSTIC = _env("DETECT_CLASS_AGNOSTIC", "1") == "1"

# ---- Embedding (open_clip) -------------------------------------------------
# ViT-B-16 >> ViT-B-32 for fine-grained retail SKUs. On this shelf, B-32 ranked
# a different product (galbo) ABOVE the real GABA; B-16 puts GABA on top with a
# healthy margin. For even better separation try SigLIP:
#   CLIP_MODEL=ViT-B-16-SigLIP  CLIP_PRETRAINED=webli   (EMBED_DIM auto-detected)
CLIP_MODEL = _env("CLIP_MODEL", "ViT-B-16")
CLIP_PRETRAINED = _env("CLIP_PRETRAINED", "laion2b_s34b_b88k")
EMBED_DIM = int(_env("EMBED_DIM", "512"))   # ViT-B-16 -> 512 (auto-checked)

# ---- Retrieval -------------------------------------------------------------
# Cosine similarity below this => "Unknown" (product not in the catalog).
MATCH_THRESHOLD = float(_env("MATCH_THRESHOLD", "0.65"))   # calibrated for ViT-B-16
SEARCH_TOPK = int(_env("SEARCH_TOPK", "5"))

# ---- Device ----------------------------------------------------------------
DEVICE = _env("SHELF_DEVICE", "auto")   # auto | cpu | cuda | mps


def pick_device() -> str:
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)
