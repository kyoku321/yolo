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
#              from text prompts (YOLO_WORLD_PROMPTS) with NO training. Great
#              on a single product held close to the camera; weak on dense
#              shelves (it merges neighbouring facings).
#   "custom"-> your own weights at YOLO_WEIGHTS, e.g. a model trained on
#              SKU-110K (see scripts/train_sku110k.py). Best quality for dense
#              shelves, but out of distribution on a close-up product: it sees
#              a grid of tiny facings, not one big object.
#   "auto"  -> specialist + fallback (DEFAULT when SKU-110K weights exist).
#              Runs the custom specialist on every frame and, when its output
#              does NOT look like a dense shelf, re-runs YOLO-World and uses it
#              if it found one large object. This is what makes "hold one SKU
#              in front of the webcam" produce a bounding box instead of none.
#              See AutoDetector in shelf_demo/detector.py.
# Override anytime with DETECTOR / YOLO_WEIGHTS env vars.
_SKU110K_WEIGHTS = ROOT / "weights" / "sku110k_best.pt"
_SKU110K_AVAILABLE = _SKU110K_WEIGHTS.exists()
DETECTOR = _env("DETECTOR", "auto" if _SKU110K_AVAILABLE else "world")
YOLO_WEIGHTS = _env(
    "YOLO_WEIGHTS",
    str(_SKU110K_WEIGHTS) if _SKU110K_AVAILABLE else "yolov8n.pt",
)   # used by "coco"/"custom"

# YOLO-World open-vocabulary settings.
YOLO_WORLD_WEIGHTS = _env("YOLO_WORLD_WEIGHTS", "yolov8s-worldv2.pt")
# Prompt list for the open-vocabulary generalist. Broad retail/packaging terms
# on purpose: the fallback must fire on whatever is held up (a battery, a
# cup noodle, a bottle), and a missed prompt means a missed box. Measured on
# the repo's 17 reference photos, the extra terms below take fallback coverage
# from 13/17 to 16/17.
YOLO_WORLD_PROMPTS = _env(
    "YOLO_WORLD_PROMPTS",
    "product,package,snack bag,pouch,box,bottle,can,carton,chocolate bar,"
    "bag of chips,candy,battery,jar,tube,cylinder,container,cup,cup noodle,"
    "tin,object",
)
# The generalist scores differently from the trained specialist: it needs a
# lower confidence (0.05) and standard NMS IoU (0.5, not the specialist's 0.3).
YOLO_WORLD_CONF = float(_env("YOLO_WORLD_CONF", "0.05"))
YOLO_WORLD_IOU = float(_env("YOLO_WORLD_IOU", "0.5"))

# ---- "auto" regime switch (see AutoDetector) -------------------------------
# The specialist result is trusted (dense shelf) only when it clears ALL of:
#   coverage >= AUTO_COVERAGE_MIN  AND  largest box >= AUTO_MIN_AREA
# Measured (imgsz 960, conf 0.05-0.25): the 4K dense shelf covers 0.20-0.44 of
# the frame with a largest box of 2.0-3.5%; every close-up product covers at
# most 0.16 and its largest box is under 1% (usually under 0.4%). The margins
# are wide, so the switch is stable rather than knife-edge.
AUTO_COVERAGE_MIN = float(_env("AUTO_COVERAGE_MIN", "0.15"))
AUTO_MIN_AREA = float(_env("AUTO_MIN_AREA", "0.01"))
# On an untrusted frame, only let the generalist take over when it found a
# genuinely large object (a held-up product fills 10-70% of the frame); this
# keeps a sparse/dim shelf from losing its specialist boxes to a weak guess.
AUTO_SWITCH_AREA = float(_env("AUTO_SWITCH_AREA", "0.05"))
# Dense retail shelves need a LARGE inference size (products become tiny when a
# 4K photo is squashed to 640) and a LOW confidence (open-vocab prompts score
# low). These defaults are the single biggest reason detection "finds nothing".
DETECT_IMGSZ = int(_env("DETECT_IMGSZ", "1280"))       # try 1920 for 4K photos
# Default conf follows the detector: trained SKU-110K weights are well
# calibrated (0.2 keeps the UI clean); YOLO-World scores low, needs 0.05.
# Under "auto" the slider drives the specialist; the generalist always runs at
# its own YOLO_WORLD_CONF.
DETECT_CONF = float(_env(
    "DETECT_CONF", "0.05" if DETECTOR == "world" else "0.2"
))
# NMS IoU: 0.5 is standard, but the trained single-class model emits box
# pairs offset by ~half a box on the same product (IoU ~0.4-0.5) — 0.5 keeps
# both and you see two boxes on one item. 0.3 suppresses them without
# merging genuinely adjacent facings. (YOLO-World uses YOLO_WORLD_IOU.)
DETECT_IOU = float(_env(
    "DETECT_IOU", "0.5" if DETECTOR == "world" else "0.3"
))
DETECT_MAX_DET = int(_env("DETECT_MAX_DET", "1000"))   # shelves are dense
# If True, every detected box is treated as a product region regardless of its
# COCO class. Correct behaviour for a single-class SKU-110K model too.
DETECT_CLASS_AGNOSTIC = _env("DETECT_CLASS_AGNOSTIC", "1") == "1"

# ---- Live tracking + temporal SKU fusion -----------------------------------
# The live recognizer tracks boxes across frames and fuses SKU votes over
# time instead of labeling each box from a single first-frame CLIP lookup:
#   SHELF_TRACKER selects the multi-object tracker used for box tracking:
#   "bytetrack.yaml" (default, Kalman-filtered, survives brief misses) or
#   "botsort.yaml" (adds ReID + camera-motion compensation; needs the
#   `boxmot` package). "off" falls back to the legacy greedy-IoU matcher.
TRACKER = _env("SHELF_TRACKER", "bytetrack.yaml")
# A newly-appeared track starts UNCONFIRMED (it shows as "scanning" and can
# never be grasped). It is confirmed once at least CONFIRM_FRAMES votes exist
# and the winning SKU holds >= VOTE_MIN of the last VOTE_WINDOW votes.
VOTE_WINDOW = int(_env("VOTE_WINDOW", "5"))
VOTE_MIN = int(_env("VOTE_MIN", "3"))
CONFIRM_FRAMES = int(_env("CONFIRM_FRAMES", "3"))
# Hysteresis: a new/different SKU must clear MATCH_THRESHOLD to lock in, but
# the currently-held SKU only has to stay above MATCH_THRESHOLD_KEEP. This
# stops label flapping when the score oscillates around the threshold.
MATCH_THRESHOLD_KEEP = float(_env("MATCH_THRESHOLD_KEEP", "0.55"))
# Re-verification: confirmed/lived tracks are re-embedded every
# REEMBED_INTERVAL frames (round-robin, at most REEMBED_MAX_PER_FRAME boxes
# per frame), so a bad first-frame label self-corrects instead of being
# locked in for the lifetime of the track.
REEMBED_INTERVAL = int(_env("REEMBED_INTERVAL", "30"))
REEMBED_MAX_PER_FRAME = int(_env("REEMBED_MAX_PER_FRAME", "4"))
# Hard cap on how many BOXES reach CLIP per live frame (all sources: new,
# unconfirmed, re-verified). Unconfirmed tracks are re-embedded every frame by
# design, and on a shelf whose products are not in the catalog they never
# confirm — without a cap every box re-embeds every frame and the stream
# collapses to <1 FPS. Boxes that miss the cap are picked up on a later frame;
# each track keeps its own vote history, so confirmation is delayed, not lost.
EMBED_MAX_PER_FRAME = int(_env("EMBED_MAX_PER_FRAME", "8"))

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

# Multi-scale query crops (test-time augmentation). The catalog stores the
# REGISTERED PHOTO while the query is a tight detector box, so a tight crop
# loses the context the reference has and the true match's cosine drops.
# Every box is embedded at each scale and the best score per SKU wins, which
# can only raise the true match. Measured over the 17 reference photos
# (scripts/eval_match.py): mean true-match score 0.751 -> 0.824 and 15/17 clear
# MATCH_THRESHOLD vs ~9/17 for the tight crop alone. Scale 1.0 is the plain
# crop. Costs one extra CLIP embedding per box per scale beyond the first;
# set MATCH_SCALES=1.0 to disable.
MATCH_SCALES = tuple(
    float(s) for s in _env("MATCH_SCALES", "1.0,1.2").split(",") if s.strip()
) or (1.0,)
# CLIP images per forward pass. The photo tab can produce hundreds of crops
# from a 4K shelf (x MATCH_SCALES); chunking bounds peak memory instead of
# stacking them all into one tensor.
EMBED_BATCH_SIZE = int(_env("EMBED_BATCH_SIZE", "32"))

# ---- Device ----------------------------------------------------------------
DEVICE = _env("SHELF_DEVICE", "auto")   # auto | cpu | cuda | mps

# ---- Eye-in-hand RGB-D (RealSense + robotic arm) ---------------------------
# See docs/plans/2026-08-27-eye-in-hand-design.md. Used only by camera.py /
# pose3d.py / rgbd_live.py; the 2D web demo does not touch any of this.
RS_WIDTH = int(_env("RS_WIDTH", "1280"))
RS_HEIGHT = int(_env("RS_HEIGHT", "720"))
RS_FPS = int(_env("RS_FPS", "30"))
# Depth beyond this is treated as invalid (RealSense noise grows with range;
# grasp distances for eye-in-hand are well under 1 m).
RS_DEPTH_M_MAX = float(_env("RS_DEPTH_M_MAX", "2.5"))
# Solved hand-eye transform T_ee_cam (4x4, as rotvec+translation JSON),
# written by scripts/calibrate_handeye.py.
HANDEYE_PATH = Path(_env("HANDEYE_PATH", str(DATA_DIR / "handeye.json")))
# Center shrink factor of a detection box when sampling depth for 3D
# localisation: the box always contains background/corners, so read depth
# only from its central TARGET3D_ROI_SHRINK x TARGET3D_ROI_SHRINK region.
TARGET3D_ROI_SHRINK = float(_env("TARGET3D_ROI_SHRINK", "0.4"))


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
