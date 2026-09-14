"""YOLO shelf detector.

Detects product *facings* on a shelf image and returns their bounding boxes.
We do not classify the SKU here — that is the job of the embedding + retrieval
stage. This mirrors the industry-standard SKU-110K approach where detection is
a single "product" class.

Two backends exist and they fail in opposite regimes (measured on this repo's
own images — see scripts/verify_detection.py):

* ``custom``  SKU-110K-trained YOLO. Excellent on dense shelves of small
  facings, but the whole training prior is "many small products in a grid".
  A single product held close to the webcam — the normal Live-tab pose — is
  out of distribution: at default confidence it finds *nothing*, and at low
  confidence it hallucinates hundreds of tiny boxes scattered over the label.
* ``world``   YOLO-World open-vocabulary. Finds one clean, high-confidence box
  around a close-up product, but merges/ignores facings on dense shelves, so
  it is much weaker there.

``auto`` (the default when a trained specialist is present) runs the
specialist, checks whether its output actually has the shape of a dense shelf,
and only then falls back to the open-vocabulary generalist. Dense shelves keep
their tight specialist boxes; close-up products get a usable bounding box
instead of none.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import config


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def crop(self, image: Image.Image) -> Image.Image:
        return image.crop(self.xyxy)


# ---- regime heuristics (used by AutoDetector) ------------------------------

def coverage_frac(boxes: list[Box], width: int, height: int,
                  grid: int = 96) -> float:
    """Fraction of the frame covered by the UNION of ``boxes``.

    Measured on a coarse grid so a 4K frame with hundreds of boxes stays
    cheap. This is the feature that separates the two regimes: a dense shelf's
    boxes tile the products and cover a large part of the frame, while the
    specialist's sub-detections on one close-up product cover almost nothing.
    """
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    m = np.zeros((grid, grid), dtype=bool)
    for b in boxes:
        x0 = max(0, int(b.x1 / width * grid))
        x1 = min(grid, int(np.ceil(b.x2 / width * grid)))
        y0 = max(0, int(b.y1 / height * grid))
        y1 = min(grid, int(np.ceil(b.y2 / height * grid)))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
    return float(m.mean())


def max_area_frac(boxes: list[Box], width: int, height: int) -> float:
    """Largest box as a fraction of the frame area (0.0 when empty)."""
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    return max((b.x2 - b.x1) * (b.y2 - b.y1) for b in boxes) / float(
        width * height)


def padded_crop(image: Image.Image, box: Box, scale: float = 1.0) -> Image.Image:
    """Crop ``box``, optionally grown symmetrically about its centre.

    Retrieval compares a tight detector box against the *registered full
    photo*; growing the box restores some of the reference's context and lifts
    the true match's cosine (ALKALINE 0.59 -> 0.67 at scale 1.2 — see
    scripts/eval_match.py). Scale 1.0 is the plain crop.
    """
    if scale <= 1.0:
        return box.crop(image)
    cx, cy = (box.x1 + box.x2) / 2.0, (box.y1 + box.y2) / 2.0
    w, h = (box.x2 - box.x1) * scale, (box.y2 - box.y1) * scale
    x1 = max(0, int(round(cx - w / 2)))
    y1 = max(0, int(round(cy - h / 2)))
    x2 = min(image.width, int(round(cx + w / 2)))
    y2 = min(image.height, int(round(cy + h / 2)))
    if x2 <= x1 or y2 <= y1:      # degenerate (box at the frame edge)
        return box.crop(image)
    return image.crop((x1, y1, x2, y2))


class Detector:
    """One ultralytics model behind the common ``detect`` interface."""

    def __init__(
        self,
        weights: str | None = None,
        device: str | None = None,
        detector: str | None = None,
        prompts: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ):
        self.device = device or config.pick_device()
        self.kind = detector or config.DETECTOR
        if self.kind == "auto":
            raise ValueError(
                'Detector does not implement "auto"; use '
                "build_detector() (or AutoDetector) for the two-model "
                "specialist + fallback path."
            )
        self.conf = config.DETECT_CONF if conf is None else float(conf)
        self.iou = config.DETECT_IOU if iou is None else float(iou)
        self.prompts: list[str] = []

        if self.kind == "world":
            from ultralytics import YOLOWorld  # lazy import
            self.weights = weights or config.YOLO_WORLD_WEIGHTS
            self.model = YOLOWorld(self.weights)
            self.prompts = [
                p.strip()
                for p in (prompts or config.YOLO_WORLD_PROMPTS).split(",")
                if p.strip()
            ]
            self.model.set_classes(self.prompts)
        else:  # "coco" or "custom" — both are plain YOLO weights
            from ultralytics import YOLO  # lazy import
            self.weights = weights or config.YOLO_WEIGHTS
            self.model = YOLO(self.weights)

    def detect(self, image: Image.Image, conf: float | None = None,
               imgsz: int | None = None, iou: float | None = None) -> list[Box]:
        results = self.model.predict(
            source=image,
            imgsz=imgsz if imgsz is not None else config.DETECT_IMGSZ,
            conf=self.conf if conf is None else conf,
            iou=self.iou if iou is None else iou,
            max_det=config.DETECT_MAX_DET,
            # Merge boxes that different prompts ("package"/"box"/"pouch") fire on
            # the same object; without this NMS is per-class and you get stacks
            # of overlapping duplicate boxes.
            agnostic_nms=config.DETECT_CLASS_AGNOSTIC,
            device=self.device,
            verbose=False,
        )
        boxes: list[Box] = []
        if not results:
            return boxes
        r = results[0]
        return _rows_to_boxes(r.boxes, image.size)

    def warm(self) -> None:
        """No-op: this backend is already loaded in __init__."""

    @property
    def active_backend(self) -> str:
        """Which backend produced the boxes: "custom" | "world" | "coco"."""
        return self.kind


def _rows_to_boxes(rows, size: tuple[int, int]) -> list[Box]:
    """Ultralytics result boxes -> clamped, validated Box list."""
    w, h = size
    boxes: list[Box] = []
    for b in rows:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        # clamp to image bounds
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(0, min(int(round(x2)), w))
        y2 = max(0, min(int(round(y2)), h))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(Box(x1, y1, x2, y2, float(b.conf[0])))
    return boxes


class AutoDetector:
    """Dense-shelf specialist + open-vocabulary close-up fallback.

    Decision per frame (thresholds in config, all env-overridable):

        coverage  = fraction of the frame covered by the specialist's boxes
        max_area  = largest specialist box as a fraction of the frame
        trusted   = boxes and coverage >= AUTO_COVERAGE_MIN
                    and max_area >= AUTO_MIN_AREA

    Untrusted frames run the generalist; its result replaces the specialist's
    only when it actually found a LARGE object (>= AUTO_SWITCH_AREA), which is
    what a held-up product looks like and what the specialist cannot represent.
    Otherwise the specialist result is kept. With DETECTOR=custom or
    DETECTOR=world this class is bypassed entirely.

    The generalist is loaded lazily: installs that only ever see dense shelves
    never pay YOLO-World's startup cost or pull its text-encoder dependency.
    ``warm()`` forces it (app.py calls it at boot so the first close-up frame
    is not stalled by a multi-second load).
    """

    kind = "auto"

    def __init__(self, specialist: Detector | None = None,
                 generalist: Detector | None = None):
        self.specialist = specialist or Detector(detector="custom")
        self._generalist = generalist
        self._lock = threading.Lock()
        self._generalist_error: str | None = None
        self.device = self.specialist.device
        self.weights = self.specialist.weights
        self.conf = self.specialist.conf
        self.iou = self.specialist.iou
        # prompts are a generalist property; expose them for the boot banner
        self.prompts = [
            p.strip() for p in config.YOLO_WORLD_PROMPTS.split(",") if p.strip()
        ]
        # Diagnostics from the most recent detect() call (which path ran).
        self.last_path = "specialist"

    # ---- generalist lifecycle ---------------------------------------------
    @property
    def generalist(self) -> Detector | None:
        if self._generalist is None and self._generalist_error is None:
            with self._lock:
                if self._generalist is None and self._generalist_error is None:
                    try:
                        self._generalist = Detector(
                            detector="world",
                            conf=config.YOLO_WORLD_CONF,
                            iou=config.YOLO_WORLD_IOU,
                        )
                    except Exception as e:  # noqa: BLE001 - keep the demo alive
                        self._generalist_error = repr(e)
                        print(f"[detector] open-vocab fallback unavailable "
                              f"({e!r}); dense-shelf specialist only")
        return self._generalist

    def warm(self) -> None:
        """Preload the generalist so the first fallback frame is not delayed."""
        self.generalist

    @property
    def active_backend(self) -> str:
        """Which backend produced the LAST detect(): "custom" or "world".

        The per-frame answer to "am I on the trained model or on YOLO-World
        right now?" — surfaced by the Live tab and debug_detect.
        """
        return "world" if self.last_path == "fallback" else "custom"

    # ---- detection ---------------------------------------------------------
    def detect(self, image: Image.Image, conf: float | None = None,
               imgsz: int | None = None) -> list[Box]:
        conf = self.specialist.conf if conf is None else float(conf)
        boxes = self.specialist.detect(image, conf=conf, imgsz=imgsz)
        w, h = image.size
        if self._trusted(boxes, w, h):
            self.last_path = "specialist"
            return boxes

        generalist = self.generalist
        if generalist is None:
            self.last_path = "specialist"
            return boxes
        gboxes = generalist.detect(image, conf=config.YOLO_WORLD_CONF,
                                   imgsz=imgsz)
        if gboxes and max_area_frac(gboxes, w, h) >= config.AUTO_SWITCH_AREA:
            self.last_path = "fallback"
            return gboxes
        self.last_path = "specialist"
        return boxes

    @staticmethod
    def _trusted(boxes: list[Box], w: int, h: int) -> bool:
        if not boxes:
            return False
        if max_area_frac(boxes, w, h) < config.AUTO_MIN_AREA:
            return False
        return coverage_frac(boxes, w, h) >= config.AUTO_COVERAGE_MIN


def build_detector(kind: str | None = None) -> Detector | AutoDetector:
    """Construct the detector named by ``kind`` (default: config.DETECTOR).

    "auto" -> specialist + fallback; anything else -> a single Detector.
    "auto" needs specialist weights to exist; without them it degrades to
    "world" instead of silently using a COCO model as the specialist.
    """
    kind = (kind or config.DETECTOR).strip().lower()
    if kind == "auto":
        from pathlib import Path
        if not Path(config.YOLO_WEIGHTS).exists():
            print(f"[detector] DETECTOR=auto needs specialist weights; "
                  f"{config.YOLO_WEIGHTS!r} not found -> using 'world'")
            return Detector(detector="world")
        return AutoDetector()
    return Detector(detector=kind)
