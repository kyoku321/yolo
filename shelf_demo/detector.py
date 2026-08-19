"""YOLO shelf detector.

Detects product *facings* on a shelf image and returns their bounding boxes.
We do not classify the SKU here — that is the job of the embedding + retrieval
stage. This mirrors the industry-standard SKU-110K approach where detection is
a single "product" class.
"""
from __future__ import annotations

from dataclasses import dataclass

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


class Detector:
    def __init__(
        self,
        weights: str | None = None,
        device: str | None = None,
        detector: str | None = None,
        prompts: str | None = None,
    ):
        self.device = device or config.pick_device()
        self.kind = detector or config.DETECTOR
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
               imgsz: int | None = None) -> list[Box]:
        results = self.model.predict(
            source=image,
            imgsz=imgsz if imgsz is not None else config.DETECT_IMGSZ,
            conf=conf if conf is not None else config.DETECT_CONF,
            iou=config.DETECT_IOU,
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
        w, h = image.size
        for b in r.boxes:
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
