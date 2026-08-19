"""End-to-end shelf recognition pipeline.

register_product(): reference photos -> embeddings -> catalog (no retraining).
recognize():        shelf photo -> detect -> crop -> embed -> retrieve SKU.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image

from . import config
from .database import Catalog, Product
from .detector import Box, Detector
from .embedder import Embedder


@dataclass
class Recognition:
    box: Box
    sku_id: int | None
    name: str
    score: float
    is_match: bool
    track_id: int | None = None   # set by the live tracker; None in photo mode

    @property
    def label(self) -> str:
        if self.is_match:
            return f"{self.name} ({self.score:.2f})"
        return f"Unknown ({self.score:.2f})"


class ShelfPipeline:
    def __init__(self, detector: Detector | None = None,
                 embedder: Embedder | None = None,
                 catalog: Catalog | None = None):
        self.detector = detector or Detector()
        self.embedder = embedder or Embedder()
        # Keep FAISS dim in sync with whatever CLIP model is loaded.
        self.catalog = catalog or Catalog(dim=self.embedder.dim)

    # ---- registration ------------------------------------------------------
    def register_product(
        self,
        images: list[Image.Image],
        name: str,
        barcode: str = "",
        price: float = 0.0,
        category: str = "",
    ) -> int:
        if not images:
            raise ValueError("At least one reference image is required.")
        if not name.strip():
            raise ValueError("Product name is required.")

        embeddings = self.embedder.embed(images)
        sku_id = self.catalog.add_product(
            name=name.strip(),
            embeddings=embeddings,
            barcode=barcode.strip(),
            price=float(price or 0.0),
            category=category.strip(),
        )
        self._save_reference_crops(sku_id, images)
        return sku_id

    def _save_reference_crops(self, sku_id: int, images: list[Image.Image]) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        for i, img in enumerate(images):
            path = config.CROP_DIR / f"sku{sku_id:04d}_{stamp}_{i}.jpg"
            img.convert("RGB").save(path, quality=90)

    def reference_photos(self, sku_id: int) -> list[str]:
        """Filenames of the reference photos saved for a SKU (may be empty)."""
        return sorted(p.name for p in config.CROP_DIR.glob(f"sku{sku_id:04d}_*.jpg"))

    def delete_product(self, sku_id: int) -> None:
        """Delete a SKU: vectors + metadata row AND its saved reference photos."""
        self.catalog.delete_product(sku_id)
        for name in self.reference_photos(sku_id):
            (config.CROP_DIR / name).unlink(missing_ok=True)

    # ---- recognition -------------------------------------------------------
    def match_boxes(
        self,
        image: Image.Image,
        boxes: list[Box],
        threshold: float | None = None,
    ) -> list[Recognition]:
        """Embed each box crop and retrieve its best catalog match.

        Split out from recognize() so the live/webcam tracker can run the
        expensive CLIP step only on newly-appeared boxes.
        """
        threshold = config.MATCH_THRESHOLD if threshold is None else threshold
        if not boxes:
            return []
        crops = [b.crop(image) for b in boxes]
        embeddings = self.embedder.embed(crops)
        hits = self.catalog.search(embeddings, topk=config.SEARCH_TOPK)

        results: list[Recognition] = []
        for box, hit in zip(boxes, hits):
            if hit:
                sku_id, score = hit[0]
                product = self.catalog.get(sku_id)
                name = product.name if product else "?"
                is_match = score >= threshold and product is not None
                results.append(
                    Recognition(box, sku_id if is_match else None,
                                name if is_match else "", score, is_match)
                )
            else:
                results.append(Recognition(box, None, "", 0.0, False))
        return results

    def recognize(
        self,
        shelf_image: Image.Image,
        conf: float | None = None,
        threshold: float | None = None,
        imgsz: int | None = None,
    ) -> list[Recognition]:
        threshold = config.MATCH_THRESHOLD if threshold is None else threshold
        image = shelf_image.convert("RGB")

        boxes = self.detector.detect(image, conf=conf, imgsz=imgsz)
        return self.match_boxes(image, boxes, threshold)

    # ---- convenience -------------------------------------------------------
    def catalog_products(self) -> list[Product]:
        return self.catalog.list_products()
