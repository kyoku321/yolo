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
from .detector import AutoDetector, Box, Detector, build_detector, padded_crop
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


def rank_boxes(
    catalog: Catalog,
    embedder: Embedder,
    image: Image.Image,
    boxes: list[Box],
    scales: tuple[float, ...] | None = None,
    topk: int | None = None,
) -> list[list[tuple[int, float]]]:
    """Rank catalog SKUs for each box, best-first, using multi-scale crops.

    The catalog stores the REGISTERED PHOTO while a query box is a tight
    detector crop, so a tight crop alone loses the reference's context and
    under-scores the true match. Every box is therefore embedded at each scale
    in MATCH_SCALES (1.0 = the plain crop) and the best score per SKU wins —
    test-time augmentation, so it can only raise the true match.

    Returns one list of (sku_id, score) per input box, at most ``topk`` long.
    """
    if not boxes:
        return []
    scales = tuple(scales) if scales else tuple(config.MATCH_SCALES)
    topk = config.SEARCH_TOPK if topk is None else topk
    crops = [padded_crop(image, b, s) for b in boxes for s in scales]
    embeddings = embedder.embed(crops)
    hits = catalog.search(embeddings, topk=topk)

    ns = len(scales)
    ranked_all: list[list[tuple[int, float]]] = []
    for i in range(len(boxes)):
        best: dict[int, float] = {}
        for hit in hits[i * ns:(i + 1) * ns]:
            for sku_id, score in hit:
                if sku_id not in best or score > best[sku_id]:
                    best[sku_id] = score
        ranked_all.append(sorted(best.items(), key=lambda kv: -kv[1])[:topk])
    return ranked_all


class ShelfPipeline:
    def __init__(self, detector: Detector | AutoDetector | None = None,
                 embedder: Embedder | None = None,
                 catalog: Catalog | None = None):
        self.detector = detector or build_detector()
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
        expensive CLIP step only on newly-appeared boxes. Ranking itself is in
        rank_boxes() so the debug tooling scores boxes exactly like the app.
        """
        threshold = config.MATCH_THRESHOLD if threshold is None else threshold
        if not boxes:
            return []
        ranked_all = rank_boxes(self.catalog, self.embedder, image, boxes)

        results: list[Recognition] = []
        for box, ranked in zip(boxes, ranked_all):
            if ranked:
                sku_id, score = ranked[0]
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
