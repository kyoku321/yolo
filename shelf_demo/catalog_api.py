"""JSON handlers backing the Catalog tab's custom HTML table.

Kept framework-free (plain dicts in / dicts out) so they are trivially
testable; app.py wraps them into FastAPI routes with asyncio.to_thread.
"""
from __future__ import annotations

from .pipeline import ShelfPipeline

PHOTOS_URL_PREFIX = "/catalog/photos/"   # StaticFiles mount, see app.py


def list_catalog(pipeline: ShelfPipeline) -> dict:
    rows = []
    for p in pipeline.catalog_products():
        rows.append({
            "sku_id": p.sku_id,
            "name": p.name,
            "category": p.category,
            "barcode": p.barcode,
            "price": p.price,
            "n_refs": p.n_refs,
            "created_at": p.created_at,
            "photos": [PHOTOS_URL_PREFIX + n
                       for n in pipeline.reference_photos(p.sku_id)],
        })
    return {"ok": True, "skus": rows, "n_vectors": pipeline.catalog.n_vectors}


def delete_sku(pipeline: ShelfPipeline, payload: dict) -> dict:
    try:
        sku_id = int(payload.get("sku_id"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "invalid sku_id"}
    if pipeline.catalog.get(sku_id) is None:
        return {"ok": False, "error": f"SKU #{sku_id} not found"}
    pipeline.delete_product(sku_id)
    return {"ok": True, "deleted": sku_id}
