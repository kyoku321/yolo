"""Headless end-to-end check: register a product, then recognize it on a
synthetic 'shelf' built from the same reference image.

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shelf_demo.embedder import Embedder          # noqa: E402
from shelf_demo.database import Catalog           # noqa: E402


def main() -> None:
    print("Loading CLIP embedder ...")
    emb = Embedder()
    print(f"  model={emb.model_name} pretrained={emb.pretrained} "
          f"dim={emb.dim} device={emb.device}")

    cat = Catalog(dim=emb.dim,
                  db_path="data/_smoke.sqlite",
                  index_path="data/_smoke.npz")
    cat.reset()

    # Two distinct synthetic products (solid color swatches).
    red = Image.new("RGB", (224, 224), (200, 30, 30))
    blue = Image.new("RGB", (224, 224), (30, 30, 200))

    id_red = cat.add_product("Red Can", emb.embed([red]), category="test")
    id_blue = cat.add_product("Blue Box", emb.embed([blue]), category="test")
    print(f"Registered SKUs: red=#{id_red} blue=#{id_blue} "
          f"vectors={cat.n_vectors}")

    # Query with a near-duplicate red crop.
    query = Image.new("RGB", (224, 224), (190, 40, 40))
    hits = cat.search(emb.embed([query]), topk=2)[0]
    print("Query (reddish) top hits:")
    for sku_id, score in hits:
        p = cat.get(sku_id)
        print(f"  #{sku_id} {p.name:10s} score={score:.3f}")

    best_id = hits[0][0]
    assert best_id == id_red, "expected red product to win"
    print("\n✅ Smoke test passed: retrieval returns the correct SKU.")

    cat.close()


if __name__ == "__main__":
    main()
