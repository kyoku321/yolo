"""Product catalog: SQLite metadata + NumPy vector store.

Each SKU has one row of metadata (name, barcode, price, category) and one or
more reference embeddings in the vector store. Vectors are tagged with their
sku_id, so a SKU can own several vectors and can be deleted cleanly.

Why NumPy and not FAISS here? On macOS `faiss-cpu` and `torch` each bundle
their own OpenMP runtime; loading both in one (threaded Gradio) process leads
to "OMP: Error #15" aborts and intermittent segfaults. At demo scale the index
is just a cosine matrix-multiply over a few thousand normalized vectors, which
NumPy does in well under a millisecond — no native OpenMP conflict. For real
scale (100k+ SKUs) swap this store for FAISS or Milvus; the public API below
(add / search / delete) is intentionally the same shape.

Registering a product = insert a row + add its reference embeddings. No model
retraining is involved, which is the whole point of the retrieval approach.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np

from . import config


@dataclass
class Product:
    sku_id: int
    name: str
    barcode: str
    price: float
    category: str
    n_refs: int
    created_at: str

    def as_dict(self) -> dict:
        return asdict(self)


class VectorStore:
    """Exact cosine nearest-neighbour over L2-normalized vectors."""

    def __init__(self, dim: int, path):
        self.dim = dim
        self.path = str(path)
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids = np.zeros((0,), dtype=np.int64)
        self._load()

    def _load(self) -> None:
        import os

        if os.path.exists(self.path):
            data = np.load(self.path)
            v, i = data["vectors"], data["ids"]
            if v.ndim == 2 and v.shape[0] > 0:
                # Adopt the stored dimension (e.g. after switching embedders the
                # requested dim may differ; the file is the source of truth).
                self.dim = int(v.shape[1])
                self.vectors, self.ids = v.astype(np.float32), i.astype(np.int64)

    def save(self) -> None:
        np.savez(self.path, vectors=self.vectors, ids=self.ids)

    @property
    def ntotal(self) -> int:
        return int(self.vectors.shape[0])

    def add(self, embeddings: np.ndarray, sku_id: int) -> None:
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        ids = np.full((emb.shape[0],), sku_id, dtype=np.int64)
        self.vectors = np.vstack([self.vectors, emb]) if self.ntotal else emb
        self.ids = np.concatenate([self.ids, ids])

    def remove(self, sku_id: int) -> None:
        keep = self.ids != sku_id
        self.vectors = self.vectors[keep]
        self.ids = self.ids[keep]

    def clear(self) -> None:
        self.vectors = np.zeros((0, self.dim), dtype=np.float32)
        self.ids = np.zeros((0,), dtype=np.int64)

    def search(self, queries: np.ndarray, topk: int):
        """For each query, return [(sku_id, score), ...] best-first, one entry
        per SKU (best-scoring vector wins for a multi-reference SKU)."""
        n = queries.shape[0]
        if self.ntotal == 0 or n == 0:
            return [[] for _ in range(n)]

        sims = queries.astype(np.float32) @ self.vectors.T  # (n, ntotal), cosine
        out = []
        for row in sims:
            best: dict[int, float] = {}
            for vec_idx, sku_id in enumerate(self.ids):
                s = float(row[vec_idx])
                sid = int(sku_id)
                if sid not in best or s > best[sid]:
                    best[sid] = s
            ranked = sorted(best.items(), key=lambda kv: -kv[1])[:topk]
            out.append([(sid, s) for sid, s in ranked])
        return out


class Catalog:
    def __init__(
        self,
        db_path=config.DB_PATH,
        index_path=config.INDEX_PATH,
        dim: int = config.EMBED_DIM,
    ):
        config.ensure_dirs()
        self.db_path = str(db_path)
        self.dim = dim

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self.store = VectorStore(dim=dim, path=index_path)

    # ---- schema ------------------------------------------------------------
    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS products (
                sku_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                barcode    TEXT DEFAULT '',
                price      REAL DEFAULT 0.0,
                category   TEXT DEFAULT '',
                n_refs     INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    # ---- writes ------------------------------------------------------------
    def add_product(
        self,
        name: str,
        embeddings: np.ndarray,
        barcode: str = "",
        price: float = 0.0,
        category: str = "",
    ) -> int:
        """Insert a SKU and its reference embeddings. Returns the new sku_id."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embeddings must be (N, {self.dim}), got {embeddings.shape}"
            )
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self.conn.execute(
            "INSERT INTO products (name, barcode, price, category, n_refs, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (name, barcode, price, category, int(embeddings.shape[0]), now),
        )
        sku_id = int(cur.lastrowid)
        self.store.add(embeddings, sku_id)
        self.conn.commit()
        self.store.save()
        return sku_id

    def delete_product(self, sku_id: int) -> None:
        self.store.remove(sku_id)
        self.conn.execute("DELETE FROM products WHERE sku_id = ?", (sku_id,))
        self.conn.commit()
        self.store.save()

    def reset(self) -> None:
        self.conn.execute("DELETE FROM products")
        self.conn.commit()
        self.store.clear()
        self.store.save()

    # ---- reads -------------------------------------------------------------
    def get(self, sku_id: int) -> Product | None:
        row = self.conn.execute(
            "SELECT * FROM products WHERE sku_id = ?", (sku_id,)
        ).fetchone()
        return Product(**dict(row)) if row else None

    def list_products(self) -> list[Product]:
        rows = self.conn.execute(
            "SELECT * FROM products ORDER BY sku_id"
        ).fetchall()
        return [Product(**dict(r)) for r in rows]

    @property
    def n_vectors(self) -> int:
        return self.store.ntotal

    def search(self, embeddings: np.ndarray, topk: int = config.SEARCH_TOPK):
        return self.store.search(embeddings, topk)

    def close(self) -> None:
        self.conn.close()
