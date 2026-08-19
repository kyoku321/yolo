"""Supermarket shelf product recognition demo.

Pipeline: YOLO detection -> crop -> CLIP embedding -> FAISS retrieval.
New products are registered by embedding a few reference photos, so adding a
SKU never requires retraining a model.
"""

__all__ = ["config"]
