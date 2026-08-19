"""CLIP image embedder (open_clip).

Turns a product crop into a normalized feature vector. The same model is used
both to register reference photos and to encode shelf crops at query time, so
the two live in the same vector space and cosine similarity is meaningful.

Note: zero-shot CLIP is only a starting point for fine-grained SKU matching.
For production, fine-tune the visual encoder on retail data (see README).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from . import config


class Embedder:
    def __init__(
        self,
        model_name: str | None = None,
        pretrained: str | None = None,
        device: str | None = None,
    ):
        import open_clip
        import torch

        self.torch = torch
        self.device = device or config.pick_device()
        self.model_name = model_name or config.CLIP_MODEL
        self.pretrained = pretrained or config.CLIP_PRETRAINED

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model.eval().to(self.device)
        self.dim = int(self.model.visual.output_dim)

    @property
    def torch_no_grad(self):
        return self.torch.no_grad()

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        """Return an (N, dim) float32 array of L2-normalized embeddings."""
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)

        batch = self.torch.stack(
            [self.preprocess(img.convert("RGB")) for img in images]
        ).to(self.device)

        with self.torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.detach().cpu().float().numpy().astype(np.float32)

    def embed_one(self, image: Image.Image) -> np.ndarray:
        return self.embed([image])[0]
