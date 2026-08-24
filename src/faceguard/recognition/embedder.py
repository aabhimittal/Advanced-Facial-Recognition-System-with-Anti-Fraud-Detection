"""Face embedding.

Two backends behind one interface:

* ``"arcface"`` — real state-of-the-art deep embeddings via InsightFace, used
  automatically when the ``full`` extra is installed. This is what you'd deploy.
* ``"numpy"`` — a lightweight, dependency-free descriptor (gradient-orientation
  histograms over a spatial grid, L2-normalised). It is *not* competitive with
  deep embeddings, but it is deterministic and lets the whole pipeline — and the
  test suite — run anywhere with just numpy. It is good enough to distinguish
  clearly different faces in demos.

The abstraction means the novel anti-fraud contribution does not depend on which
recogniser you plug in.
"""

from __future__ import annotations

import numpy as np

from ..liveness.base import to_grayscale


class FaceEmbedder:
    def __init__(self, backend: str = "auto", dim: int = 128):
        self.dim = dim
        self.backend = self._resolve(backend)
        self._arc = None
        if self.backend == "arcface":
            self._init_arcface()

    def _resolve(self, backend: str) -> str:
        if backend != "auto":
            return backend
        try:
            import insightface  # noqa: F401
            return "arcface"
        except Exception:  # noqa: BLE001 - optional backend: any failure falls back
            return "numpy"

    def _init_arcface(self) -> None:
        from insightface.app import FaceAnalysis

        self._arc = FaceAnalysis(name="buffalo_l")
        self._arc.prepare(ctx_id=-1)

    def embed(self, face: np.ndarray) -> np.ndarray:
        if self.backend == "arcface":
            faces = self._arc.get(np.asarray(face))
            if not faces:
                return np.zeros(512, dtype=np.float64)
            return _l2(faces[0].embedding.astype(np.float64))
        return self._numpy_embed(face)

    def _numpy_embed(self, face: np.ndarray, grid: int = 8, bins: int = 8) -> np.ndarray:
        """Grid histogram of gradient orientations — a compact, deterministic descriptor."""
        gray = to_grayscale(face)
        gy, gx = np.gradient(gray)
        mag = np.hypot(gx, gy)
        ang = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)  # -> [0,1)
        h, w = gray.shape
        ys = np.linspace(0, h, grid + 1, dtype=int)
        xs = np.linspace(0, w, grid + 1, dtype=int)
        feats = []
        for i in range(grid):
            for j in range(grid):
                m = mag[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                a = ang[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                hist, _ = np.histogram(a, bins=bins, range=(0, 1), weights=m)
                feats.append(hist)
        return _l2(np.concatenate(feats))


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
