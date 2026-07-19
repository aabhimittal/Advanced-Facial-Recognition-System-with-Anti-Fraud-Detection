"""Identity gallery + cosine-distance matching."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class FaceMatcher:
    """A simple enrol/identify gallery keyed by name.

    Distance is cosine distance in ``[0, 2]`` (0 = identical direction). Multiple
    enrolments per identity are averaged into a prototype.
    """

    def __init__(self, match_threshold: float = 0.6):
        self.match_threshold = match_threshold
        self._gallery: Dict[str, List[np.ndarray]] = {}

    def enroll(self, name: str, embedding: np.ndarray) -> None:
        self._gallery.setdefault(name, []).append(_unit(np.asarray(embedding, dtype=np.float64)))

    def prototypes(self) -> Dict[str, np.ndarray]:
        return {name: _unit(np.mean(vs, axis=0)) for name, vs in self._gallery.items()}

    def identify(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Return ``(name, distance)`` for the best match, or ``(None, best_dist)``.

        ``name`` is ``None`` when the nearest identity is farther than the
        threshold (an unknown face).
        """
        q = _unit(np.asarray(embedding, dtype=np.float64))
        best_name, best_dist = None, float("inf")
        for name, proto in self.prototypes().items():
            dist = 1.0 - float(np.dot(q, proto))
            if dist < best_dist:
                best_name, best_dist = name, dist
        if best_name is not None and best_dist <= self.match_threshold:
            return best_name, best_dist
        return None, best_dist


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
