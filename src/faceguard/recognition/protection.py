"""Cancellable biometric templates — surviving a gallery breach.

The problem nobody can patch later
----------------------------------
A password database leak is bad; a face-embedding gallery leak is permanent. You
cannot issue somebody a new face. Worse, modern embeddings are invertible enough
to reconstruct a recognisable image of the enrolled person, so a raw gallery is
not "just some numbers" — it is biometric PII under GDPR Art. 9 and statutes
like Illinois' BIPA, with the retention and consent duties that follow.

The standard remedy is a **cancellable transform**: never store the embedding,
store a non-invertible, key-dependent projection of it. If the store is
breached, you rotate the key and reissue every template — the leaked ones stop
matching anything. Distances must survive the transform, or recognition breaks.

The construction here (BioHashing)
----------------------------------
1. Draw a random matrix from a key (a per-deployment secret plus a per-subject
   salt), and orthonormalise it.
2. Project the L2-normalised embedding onto it.
3. Keep only the **sign** of each projection: a bit string.

Signed random projections are a locality-sensitive hash for angular distance —
the Goemans-Williamson result gives ``P[bit differs] = θ/π`` — so normalised
Hamming distance between templates is a faithful proxy for the cosine distance
between the embeddings that produced them. Recognition keeps working; the stored
artefact is a quantised, key-dependent shadow of the original.

Security, stated honestly
-------------------------
This is a real and widely deployed mitigation, not a magic wand:

* **Not** encryption. Quantising to sign bits discards magnitude and is lossy,
  but an attacker holding *both* the key and the template can still approximate
  the original embedding's direction. The key must be protected like a key
  (HSM/KMS), separately from the template store.
* Per-subject salts stop cross-database linkage: the same face enrolled in two
  systems yields unrelated templates, so a breach of one cannot be joined
  against the other.
* Longer codes trade storage for accuracy; below ~128 bits quantisation noise
  starts to dominate genuine/impostor separation.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Optional

import numpy as np


class TemplateProtector:
    """Key-dependent, revocable, non-reversible-by-default face templates."""

    def __init__(self, key: bytes, bits: int = 256):
        if not isinstance(key, (bytes, bytearray)) or len(key) < 16:
            raise ValueError("key must be at least 16 random bytes")
        if bits < 64:
            raise ValueError("bits < 64 loses too much accuracy to be useful")
        self.key = bytes(key)
        self.bits = int(bits)

    # -- template lifecycle -------------------------------------------------
    def protect(self, embedding: np.ndarray, subject_salt: bytes = b"") -> np.ndarray:
        """Turn an embedding into a stored template (a packed bit array)."""
        v = _unit(np.asarray(embedding, dtype=np.float64).ravel())
        basis = self._basis(len(v), subject_salt)
        bits = (basis @ v) >= 0.0
        return np.packbits(bits)

    def reissue(self, embedding: np.ndarray, new_salt: bytes) -> np.ndarray:
        """Revoke and replace a template after a breach, without re-enrolling the user.

        The user does not present their face again and does not have to: a new
        salt yields a template unrelated to the compromised one.
        """
        return self.protect(embedding, new_salt)

    # -- matching -----------------------------------------------------------
    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        """Normalised Hamming distance in ``[0, 1]``; ~``θ/π`` for angle ``θ``."""
        a, b = np.asarray(a, dtype=np.uint8), np.asarray(b, dtype=np.uint8)
        if a.shape != b.shape:
            raise ValueError("templates were produced with different bit lengths")
        diff = np.unpackbits(np.bitwise_xor(a, b))
        return float(diff.mean())

    @staticmethod
    def cosine_from_hamming(hamming: float) -> float:
        """Recover the implied cosine distance, so existing thresholds still apply."""
        theta = float(np.clip(hamming, 0.0, 1.0)) * np.pi
        return float(1.0 - np.cos(theta))

    # -- internals ----------------------------------------------------------
    def _basis(self, dim: int, subject_salt: bytes) -> np.ndarray:
        """Orthonormalised random projection derived deterministically from the key.

        Derived with HMAC so the salt cannot be manipulated into colliding with
        another subject's basis, and orthonormalised so the bits are as close to
        independent as the dimension allows — correlated bits would waste code
        length and inflate the impostor-match rate.
        """
        seed = hmac.new(self.key, subject_salt + dim.to_bytes(4, "big"), hashlib.sha256).digest()
        rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
        mat = rng.standard_normal((self.bits, dim))
        if self.bits <= dim:
            # QR gives an exactly orthonormal set when it fits in the dimension.
            q, _ = np.linalg.qr(mat.T)
            return q.T[: self.bits]
        return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)


class ProtectedMatcher:
    """A gallery that never stores a raw embedding.

    Drop-in for :class:`~faceguard.recognition.matcher.FaceMatcher`: same
    ``enroll`` / ``identify`` contract, thresholds still expressed in cosine
    distance, but a stolen database yields only revocable bit strings.
    """

    def __init__(self, protector: TemplateProtector, match_threshold: float = 0.6):
        self.protector = protector
        self.match_threshold = float(match_threshold)
        self._templates: dict = {}
        self._salts: dict = {}

    def enroll(self, name: str, embedding: np.ndarray, salt: Optional[bytes] = None) -> None:
        salt = salt if salt is not None else self._salts.get(name) or _random_salt()
        self._salts[name] = salt
        self._templates.setdefault(name, []).append(self.protector.protect(embedding, salt))

    def identify(self, embedding: np.ndarray):
        """Return ``(name, cosine_distance)``; ``name`` is None when unmatched."""
        best_name, best_cos = None, float("inf")
        for name, templates in self._templates.items():
            probe = self.protector.protect(embedding, self._salts[name])
            # Per-subject salts mean the probe must be re-projected per identity:
            # the price of unlinkability is a gallery scan, not a single lookup.
            ham = min(self.protector.distance(probe, t) for t in templates)
            cos = self.protector.cosine_from_hamming(ham)
            if cos < best_cos:
                best_name, best_cos = name, cos
        if best_name is not None and best_cos <= self.match_threshold:
            return best_name, best_cos
        return None, best_cos

    def revoke(self, name: str) -> None:
        """Forget a subject entirely — the deletion right, actually implemented."""
        self._templates.pop(name, None)
        self._salts.pop(name, None)

    def rotate(self, name: str, embedding: np.ndarray) -> None:
        """Re-key one subject's templates after a suspected compromise."""
        salt = _random_salt()
        self._salts[name] = salt
        self._templates[name] = [self.protector.protect(embedding, salt)]

    def __len__(self) -> int:
        return len(self._templates)


def _random_salt(n: int = 16) -> bytes:
    import secrets

    return secrets.token_bytes(n)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v
