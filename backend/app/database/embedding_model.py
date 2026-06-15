"""
Embedding Model for the Digitalisierung System.

Provides lazy-loaded sentence-transformer embeddings for correction vectors,
accelerated on Apple Metal (MPS) when available.  Falls back to a
*deterministic hash embedding* — not a zero vector — when
sentence-transformers is unavailable, so similarity search retains real
lexical signal in degraded environments.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: sentence-transformers
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[misc, assignment]
    _HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "sentence-transformers not installed. EmbeddingModel will use a "
        "deterministic hash embedding. Install with: pip install sentence-transformers"
    )


def _resolve_device() -> str:
    """Pick the best available torch device: Metal (mps) > CUDA > CPU."""
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class EmbeddingModel:
    """Lazy-loaded sentence-transformer wrapper for correction embeddings.

    The model is loaded on first :meth:`embed` to avoid heavy startup I/O and
    keep the dependency optional.  On Apple Silicon it loads on the ``mps``
    device so encoding runs on the Metal GPU.
    """

    DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    _EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name: str = model_name or self.DEFAULT_MODEL
        self._model: Optional[object] = None
        self._loaded: bool = False
        self._device: str = "cpu"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the SentenceTransformer model onto the best device."""
        if self._loaded:
            return
        if not _HAS_SENTENCE_TRANSFORMERS:
            logger.debug("Hash-embedding mode for '%s'", self._model_name)
            self._loaded = True
            return
        self._device = _resolve_device()
        logger.info(
            "Loading embedding model '%s' on device=%s ...",
            self._model_name,
            self._device,
        )
        try:
            self._model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:
            # e.g. no network for first download, or MPS load issue → degrade.
            logger.warning(
                "Embedding model load failed (%s); using hash embedding.", exc
            )
            self._model = None
        self._loaded = True

    def _hash_embedding(self, text: str) -> list[float]:
        """Deterministic 384-dim hashed bag-of-tokens embedding (L2-normalised).

        Crude but real: two texts sharing tokens get non-zero cosine
        similarity, unlike an all-zero stub. Keeps the correction loop usable
        without the ML stack.
        """
        dim = self._EMBEDDING_DIMENSION
        vector = [0.0] * dim
        tokens = [t for t in re.split(r"\W+", text.lower()) if t]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            vector[idx] += 1.0 if (digest[4] & 1) else -1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]
        return vector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text* (hash fallback if needed)."""
        self._load()
        if self._model is None:
            return self._hash_embedding(text)
        import numpy as np

        vector: "np.ndarray" = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in one forward pass."""
        self._load()
        if self._model is None:
            return [self._hash_embedding(t) for t in texts]
        import numpy as np

        vectors: "np.ndarray" = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()

    @property
    def is_loaded(self) -> bool:
        """Whether the model load has been attempted (True in fallback too)."""
        return self._loaded

    @property
    def is_stub(self) -> bool:
        """Whether embeddings come from the hash fallback (no real model)."""
        return self._loaded and self._model is None

    @property
    def device(self) -> str:
        """Torch device the model loaded on ('mps' on Apple Silicon)."""
        return self._device

    @property
    def dimension(self) -> int:
        """Dimensionality of produced embedding vectors."""
        return self._EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        """The Hugging Face model identifier in use."""
        return self._model_name
