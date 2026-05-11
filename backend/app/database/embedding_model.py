"""
Embedding Model for the Digitalisierung ABE System.

Provides lazy-loaded sentence-transformer embeddings for correction vectors.
Implements a fallback stub mode for development environments without
sentence-transformers installed.
"""

from __future__ import annotations

import logging
import os
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
        "sentence-transformers not installed. EmbeddingModel will operate in "
        "stub mode (no-op embeddings). Install with: pip install sentence-transformers"
    )


class EmbeddingModel:
    """Lazy-loaded sentence-transformer wrapper for correction embeddings.

    The model is loaded on the first call to :meth:`embed` rather than at
    import time to avoid heavy I/O at startup and to keep the dependency
    truly optional.
    """

    DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    _EMBEDDING_DIMENSION: int = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialise the embedding model (lazy-loading).

        Args:
            model_name: Hugging Face model identifier.  When ``None`` the
                :attr:`DEFAULT_MODEL` is used.
        """
        self._model_name: str = model_name or self.DEFAULT_MODEL
        self._model: Optional[object] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the SentenceTransformer model into memory."""
        if self._loaded:
            return
        if not _HAS_SENTENCE_TRANSFORMERS:
            logger.debug(
                "Stub mode: skipping model load for '%s'", self._model_name
            )
            self._loaded = True
            return
        logger.info("Loading embedding model '%s' ...", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._loaded = True
        logger.info("Embedding model '%s' loaded.", self._model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*.

        If sentence-transformers is not available a deterministic zero
        vector is returned (stub mode) so that downstream code can still
        run without the heavy dependency.

        Args:
            text: The input text to embed.

        Returns:
            A list of floats representing the embedding vector.  Length
            is :attr:`_EMBEDDING_DIMENSION` in normal mode.
        """
        self._load()
        if self._model is None:
            # Stub mode -- deterministic zero vector so downstream code
            # still works (albeit without semantic search quality).
            return [0.0] * self._EMBEDDING_DIMENSION
        import numpy as np

        vector: np.ndarray = self._model.encode(text)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts in one forward pass.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, one per input string.
        """
        self._load()
        if self._model is None:
            return [[0.0] * self._EMBEDDING_DIMENSION for _ in texts]
        import numpy as np

        vectors: np.ndarray = self._model.encode(texts)
        return vectors.tolist()

    @property
    def is_loaded(self) -> bool:
        """Whether the underlying transformer model has been loaded.

        Returns True in stub mode (no heavy model needed) as well as
        when the real model is in memory.
        """
        return self._loaded

    @property
    def dimension(self) -> int:
        """Dimensionality of produced embedding vectors."""
        return self._EMBEDDING_DIMENSION

    @property
    def model_name(self) -> str:
        """The Hugging Face model identifier in use."""
        return self._model_name
