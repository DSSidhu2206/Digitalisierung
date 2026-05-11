"""
Chroma Manager for the Digitalisierung ABE System.

Provides persistent vector storage, similarity-based retrieval, and
atomic CRUD operations for the few-shot correction learning loop.
Implements a stub mode for development environments without chromadb
installed.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: chromadb
# ---------------------------------------------------------------------------
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _HAS_CHROMADB = True
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[misc, assignment]
    _HAS_CHROMADB = False
    logger.warning(
        "chromadb not installed. ChromaManager will operate in stub mode "
        "(no-op). Install with: pip install chromadb"
    )

# ---------------------------------------------------------------------------
# Local imports (safe because they are in the same package)
# ---------------------------------------------------------------------------
from app.database.embedding_model import EmbeddingModel


class ChromaManager:
    """Local ChromaDB client for few-shot correction storage and retrieval.

    The manager handles:
    * Persistent vector storage of corrections with their embeddings.
    * Metadata-filtered similarity search scoped by ``field`` and ``doc_type``.
    * Atomic write operations (embedding generation + DB persist).
    * Stub-mode fallback when *chromadb* is not installed.

    Attributes:
        COLLECTION_NAME: Name of the ChromaDB collection used for corrections.
        EMBEDDING_MODEL: Hugging Face model identifier for embeddings.
        MAX_CORRECTIONS_PER_QUERY: Hard cap on results to prevent prompt
            overflow (token-limit safety per spec Section 12).
    """

    COLLECTION_NAME: str = "corrections"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_CORRECTIONS_PER_QUERY: int = 5

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, persist_dir: str = "./chroma_data") -> None:
        """Create a persistent ChromaDB client and correction collection.

        Args:
            persist_dir: Directory path for ChromaDB persistence.
                Created automatically if it does not exist.
        """
        self._persist_dir: str = persist_dir
        self._embedder: EmbeddingModel = EmbeddingModel(self.EMBEDDING_MODEL)

        if not _HAS_CHROMADB:
            logger.info(
                "ChromaManager running in STUB mode (chromadb not installed)."
            )
            self._client: Optional[Any] = None
            self._collection: Optional[Any] = None
            self._stub_data: Dict[str, Dict[str, Any]] = {}
            return

        os.makedirs(persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaManager initialised -- collection '%s' at '%s'",
            self.COLLECTION_NAME,
            persist_dir,
        )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def add_correction(
        self,
        original_text: str,
        corrected_value: str,
        field_name: str,
        document_type: str,
        reason: str = "",
    ) -> str:
        """Atomically store a correction with its embedding vector.

        The operation is atomic: the embedding is computed *before* any
        database mutation so that a failure in either step leaves the
        store in a consistent state.

        Args:
            original_text: The raw / wrongly-extracted text.
            corrected_value: The human-corrected value.
            field_name: Schema field name (e.g. ``"postleitzahl"``).
            document_type: Document type discriminator (e.g.
                ``"Meldebescheinigung"``).
            reason: Optional human-readable explanation for the correction.

        Returns:
            The UUID string assigned to the new correction record.

        Raises:
            RuntimeError: If embedding generation fails.
        """
        correction_id: str = str(uuid4())

        # --- 1. Atomic pre-computation: embedding must succeed first ---
        try:
            embedding: list[float] = self._embedder.embed(
                f"{field_name}: {original_text}"
            )
        except Exception as exc:
            logger.error("Embedding generation failed: %s", exc)
            raise RuntimeError(
                f"Failed to generate embedding for correction: {exc}"
            ) from exc

        # --- 2. Serialise document payload ---
        document: str = json.dumps(
            {
                "original": original_text,
                "corrected": corrected_value,
                "field": field_name,
                "doc_type": document_type,
                "reason": reason,
            },
            ensure_ascii=False,
        )

        # --- 3. Persist to ChromaDB (or stub) ---
        if self._collection is None:
            # Stub mode
            self._stub_data[correction_id] = {
                "embedding": embedding,
                "document": document,
                "metadata": {
                    "field": field_name,
                    "doc_type": document_type,
                },
            }
            logger.debug("Stub add_correction: id=%s", correction_id)
            return correction_id

        try:
            self._collection.add(
                embeddings=[embedding],
                documents=[document],
                metadatas=[{"field": field_name, "doc_type": document_type}],
                ids=[correction_id],
            )
        except Exception as exc:
            logger.error("ChromaDB add failed: %s", exc)
            raise RuntimeError(
                f"Failed to persist correction to ChromaDB: {exc}"
            ) from exc

        logger.info("Correction stored: id=%s field=%s", correction_id, field_name)
        return correction_id

    def retrieve_relevant(
        self,
        field_name: str,
        current_text: str,
        document_type: str,
        n_results: int = MAX_CORRECTIONS_PER_QUERY,
    ) -> list[dict[str, Any]]:
        """Retrieve similar past corrections for few-shot injection.

        Results are ranked by cosine similarity and capped at
        :attr:`MAX_CORRECTIONS_PER_QUERY` to respect the token budget.

        Args:
            field_name: Schema field to scope the search.
            current_text: The text currently being processed.
            document_type: Document type discriminator.
            n_results: Maximum number of corrections to return.
                Silently clamped to :attr:`MAX_CORRECTIONS_PER_QUERY`.

        Returns:
            List of parsed correction dictionaries, ordered by descending
            similarity.  Each dict contains keys: ``original``,
            ``corrected``, ``field``, ``doc_type``, ``reason``.
        """
        # Token-limit safety: hard cap (spec Section 12)
        n_results = min(n_results, self.MAX_CORRECTIONS_PER_QUERY)
        if n_results <= 0:
            return []

        # Compute query embedding
        try:
            embedding: list[float] = self._embedder.embed(
                f"{field_name}: {current_text}"
            )
        except Exception as exc:
            logger.error("Embedding generation failed during retrieval: %s", exc)
            return []

        # Stub mode: brute-force cosine on in-memory dict
        if self._collection is None:
            return self._stub_query(
                field_name=field_name,
                document_type=document_type,
                query_embedding=embedding,
                n_results=n_results,
            )

        # Production mode: ChromaDB similarity search
        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where={"$and": [{"field": field_name}, {"doc_type": document_type}]},
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        # Parse JSON documents into dicts
        corrections: list[dict[str, Any]] = []
        if results and results.get("documents") and results["documents"][0]:
            for doc in results["documents"][0]:
                try:
                    corrections.append(json.loads(doc))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed correction document: %s", doc)
                    continue
        return corrections

    def get_correction_by_id(self, correction_id: str) -> Optional[dict[str, Any]]:
        """Direct lookup of a single correction by its UUID.

        Args:
            correction_id: The UUID string returned by :meth:`add_correction`.

        Returns:
            Parsed correction dict, or ``None`` if not found.
        """
        if not correction_id:
            return None

        if self._collection is None:
            stub = self._stub_data.get(correction_id)
            if stub is None:
                return None
            try:
                return json.loads(stub["document"])
            except json.JSONDecodeError:
                return None

        try:
            results = self._collection.get(ids=[correction_id])
        except Exception as exc:
            logger.error("ChromaDB get failed for id=%s: %s", correction_id, exc)
            return None

        if results and results.get("documents") and results["documents"]:
            try:
                return json.loads(results["documents"][0])
            except json.JSONDecodeError:
                return None
        return None

    def delete_correction(self, correction_id: str) -> bool:
        """Remove a correction from the vector store.

        Args:
            correction_id: The UUID string of the correction to remove.

        Returns:
            ``True`` if the correction existed and was deleted,
            ``False`` otherwise.
        """
        if not correction_id:
            return False

        if self._collection is None:
            if correction_id in self._stub_data:
                del self._stub_data[correction_id]
                logger.debug("Stub delete_correction: id=%s", correction_id)
                return True
            return False

        try:
            # Verify it exists before deleting
            existing = self._collection.get(ids=[correction_id])
            if not existing or not existing.get("ids"):
                return False
            self._collection.delete(ids=[correction_id])
            logger.info("Correction deleted: id=%s", correction_id)
            return True
        except Exception as exc:
            logger.error(
                "ChromaDB delete failed for id=%s: %s", correction_id, exc
            )
            return False

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about stored corrections.

        Returns:
            Dictionary with keys:
            - ``total``: Total number of corrections stored.
            - ``per_field``: ``{field_name: count}`` mapping.
            - ``per_doc_type``: ``{document_type: count}`` mapping.
        """
        if self._collection is None:
            return self._stub_stats()

        try:
            all_data = self._collection.get()
        except Exception as exc:
            logger.error("ChromaDB get_stats failed: %s", exc)
            return {"total": 0, "per_field": {}, "per_doc_type": {}}

        metadatas: list[dict] = all_data.get("metadatas") or []
        per_field: Dict[str, int] = {}
        per_doc_type: Dict[str, int] = {}

        for meta in metadatas:
            if meta is None:
                continue
            f = meta.get("field")
            d = meta.get("doc_type")
            if f:
                per_field[f] = per_field.get(f, 0) + 1
            if d:
                per_doc_type[d] = per_doc_type.get(d, 0) + 1

        return {
            "total": len(metadatas),
            "per_field": per_field,
            "per_doc_type": per_doc_type,
        }

    # ------------------------------------------------------------------
    # Stub-mode helpers
    # ------------------------------------------------------------------

    def _stub_query(
        self,
        field_name: str,
        document_type: str,
        query_embedding: list[float],
        n_results: int,
    ) -> list[dict[str, Any]]:
        """Brute-force cosine similarity on in-memory stub data."""
        import math

        candidates: list[tuple[float, dict[str, Any]]] = []

        for correction_id, data in self._stub_data.items():
            meta = data["metadata"]
            if meta.get("field") != field_name or meta.get("doc_type") != document_type:
                continue
            emb = data["embedding"]
            similarity = self._cosine_similarity(query_embedding, emb)
            try:
                parsed = json.loads(data["document"])
            except json.JSONDecodeError:
                continue
            candidates.append((similarity, parsed))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:n_results]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two equal-length float vectors."""
        import math

        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _stub_stats(self) -> dict[str, Any]:
        """Compute stats from stub-mode in-memory store."""
        per_field: Dict[str, int] = {}
        per_doc_type: Dict[str, int] = {}
        for data in self._stub_data.values():
            meta = data["metadata"]
            f = meta.get("field")
            d = meta.get("doc_type")
            if f:
                per_field[f] = per_field.get(f, 0) + 1
            if d:
                per_doc_type[d] = per_doc_type.get(d, 0) + 1
        return {
            "total": len(self._stub_data),
            "per_field": per_field,
            "per_doc_type": per_doc_type,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_stub(self) -> bool:
        """Whether the manager is running in stub mode (no chromadb)."""
        return self._collection is None

    @property
    def collection_name(self) -> str:
        """Name of the underlying ChromaDB collection."""
        return self.COLLECTION_NAME
