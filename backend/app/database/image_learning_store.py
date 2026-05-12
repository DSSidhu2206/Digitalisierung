"""
Persistent image-learning memory for dirty document images.

This module does not fine-tune model weights.  It gives the extraction
pipeline a durable retrieval memory: every ingested image contributes OCR
text, quality signals, labels, and metadata that can be retrieved as
few-shot context for later images.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import hashlib
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.database.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _HAS_CHROMADB = True
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[misc, assignment]
    _HAS_CHROMADB = False


@dataclass(frozen=True)
class LearnedImageRecord:
    """A durable learning example derived from one document image."""

    image_id: str
    image_path: str
    filename: str
    source_split: str = ""
    label_name: str = ""
    label_index: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    file_size: Optional[int] = None
    sha256: str = ""
    quality_score: float = 0.0
    form_type: str = "Unbekannt"
    orientation: str = "correct"
    ocr_text: str = ""
    visual_hash: str = ""

    @property
    def summary_text(self) -> str:
        """Return the text embedded for similarity retrieval."""
        parts = [
            f"label={self.label_name or 'unknown'}",
            f"form_type={self.form_type}",
            f"quality={self.quality_score:.2f}",
            self.ocr_text[:2000],
        ]
        return "\n".join(part for part in parts if part)


class ImageLearningStore:
    """Store and retrieve learned image examples at 400K-image scale."""

    COLLECTION_NAME = "learned_images"
    DEFAULT_SQLITE_NAME = "image_learning.sqlite3"
    MAX_PROMPT_EXAMPLES = 5

    def __init__(
        self,
        persist_dir: str = "./chroma_data",
        sqlite_path: Optional[str] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path = Path(sqlite_path) if sqlite_path else self.persist_dir / self.DEFAULT_SQLITE_NAME
        self.vector_dir = self.persist_dir / "image_learning_vectors"
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self._embedder = EmbeddingModel()
        self._lock = threading.RLock()
        self._sqlite = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self._sqlite.row_factory = sqlite3.Row
        self._init_sqlite()

        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        if _HAS_CHROMADB:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.vector_dir),
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception as exc:
                logger.warning(
                    "Chroma image-learning index unavailable; using SQLite/FTS only: %s",
                    exc,
                )
                self._client = None
                self._collection = None
        else:
            logger.warning(
                "chromadb not installed; image learning will persist metadata "
                "to SQLite but similarity retrieval will use text fallback only."
            )

    def _init_sqlite(self) -> None:
        self._sqlite.execute(
            """
            CREATE TABLE IF NOT EXISTS learned_images (
                image_id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                source_split TEXT DEFAULT '',
                label_name TEXT DEFAULT '',
                label_index INTEGER,
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                sha256 TEXT DEFAULT '',
                quality_score REAL DEFAULT 0,
                form_type TEXT DEFAULT 'Unbekannt',
                orientation TEXT DEFAULT 'correct',
                ocr_text TEXT DEFAULT '',
                visual_hash TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        self._ensure_column("learned_images", "visual_hash", "TEXT DEFAULT ''")
        self._sqlite.execute(
            "CREATE INDEX IF NOT EXISTS idx_learned_images_label ON learned_images(label_name)"
        )
        self._sqlite.execute(
            "CREATE INDEX IF NOT EXISTS idx_learned_images_form_type ON learned_images(form_type)"
        )
        self._sqlite.execute(
            "CREATE INDEX IF NOT EXISTS idx_learned_images_visual_hash ON learned_images(visual_hash)"
        )
        self._sqlite.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS learned_images_fts
            USING fts5(image_id UNINDEXED, label_name, form_type, ocr_text)
            """
        )
        self._sqlite.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            row["name"]
            for row in self._sqlite.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            self._sqlite.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self) -> None:
        """Close the SQLite connection."""
        self._sqlite.close()

    def has_image(self, image_id: str) -> bool:
        with self._lock:
            row = self._sqlite.execute(
                "SELECT 1 FROM learned_images WHERE image_id = ? LIMIT 1",
                (image_id,),
            ).fetchone()
        return row is not None

    def add_records(self, records: list[LearnedImageRecord]) -> int:
        """Persist records and embeddings, skipping existing image ids."""
        new_records = [record for record in records if not self.has_image(record.image_id)]
        if not new_records:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._sqlite:
                self._sqlite.executemany(
                    """
                    INSERT OR IGNORE INTO learned_images (
                        image_id, image_path, filename, source_split, label_name,
                        label_index, width, height, file_size, sha256, quality_score,
                        form_type, orientation, ocr_text, visual_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            record.image_id,
                            record.image_path,
                            record.filename,
                            record.source_split,
                            record.label_name,
                            record.label_index,
                            record.width,
                            record.height,
                            record.file_size,
                            record.sha256,
                            record.quality_score,
                            record.form_type,
                            record.orientation,
                            record.ocr_text,
                            record.visual_hash,
                            now,
                        )
                        for record in new_records
                    ],
                )
                self._sqlite.executemany(
                    """
                    INSERT INTO learned_images_fts (
                        image_id, label_name, form_type, ocr_text
                    ) VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            record.image_id,
                            record.label_name,
                            record.form_type,
                            record.ocr_text,
                        )
                        for record in new_records
                    ],
                )

        if self._collection is not None:
            try:
                embeddings = self._embed_texts([record.summary_text for record in new_records])
                self._collection.upsert(
                    ids=[record.image_id for record in new_records],
                    embeddings=embeddings,
                    documents=[self._document_payload(record) for record in new_records],
                    metadatas=[self._metadata(record) for record in new_records],
                )
            except Exception as exc:
                logger.warning(
                    "Vector indexing failed; records remain available in SQLite/FTS: %s",
                    exc,
                )
                self._collection = None

        return len(new_records)

    def retrieve_similar(
        self,
        query_text: str,
        document_type: str = "Unbekannt",
        limit: int = MAX_PROMPT_EXAMPLES,
    ) -> list[dict[str, Any]]:
        """Return similar learned image examples for prompt/context use."""
        limit = max(0, min(limit, self.MAX_PROMPT_EXAMPLES))
        if limit == 0:
            return []

        if self._collection is not None and query_text.strip():
            try:
                query_embedding = self._embed_text(query_text)
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                )
                docs = results.get("documents", [[]])[0] if results else []
                examples = [json.loads(doc) for doc in docs if doc]
                if document_type != "Unbekannt":
                    typed = [
                        example for example in examples
                        if example.get("form_type") in (document_type, "Unbekannt")
                    ]
                    return typed[:limit] or examples[:limit]
                return examples[:limit]
            except Exception as exc:
                logger.warning("learned image retrieval failed: %s", exc)

        return self._sqlite_fts_fallback(query_text, document_type, limit)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._embedder.embed_batch(texts)
        except Exception as exc:
            logger.warning("semantic embedder unavailable; using local hash embeddings: %s", exc)
            return [self._hash_embedding(text) for text in texts]

    def _embed_text(self, text: str) -> list[float]:
        try:
            return self._embedder.embed(text)
        except Exception as exc:
            logger.warning("semantic embedder unavailable; using local hash embedding: %s", exc)
            return self._hash_embedding(text)

    @staticmethod
    def _hash_embedding(text: str, dimensions: int = 384) -> list[float]:
        vector = [0.0] * dimensions
        tokens = [token for token in text.lower().split() if token]
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def build_prompt_section(self, examples: list[dict[str, Any]]) -> str:
        """Format learned image examples for extraction prompts."""
        if not examples:
            return ""
        lines = ["=== LEARNED IMAGE EXAMPLES (use as OCR/quality patterns) ==="]
        for index, example in enumerate(examples[: self.MAX_PROMPT_EXAMPLES], start=1):
            ocr = " ".join(str(example.get("ocr_text", "")).split())[:400]
            lines.append(
                f"{index}. label={example.get('label_name') or 'unknown'}; "
                f"form_type={example.get('form_type') or 'Unbekannt'}; "
                f"quality={float(example.get('quality_score') or 0):.2f}; "
                f"ocr='{ocr}'"
            )
        lines.append("=== END LEARNED IMAGE EXAMPLES ===")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Return corpus size and high-level breakdowns."""
        with self._lock:
            total = self._sqlite.execute("SELECT COUNT(*) AS n FROM learned_images").fetchone()["n"]
            aggregate = self._sqlite.execute(
                """
                SELECT
                    AVG(quality_score) AS avg_quality_score,
                    AVG(LENGTH(ocr_text)) AS avg_ocr_chars,
                    SUM(CASE WHEN LENGTH(TRIM(ocr_text)) > 0 THEN 1 ELSE 0 END) AS ocr_rows,
                    SUM(CASE WHEN LENGTH(TRIM(visual_hash)) > 0 THEN 1 ELSE 0 END) AS visual_hash_rows
                FROM learned_images
                """
            ).fetchone()
            labels = self._sqlite.execute(
                """
                SELECT label_name, COUNT(*) AS n
                FROM learned_images
                GROUP BY label_name
                ORDER BY n DESC
                LIMIT 20
                """
            ).fetchall()
        return {
            "total_images": total,
            "vector_index_enabled": self._collection is not None,
            "sqlite_path": str(self.sqlite_path),
            "avg_quality_score": float(aggregate["avg_quality_score"] or 0.0),
            "avg_ocr_chars": float(aggregate["avg_ocr_chars"] or 0.0),
            "ocr_coverage_rate": (
                float(aggregate["ocr_rows"] or 0.0) / total if total else 0.0
            ),
            "visual_hash_coverage_rate": (
                float(aggregate["visual_hash_rows"] or 0.0) / total if total else 0.0
            ),
            "top_labels": {row["label_name"] or "unknown": row["n"] for row in labels},
        }

    def _sqlite_fts_fallback(
        self,
        query_text: str,
        document_type: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        terms = [self._clean_fts_term(term) for term in query_text.lower().split()]
        terms = [term for term in terms if len(term) >= 4][:8]
        with self._lock:
            if not terms:
                rows = self._sqlite.execute(
                    """
                    SELECT * FROM learned_images
                    WHERE (? = 'Unbekannt' OR form_type IN (?, 'Unbekannt'))
                    ORDER BY quality_score DESC
                    LIMIT ?
                    """,
                    (document_type, document_type, limit),
                ).fetchall()
            else:
                query = " OR ".join(terms)
                rows = self._sqlite.execute(
                    """
                    SELECT learned_images.*
                    FROM learned_images_fts
                    JOIN learned_images USING(image_id)
                    WHERE learned_images_fts MATCH ?
                    ORDER BY quality_score DESC
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
        return [self._row_to_example(row) for row in rows]

    @staticmethod
    def _clean_fts_term(term: str) -> str:
        return "".join(ch for ch in term if ch.isalnum() or ch in ("_", "-"))

    @staticmethod
    def _document_payload(record: LearnedImageRecord) -> str:
        return json.dumps(
            {
                "image_id": record.image_id,
                "filename": record.filename,
                "label_name": record.label_name,
                "form_type": record.form_type,
                "quality_score": record.quality_score,
                "visual_hash": record.visual_hash,
                "ocr_text": record.ocr_text[:2000],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _metadata(record: LearnedImageRecord) -> dict[str, Any]:
        return {
            "filename": record.filename,
            "label_name": record.label_name,
            "form_type": record.form_type,
            "quality_score": float(record.quality_score),
            "visual_hash": record.visual_hash,
        }

    @staticmethod
    def _row_to_example(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "image_id": row["image_id"],
            "filename": row["filename"],
            "label_name": row["label_name"],
            "form_type": row["form_type"],
            "quality_score": row["quality_score"],
            "visual_hash": row["visual_hash"],
            "ocr_text": row["ocr_text"],
        }
