"""
Audit Logger — extraction metadata logging.

Records every extraction result with full provenance to a persistent
JSONL log file for later analysis and compliance.

Spec: Section 8.3 — Audit Logger
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.models import DocumentType, ExtractionMetadata, ExtractionResult

logger = logging.getLogger(__name__)


class AuditLogger:
    """Every extraction gets a full metadata audit trail.

    The logger appends structured JSON Lines records to a persistent
    file so that extraction history can be replayed, analysed, and
    audited long after the request completes.
    """

    def __init__(
        self,
        log_path: str = "./logs/audit.jsonl",
        rotate_size_mb: int = 100,
        max_archives: int = 10,
    ) -> None:
        """Create the audit logger.

        Args:
            log_path: Path to the JSONL audit log file.  The parent
                directory is created automatically if it does not exist.
            rotate_size_mb: Size threshold (MB) for log rotation (stub).
            max_archives: Maximum number of archived log files (stub).
        """
        self._log_path: str = log_path
        self.rotate_size_mb = rotate_size_mb
        self.max_archives = max_archives
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, result: ExtractionResult) -> None:
        """Log an extraction result with full provenance.

        The record is appended as a single JSON Lines row containing
        the extraction metadata and summary statistics.

        Args:
            result: The complete extraction result to log.
        """
        try:
            record: dict[str, Any] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "extraction_id": str(result.metadata.extraction_id),
                "document_type": result.document_type.value,
                "vlm_model": result.metadata.vlm_model,
                "llm_model": result.metadata.llm_model,
                "overall_confidence": result.overall_confidence,
                "unresolved_count": result.unresolved_count,
                "validation_failure_count": result.validation_failure_count,
                "corrected_count": result.corrected_count,
                "processing_time_ms": result.metadata.processing_time_ms,
                "ram_peak_mb": result.metadata.ram_peak_mb,
                "phases_completed": [p.value for p in result.metadata.phases_completed],
                "few_shot_injections": result.metadata.few_shot_injections,
            }
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Audit logged: extraction_id=%s", result.metadata.extraction_id)
        except Exception as exc:
            logger.error("Failed to write audit log: %s", exc)

    def get_history(
        self,
        document_type: Optional[DocumentType] = None,
        limit: int = 100,
    ) -> List[ExtractionMetadata]:
        """Alias for get_extraction_history with limit support."""
        history = self.get_extraction_history(document_type)
        return history[:limit]

    def get_extraction_history(
        self, document_type: Optional[DocumentType] = None
    ) -> List[ExtractionMetadata]:
        """Retrieve past extractions for analysis.

        Args:
            document_type: If provided, filter to this document type only.

        Returns:
            List of extraction metadata objects, most recent first.
        """
        results: List[ExtractionMetadata] = []
        if not os.path.isfile(self._log_path):
            return results

        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if document_type is not None:
                            if record.get("document_type") != document_type.value:
                                continue
                        # We return lightweight metadata objects
                        meta = ExtractionMetadata(
                            extraction_id=record.get("extraction_id", ""),
                            timestamp=datetime.fromisoformat(record["timestamp"]),
                            vlm_model=record.get("vlm_model", ""),
                            llm_model=record.get("llm_model", ""),
                            document_type=DocumentType(record["document_type"])
                            if record.get("document_type")
                            else None,
                            processing_time_ms=record.get("processing_time_ms"),
                            ram_peak_mb=record.get("ram_peak_mb"),
                            few_shot_injections=record.get("few_shot_injections", 0),
                        )
                        results.append(meta)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except OSError as exc:
            logger.error("Failed to read audit log: %s", exc)

        # Most recent first
        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Compute aggregate statistics from the audit log.

        Returns:
            Dictionary with keys: ``total_extractions``,
            ``avg_confidence``, ``unresolved_rate``,
            ``avg_processing_time_ms``, ``document_type_breakdown``,
            ``model_versions``.
        """
        total: int = 0
        confidence_sum: float = 0.0
        unresolved_sum: int = 0
        field_count: int = 0
        processing_time_sum: float = 0.0
        processing_time_count: int = 0
        doc_type_counts: Dict[str, int] = {}
        vlm_models: set[str] = set()
        llm_models: set[str] = set()

        if not os.path.isfile(self._log_path):
            return self._empty_stats()

        try:
            with open(self._log_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        total += 1
                        confidence_sum += record.get("overall_confidence", 0.0)
                        unresolved_sum += record.get("unresolved_count", 0)
                        field_count += 1  # simplified: count records as proxy

                        pt = record.get("processing_time_ms")
                        if pt is not None:
                            processing_time_sum += pt
                            processing_time_count += 1

                        dt = record.get("document_type", "Unbekannt")
                        doc_type_counts[dt] = doc_type_counts.get(dt, 0) + 1

                        if record.get("vlm_model"):
                            vlm_models.add(record["vlm_model"])
                        if record.get("llm_model"):
                            llm_models.add(record["llm_model"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        except OSError as exc:
            logger.error("Failed to read audit log for stats: %s", exc)
            return self._empty_stats()

        if total == 0:
            return self._empty_stats()

        return {
            "total_extractions": total,
            "avg_confidence": round(confidence_sum / total, 4),
            "unresolved_rate": round(unresolved_sum / max(field_count, 1), 4),
            "avg_processing_time_ms": round(
                processing_time_sum / max(processing_time_count, 1), 2
            ),
            "document_type_breakdown": doc_type_counts,
            "model_versions": {
                "vlm": sorted(vlm_models),
                "llm": sorted(llm_models),
            },
        }

    @property
    def log_path(self) -> "Path":
        """Return the log file path as a Path object."""
        from pathlib import Path
        return Path(self._log_path)

    def get_extraction(self, extraction_id: "UUID") -> Optional[Any]:
        """Retrieve a single extraction result from the audit log by ID.

        Args:
            extraction_id: The UUID of the extraction to find.

        Returns:
            The ExtractionResult if found, else None.
        """
        history = self.get_extraction_history()
        for meta in history:
            if str(meta.extraction_id) == str(extraction_id):
                # Reconstruct a minimal ExtractionResult from metadata
                return ExtractionResult(
                    metadata=meta,
                    document_type=meta.document_type or DocumentType.UNBEKANNT,
                )
        return None

    def rotate_logs(self) -> None:
        """Archive the current log file and start fresh.

        Creates a gzip-compressed archive in the ``archive/`` subdirectory
        next to the log file, then truncates the active log.
        """
        import gzip
        import shutil
        from datetime import datetime
        from pathlib import Path

        if not os.path.isfile(self._log_path):
            return

        log_path = Path(self._log_path)
        archive_dir = log_path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"audit_{timestamp}.jsonl.gz"

        with open(self._log_path, "rb") as f_in:
            with gzip.open(archive_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Truncate the active log
        with open(self._log_path, "w", encoding="utf-8"):
            pass

        logger.info("Rotated audit log to %s", archive_path)

    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        """Return a stats dict with zero/empty values."""
        return {
            "total_extractions": 0,
            "avg_confidence": 0.0,
            "unresolved_rate": 0.0,
            "avg_processing_time_ms": 0.0,
            "document_type_breakdown": {},
            "model_versions": {"vlm": [], "llm": []},
        }
