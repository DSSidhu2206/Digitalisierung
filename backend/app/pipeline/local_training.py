"""
Local image-learning job control.

The UI uses this manager to start, stop, and monitor resumable local learning
against the RVL-CDIP corpus.  It reports operational stats, not fabricated
model-quality claims: coverage, OCR availability, visual fingerprints, vector
index status, and throughput.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.database.image_learning_store import ImageLearningStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Runtime controls for a local learning chunk."""

    batch_size: int = 64
    max_seconds: int = 3300
    plateau_window: int = 50000
    heartbeat_seconds: float = 1.0


class LocalTrainingManager:
    """Manage the local RVL image-learning subprocess."""

    def __init__(self, project_root: Path, image_learning: ImageLearningStore) -> None:
        self.project_root = project_root
        self.image_learning = image_learning
        self.progress_path = project_root / "documents" / "image_learning_progress.json"
        self.log_path = project_root / "logs" / "image_learning_ingest.log"
        self.script_path = project_root / "scripts" / "run_learning_until_plateau.sh"
        self.manifest_path = project_root / "documents" / "rvl_cdip_manifest.json"
        self.image_dir = project_root / "documents" / "rvl_cdip"
        self.surya_model_cache = project_root / "models" / "surya"
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._started_at: Optional[float] = None

    def start(self, config: TrainingConfig) -> dict[str, Any]:
        """Start a resumable learning chunk if one is not already running."""
        if self.is_running():
            return self.status(state_override="running")
        if not self.script_path.exists():
            raise FileNotFoundError(f"Training script not found: {self.script_path}")

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.surya_model_cache.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        device = self._resolve_surya_device(env.get("SURYA_DEVICE"))
        is_mps = device == "mps"
        worker_threads = str(max(4, min(os.cpu_count() or 8, 12)))
        env.update(
            {
                "BATCH_SIZE": str(config.batch_size),
                "MAX_SECONDS": str(config.max_seconds),
                "PLATEAU_WINDOW": str(config.plateau_window),
                "HEARTBEAT_SECONDS": str(config.heartbeat_seconds),
                "MODEL_CACHE_DIR": str(self.surya_model_cache),
                "OCR_BATCH_SIZE": env.get("OCR_BATCH_SIZE", "32" if is_mps else "16"),
                "SURYA_TASK_NAME": env.get("SURYA_TASK_NAME", "ocr_without_boxes"),
                "SURYA_DEVICE": device,
                "TORCH_DEVICE": device,
                "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", worker_threads),
                "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", worker_threads),
                "VECLIB_MAXIMUM_THREADS": env.get("VECLIB_MAXIMUM_THREADS", worker_threads),
                "NUMEXPR_NUM_THREADS": env.get("NUMEXPR_NUM_THREADS", worker_threads),
            }
        )
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        if is_mps:
            env["SURYA_RECOGNITION_BATCH_SIZE"] = str(
                min(self._coerce_int(env.get("SURYA_RECOGNITION_BATCH_SIZE"), 32), 32)
            )
            env["RECOGNITION_BATCH_SIZE"] = env["SURYA_RECOGNITION_BATCH_SIZE"]
            env["SURYA_DETECTOR_BATCH_SIZE"] = str(
                min(self._coerce_int(env.get("SURYA_DETECTOR_BATCH_SIZE"), 4), 4)
            )
        else:
            env.setdefault("SURYA_DETECTOR_BATCH_SIZE", "8")
            env.setdefault("SURYA_RECOGNITION_BATCH_SIZE", "128")
        env.setdefault("SURYA_LAYOUT_BATCH_SIZE", "12")
        with open(self.log_path, "ab") as log_file:
            self._process = subprocess.Popen(
                [str(self.script_path)],
                cwd=str(self.project_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self._started_at = time.time()
        logger.info("Started local image learning job pid=%s", self._process.pid)
        return self.status(state_override="running")

    def stop(self) -> dict[str, Any]:
        """Terminate the currently managed learning process."""
        if self.is_running() and self._process is not None:
            pid = self._process.pid
            try:
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._process.wait(timeout=10)
            logger.info("Stopped local image learning job")
            self._mark_progress_state("stopped", "stopped_by_user")
        self._process = None
        return self.status(state_override="stopped")

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def status(self, state_override: Optional[str] = None) -> dict[str, Any]:
        """Return current learning status and enterprise dashboard metrics."""
        progress = self._read_progress()
        learning_stats = self.image_learning.stats()
        total_images = int(progress.get("total_images", 0) or self._count_total_images())
        learned_total = int(learning_stats.get("total_images", 0) or 0)
        processed = int(progress.get("processed", 0) or 0)
        stored = max(int(progress.get("stored", 0) or 0), learned_total)
        skipped = int(progress.get("skipped", 0) or 0)
        processed_coverage = processed / total_images if total_images else 0.0
        indexed_coverage = stored / total_images if total_images else 0.0
        avg_quality = float(learning_stats.get("avg_quality_score", 0.0) or 0.0)
        ocr_coverage = float(learning_stats.get("ocr_coverage_rate", 0.0) or 0.0)
        avg_ocr_chars = float(learning_stats.get("avg_ocr_chars", 0.0) or 0.0)
        vector_index = bool(learning_stats.get("vector_index_enabled", False))
        learning_score = self._learning_score(
            coverage=indexed_coverage,
            avg_quality=avg_quality,
            ocr_coverage=ocr_coverage,
            avg_ocr_chars=avg_ocr_chars,
            vector_index=vector_index,
        )

        updated_at_epoch = float(progress.get("updated_at_epoch", 0.0) or 0.0)
        heartbeat_age = max(time.time() - updated_at_epoch, 0.0) if updated_at_epoch else None
        running = self.is_running()
        fresh_external_progress = (
            not running
            and progress.get("state") == "running"
            and heartbeat_age is not None
            and heartbeat_age < 10
        )
        active_progress = running or fresh_external_progress
        state = state_override or (
            "running" if running else "external" if fresh_external_progress else progress.get("state") or "idle"
        )
        pid = self._process.pid if self._process is not None else None
        surya_device = self._resolve_surya_device(os.getenv("SURYA_DEVICE"))
        mps_defaults = surya_device == "mps"
        return {
            "state": state,
            "running": running,
            "pid": pid,
            "started_at": self._started_at,
            "processed": processed,
            "run_processed": int(progress.get("run_processed", 0) or 0),
            "stored": stored,
            "skipped": skipped,
            "pending_batch": int(progress.get("pending_batch", 0) or 0) if active_progress else 0,
            "total_images": total_images,
            "progress_pct": round(indexed_coverage * 100, 3),
            "processed_progress_pct": round(processed_coverage * 100, 3),
            "indexed_progress_pct": round(indexed_coverage * 100, 3),
            "images_per_second": float(progress.get("images_per_second", 0.0) or 0.0),
            "updated_at": progress.get("updated_at"),
            "updated_at_epoch": updated_at_epoch or None,
            "heartbeat_age_seconds": round(heartbeat_age, 3) if heartbeat_age is not None else None,
            "current_file": (progress.get("current_file") or None) if active_progress else None,
            "current_index": progress.get("current_index") if active_progress else None,
            "last_event": progress.get("last_event") or None,
            "learning_score": learning_score,
            "avg_quality_score": avg_quality,
            "avg_ocr_chars": avg_ocr_chars,
            "ocr_coverage_rate": ocr_coverage,
            "visual_hash_coverage_rate": float(
                learning_stats.get("visual_hash_coverage_rate", 0.0) or 0.0
            ),
            "vector_index_enabled": vector_index,
            "ocr_engine": "surya-ocr",
            "ocr_batch_size": self._int_env("OCR_BATCH_SIZE", 32 if mps_defaults else 16),
            "surya_device": surya_device,
            "surya_recognition_batch_size": self._int_env(
                "SURYA_RECOGNITION_BATCH_SIZE",
                32 if mps_defaults else 128,
            ),
            "surya_detector_batch_size": self._int_env(
                "SURYA_DETECTOR_BATCH_SIZE",
                4 if mps_defaults else 8,
            ),
            "surya_task_name": os.getenv("SURYA_TASK_NAME") or "ocr_without_boxes",
            "top_labels": learning_stats.get("top_labels", {}),
            "recent_log": self._tail_log(),
        }

    def _read_progress(self) -> dict[str, Any]:
        if not self.progress_path.exists():
            return {}
        try:
            with open(self.progress_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

    def _mark_progress_state(self, state: str, last_event: str) -> None:
        progress = self._read_progress()
        if not progress:
            return
        progress.update(
            {
                "state": state,
                "current_file": None,
                "current_index": None,
                "pending_batch": 0,
                "last_event": last_event,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "updated_at_epoch": time.time(),
            }
        )
        tmp = self.progress_path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(progress, handle, indent=2)
            tmp.replace(self.progress_path)
        except OSError:
            logger.warning("Failed to mark local training progress as %s", state)

    def _tail_log(self, lines: int = 60) -> list[str]:
        if not self.log_path.exists():
            return []
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
        return text.splitlines()[-lines:]

    def _count_total_images(self) -> int:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, list):
                    return len(data)
            except (OSError, json.JSONDecodeError):
                pass
        if not self.image_dir.exists():
            return 0
        return sum(
            1
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        )

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        return LocalTrainingManager._coerce_int(os.getenv(name), default)

    @staticmethod
    def _coerce_int(value: Optional[str], default: int) -> int:
        try:
            return int(str(value)) if value is not None else default
        except ValueError:
            return default

    @classmethod
    def _resolve_surya_device(cls, requested: Optional[str] = None) -> str:
        normalized = (requested or "").strip().lower()
        if normalized == "cpu":
            return "cpu"
        try:
            import torch

            if normalized == "cuda":
                return "cuda" if torch.cuda.is_available() else "cpu"
            if normalized == "mps":
                return "mps" if cls._mps_is_usable(torch) else "cpu"
            if torch.cuda.is_available():
                return "cuda"
            if cls._mps_is_usable(torch):
                return "mps"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _mps_is_usable(torch: Any) -> bool:
        if platform.system() != "Darwin":
            return False
        try:
            if not getattr(torch.backends, "mps", None):
                return False
            if not torch.backends.mps.is_available():
                return False
            torch.ones(1, device="mps").cpu()
            return True
        except Exception as exc:
            logger.info("Ignoring unavailable Surya MPS device: %s", exc)
            return False

    @staticmethod
    def _learning_score(
        coverage: float,
        avg_quality: float,
        ocr_coverage: float,
        avg_ocr_chars: float,
        vector_index: bool,
    ) -> float:
        text_depth = min(avg_ocr_chars / 600.0, 1.0)
        vector_component = 1.0 if vector_index else 0.55
        score = (
            0.36 * min(max(coverage, 0.0), 1.0)
            + 0.22 * min(max(avg_quality, 0.0), 1.0)
            + 0.18 * min(max(ocr_coverage, 0.0), 1.0)
            + 0.14 * text_depth
            + 0.10 * vector_component
        )
        return round(score * 100, 2)
