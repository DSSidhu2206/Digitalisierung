"""
Surya document extraction adapter.

The rest of the application expects a stable vision interface, while Surya
exposes OCR, layout, and detection predictors.  This module keeps those Surya
details contained and returns a small, pipeline-friendly result object.
"""
from __future__ import annotations

import gc
import logging
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from PIL import Image

from app.runtime_device import configure_mps_runtime, surya_batch_sizes

logger = logging.getLogger(__name__)
_SURYA_MPS_ATTENTION_PATCHED = False


@dataclass(frozen=True)
class SuryaTextLine:
    """One OCR line returned by Surya or the fallback OCR engine."""

    text: str
    confidence: float
    bbox: Optional[dict[str, float]] = None
    polygon: list[list[float]] = field(default_factory=list)


@dataclass(frozen=True)
class SuryaLayoutBlock:
    """One document layout block returned by Surya."""

    label: str
    confidence: float
    bbox: Optional[dict[str, float]] = None
    position: Optional[int] = None


@dataclass(frozen=True)
class SuryaExtraction:
    """Normalized document extraction output."""

    text: str
    lines: list[SuryaTextLine]
    layout: list[SuryaLayoutBlock] = field(default_factory=list)
    image_bbox: Optional[tuple[float, float, float, float]] = None
    engine: str = "surya"
    model: str = "surya-ocr"


class SuryaDocumentExtractor:
    """Lazy Surya OCR/layout wrapper with a conservative fallback path."""

    MODEL_ID = "surya-ocr"

    def __init__(
        self,
        *,
        enable_layout: bool = True,
        allow_fallback: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.enable_layout = enable_layout
        self.allow_fallback = allow_fallback
        self.device = device or self._default_device()
        self.foundation: Any = None
        self.recognition: Any = None
        self.detection: Any = None
        self.layout_predictor: Any = None
        self._loaded = False
        self._engine = "surya"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def engine(self) -> str:
        return self._engine

    def load(self) -> None:
        """Load Surya predictors once and keep them resident for reuse."""
        if self._loaded:
            return

        self._configure_runtime_defaults(self.device)
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.layout import LayoutPredictor
            from surya.recognition import RecognitionPredictor

            self.device = _resolve_torch_device(self.device)
            if self.device == "mps":
                _patch_surya_mps_attention_unpack()
            logger.info("Loading Surya OCR predictors on device=%s", self.device)
            self.foundation = FoundationPredictor(device=self.device)
            self.recognition = RecognitionPredictor(self.foundation)
            self.detection = DetectionPredictor(device=self.device)
            if self.enable_layout:
                self.layout_predictor = LayoutPredictor(self.foundation)
            self._engine = "surya"
            self._loaded = True
            logger.info("Surya OCR predictors loaded.")
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(f"Unable to load Surya OCR: {exc}") from exc
            logger.warning(
                "Surya OCR unavailable; falling back to pytesseract for this run: %s",
                exc,
            )
            self._engine = "pytesseract"
            self._loaded = True

    def unload(self) -> None:
        """Release predictor references and clear accelerator caches."""
        self.foundation = None
        self.recognition = None
        self.detection = None
        self.layout_predictor = None
        self._loaded = False
        gc.collect()
        try:
            import torch

            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def extract_text(self, image_path: str | Path) -> str:
        """Return only OCR text, capped by the caller if needed."""
        return self.extract(image_path, with_layout=False).text

    def extract(
        self,
        image_path: str | Path,
        *,
        with_layout: Optional[bool] = None,
    ) -> SuryaExtraction:
        """Run OCR/layout extraction for one image."""
        results = self.extract_batch([image_path], with_layout=with_layout)
        return results[0] if results else SuryaExtraction(text="", lines=[])

    def extract_batch(
        self,
        image_paths: Iterable[str | Path],
        *,
        with_layout: Optional[bool] = None,
    ) -> list[SuryaExtraction]:
        """Run OCR/layout extraction for a batch of images."""
        paths = [Path(path) for path in image_paths]
        if not paths:
            return []
        if not self._loaded:
            self.load()

        if self._engine != "surya":
            return [self._extract_with_tesseract(path) for path in paths]

        images: list[Image.Image] = []
        sizes: list[tuple[int, int]] = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                images.append(image)
                sizes.append(image.size)

            mps_batches = surya_batch_sizes(self.device)
            recognition_batch_size = self._safe_batch_size(
                "RECOGNITION_BATCH_SIZE",
                default=mps_batches["recognition"] if self.device == "mps" else None,
                maximum=mps_batches["recognition"] if self.device == "mps" else None,
            )
            detection_batch_size = self._safe_batch_size(
                "DETECTOR_BATCH_SIZE",
                default=mps_batches["detector"] if self.device == "mps" else None,
                maximum=mps_batches["detector"] if self.device == "mps" else None,
            )
            task_name = os.getenv("SURYA_TASK_NAME", "ocr_with_boxes").strip()
            task_names = [task_name] * len(images) if task_name else None
            manual_bboxes = None
            if task_name in {"ocr_without_boxes", "block_without_boxes"}:
                manual_bboxes = [
                    [[0, 0, width, height]]
                    for width, height in sizes
                ]
            detection_predictor = (
                None
                if manual_bboxes is not None
                else self.detection
            )
            predictions = self.recognition(
                images,
                task_names=task_names,
                det_predictor=detection_predictor,
                detection_batch_size=detection_batch_size,
                recognition_batch_size=recognition_batch_size,
                bboxes=manual_bboxes,
                sort_lines=True,
            )

            layout_results: list[Any] = []
            should_layout = self.enable_layout if with_layout is None else with_layout
            if should_layout and self.layout_predictor is not None:
                layout_results = self.layout_predictor(
                    images,
                    batch_size=self._int_env("LAYOUT_BATCH_SIZE"),
                )

            return [
                self._normalise_surya_result(
                    prediction,
                    layout_results[index] if index < len(layout_results) else None,
                    width=sizes[index][0],
                    height=sizes[index][1],
                )
                for index, prediction in enumerate(predictions)
            ]
        except Exception as exc:
            if not self.allow_fallback:
                raise
            logger.warning("Surya extraction failed; falling back to pytesseract: %s", exc)
            return [self._extract_with_tesseract(path) for path in paths]
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass

    def _normalise_surya_result(
        self,
        prediction: Any,
        layout_result: Any,
        *,
        width: int,
        height: int,
    ) -> SuryaExtraction:
        lines: list[SuryaTextLine] = []
        for raw_line in self._get(prediction, "text_lines", []) or []:
            text = str(self._get(raw_line, "text", "") or "").strip()
            if not text:
                continue
            polygon = self._normalise_polygon(self._get(raw_line, "polygon", []) or [])
            lines.append(
                SuryaTextLine(
                    text=text,
                    confidence=self._coerce_confidence(
                        self._get(raw_line, "confidence", 0.85),
                        default=0.85,
                    ),
                    bbox=self._bbox_from_polygon(polygon, width, height),
                    polygon=polygon,
                )
            )

        blocks: list[SuryaLayoutBlock] = []
        if layout_result is not None:
            for raw_block in self._get(layout_result, "bboxes", []) or []:
                polygon = self._normalise_polygon(self._get(raw_block, "polygon", []) or [])
                blocks.append(
                    SuryaLayoutBlock(
                        label=str(self._get(raw_block, "label", "block") or "block"),
                        confidence=self._coerce_confidence(
                            self._get(raw_block, "confidence", 0.75),
                            default=0.75,
                        ),
                        bbox=self._bbox_from_polygon(polygon, width, height),
                        position=self._coerce_int(self._get(raw_block, "position", None)),
                    )
                )

        text = "\n".join(line.text for line in lines)
        image_bbox = self._coerce_image_bbox(self._get(prediction, "image_bbox", None))
        return SuryaExtraction(
            text=text,
            lines=lines,
            layout=blocks,
            image_bbox=image_bbox,
            engine="surya",
            model=self.MODEL_ID,
        )

    def _extract_with_tesseract(self, path: Path) -> SuryaExtraction:
        try:
            import pytesseract

            with Image.open(path) as image:
                try:
                    text = pytesseract.image_to_string(image, lang="deu+eng")
                except Exception:
                    text = pytesseract.image_to_string(image, lang="eng")
        except Exception as exc:
            logger.warning("Fallback pytesseract OCR failed for %s: %s", path, exc)
            text = ""

        lines = [
            SuryaTextLine(text=line.strip(), confidence=0.55)
            for line in text.splitlines()
            if line.strip()
        ]
        return SuryaExtraction(
            text=text.strip(),
            lines=lines,
            engine="pytesseract",
            model=self.MODEL_ID,
        )

    @staticmethod
    def _configure_runtime_defaults(device_requested: Optional[str] = None) -> None:
        """Keep Surya's default GPU batches under a 16GB local machine budget."""
        if "MODEL_CACHE_DIR" not in os.environ:
            project_root = Path(__file__).resolve().parents[3]
            cache_dir = project_root / "models" / "surya"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MODEL_CACHE_DIR"] = str(cache_dir)
        device = _resolve_torch_device(
            device_requested or os.getenv("SURYA_DEVICE") or os.getenv("TORCH_DEVICE")
        )
        os.environ["SURYA_DEVICE"] = device
        os.environ["TORCH_DEVICE"] = device
        if device == "cpu":
            worker_threads = str(max(4, min(os.cpu_count() or 8, 12)))
            os.environ.setdefault("OMP_NUM_THREADS", worker_threads)
            os.environ.setdefault("MKL_NUM_THREADS", worker_threads)
            os.environ.setdefault("VECLIB_MAXIMUM_THREADS", worker_threads)
            os.environ.setdefault("NUMEXPR_NUM_THREADS", worker_threads)
        if device == "mps":
            configure_mps_runtime("mps")
            batches = surya_batch_sizes("mps")
            os.environ["RECOGNITION_BATCH_SIZE"] = str(
                _bounded_positive_int(
                    os.getenv("RECOGNITION_BATCH_SIZE"),
                    default=batches["recognition"],
                    maximum=batches["recognition"],
                )
            )
            os.environ["DETECTOR_BATCH_SIZE"] = str(
                _bounded_positive_int(
                    os.getenv("DETECTOR_BATCH_SIZE"),
                    default=batches["detector"],
                    maximum=batches["detector"],
                )
            )
            os.environ.setdefault("LAYOUT_BATCH_SIZE", str(batches["layout"]))
        elif "RECOGNITION_BATCH_SIZE" not in os.environ:
            os.environ["RECOGNITION_BATCH_SIZE"] = os.getenv(
                "SURYA_RECOGNITION_BATCH_SIZE",
                "96",
            )
        if device != "mps" and "DETECTOR_BATCH_SIZE" not in os.environ:
            os.environ["DETECTOR_BATCH_SIZE"] = os.getenv("SURYA_DETECTOR_BATCH_SIZE", "12")
        if "LAYOUT_BATCH_SIZE" not in os.environ:
            os.environ["LAYOUT_BATCH_SIZE"] = os.getenv("SURYA_LAYOUT_BATCH_SIZE", "12")

    @staticmethod
    def _default_device() -> str:
        return _resolve_torch_device(os.getenv("SURYA_DEVICE") or os.getenv("TORCH_DEVICE"))

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _normalise_polygon(raw_polygon: Any) -> list[list[float]]:
        polygon: list[list[float]] = []
        if not isinstance(raw_polygon, list):
            return polygon
        for point in raw_polygon:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    polygon.append([float(point[0]), float(point[1])])
                except (TypeError, ValueError):
                    continue
        return polygon

    @staticmethod
    def _bbox_from_polygon(
        polygon: list[list[float]],
        width: int,
        height: int,
    ) -> Optional[dict[str, float]]:
        if not polygon or width <= 0 or height <= 0:
            return None
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return {
            "x1": max(0.0, min(min(xs) / width, 1.0)),
            "y1": max(0.0, min(min(ys) / height, 1.0)),
            "x2": max(0.0, min(max(xs) / width, 1.0)),
            "y2": max(0.0, min(max(ys) / height, 1.0)),
        }

    @staticmethod
    def _coerce_confidence(value: Any, *, default: float = 0.0) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_image_bbox(value: Any) -> Optional[tuple[float, float, float, float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_env(name: str) -> Optional[int]:
        value = os.getenv(name)
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    @classmethod
    def _safe_batch_size(
        cls,
        name: str,
        *,
        default: Optional[int] = None,
        maximum: Optional[int] = None,
    ) -> Optional[int]:
        value = cls._int_env(name)
        if value is None:
            value = default
        if value is not None and maximum is not None:
            value = min(value, maximum)
        return value


def _macos_supports_mps() -> bool:
    version = platform.mac_ver()[0]
    if not version:
        return False
    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        return False
    return major >= 14


def _resolve_torch_device(requested: Optional[str] = None) -> str:
    normalized = (requested or "").strip().lower()
    if normalized == "cpu":
        return "cpu"
    try:
        import torch

        if normalized == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "mps":
            return "mps" if _mps_is_usable(torch) else "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if _mps_is_usable(torch):
            return "mps"
    except Exception:
        pass
    return "cpu"


def _mps_is_usable(torch: Any) -> bool:
    if platform.system() != "Darwin" or not _macos_supports_mps():
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


def _patch_surya_mps_attention_unpack() -> None:
    """Avoid Surya's advanced-indexing attention unpack on Apple MPS.

    Surya 0.17.1's SDPA vision attention uses MPS advanced indexing to unpack
    ragged image tokens into padded batches.  On real RVL-CDIP batches this can
    raise torch.AcceleratorError with bogus out-of-bounds indices.  Slice copies
    are slower than ideal, but keep the work batched on the Apple GPU and avoid
    the catastrophic single-image retry path.
    """
    global _SURYA_MPS_ATTENTION_PATCHED
    if _SURYA_MPS_ATTENTION_PATCHED:
        return
    try:
        import torch
        from surya.common.surya.encoder import Qwen2_5_VLVisionSdpaAttention
    except Exception as exc:
        logger.warning("Unable to install Surya MPS attention patch: %s", exc)
        return

    original = Qwen2_5_VLVisionSdpaAttention.unpack_qkv_with_mask

    def mps_safe_unpack(self: Any, q: Any, k: Any, v: Any, cu_seqlens: Any) -> Any:
        if getattr(q.device, "type", "") != "mps":
            return original(self, q, k, v, cu_seqlens)

        device = q.device
        dtype = q.dtype
        seq_lengths = [
            max(0, int(value))
            for value in (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
        ]
        batch_size = len(seq_lengths)
        available_tokens = int(q.shape[0])
        adjusted_lengths: list[int] = []
        remaining = available_tokens
        for length in seq_lengths:
            adjusted = min(length, max(remaining, 0))
            adjusted_lengths.append(adjusted)
            remaining -= adjusted
        seq_lengths = adjusted_lengths
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        num_heads = int(q.shape[1])
        head_dim = int(q.shape[2])

        batched_shape = (batch_size, max_seq_len, num_heads, head_dim)
        batched_q = torch.zeros(batched_shape, device=device, dtype=dtype)
        batched_k = torch.zeros_like(batched_q)
        batched_v = torch.zeros_like(batched_q)
        attention_mask = torch.full(
            (batch_size, 1, max_seq_len, max_seq_len),
            fill_value=float("-inf"),
            device=device,
            dtype=dtype,
        )

        batch_index_parts = []
        position_index_parts = []
        offset = 0
        for batch_index, seq_len in enumerate(seq_lengths):
            if seq_len <= 0:
                continue
            end = offset + seq_len
            batched_q[batch_index, :seq_len, :, :].copy_(q[offset:end])
            batched_k[batch_index, :seq_len, :, :].copy_(k[offset:end])
            batched_v[batch_index, :seq_len, :, :].copy_(v[offset:end])
            attention_mask[batch_index, 0, :seq_len, :seq_len] = 0
            batch_index_parts.append(
                torch.full((seq_len,), batch_index, device=device, dtype=torch.long)
            )
            position_index_parts.append(
                torch.arange(seq_len, device=device, dtype=torch.long)
            )
            offset = end

        if batch_index_parts:
            batch_indices = torch.cat(batch_index_parts)
            position_indices = torch.cat(position_index_parts)
        else:
            batch_indices = torch.empty((0,), device=device, dtype=torch.long)
            position_indices = torch.empty((0,), device=device, dtype=torch.long)

        return (
            batched_q,
            batched_k,
            batched_v,
            attention_mask,
            batch_indices,
            position_indices,
        )

    Qwen2_5_VLVisionSdpaAttention.unpack_qkv_with_mask = mps_safe_unpack
    _SURYA_MPS_ATTENTION_PATCHED = True
    logger.info("Installed Surya MPS-safe attention unpack patch.")


def _bounded_positive_int(
    value: Optional[str],
    *,
    default: int,
    maximum: int,
) -> int:
    try:
        parsed = int(str(value)) if value is not None else default
    except ValueError:
        parsed = default
    return max(1, min(parsed, maximum))
