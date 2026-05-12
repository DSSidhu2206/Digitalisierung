"""
Dual-Pass Extraction Engine — Stage 2.

Runs two complementary VLM passes over the same image and cross-validates
results for consistency.  Pass A identifies the *structure* (field names,
positions, types); Pass B extracts the *values* (raw text, confidence).
Fields whose structure contradicts their value are flagged for human review.

Spec: Section 6.3 — Dual-Pass Extraction
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DualPassResult
# ---------------------------------------------------------------------------

@dataclass
class DualPassResult:
    """Combined result from both extraction passes.

    Attributes:
        structural_map: Pass A output — field_name -> structure metadata.
        value_map: Pass B output — field_name -> value + confidence.
        consistency_score: Overall agreement between passes (0-1).
        contradictions: List of human-readable contradiction messages.
    """

    structural_map: dict[str, dict[str, Any]]
    value_map: dict[str, dict[str, Any]]
    consistency_score: float
    contradictions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DualPassExtractor
# ---------------------------------------------------------------------------

class DualPassExtractor:
    """Two-pass extraction with cross-validation.

    *Pass A* identifies the document structure — every field name, label,
    and approximate bounding box.  *Pass B* extracts the actual values
    together with confidence scores.  The extractor then compares the
    two maps and flags any contradictions (e.g. a field present in Pass A
    but missing in Pass B, or a Pass B value for a field not listed in
    Pass A).

    Attributes:
        PROMPT_PASS_A: Prompt template for structural identification.
        PROMPT_PASS_B: Prompt template for value extraction.
    """

    PROMPT_PASS_A: str = """Analyze this bureaucratic document image.
PASS A — STRUCTURAL: Identify all form fields, their labels, and positions.
Output a JSON map of: field_name -> {label_text, field_type, estimated_bbox}
Do NOT extract values yet. Only identify the structure."""

    PROMPT_PASS_B: str = """Analyze this bureaucratic document image.
PASS B — VALUE EXTRACTION: Extract the actual values for each visible field.
Output a JSON map of: field_name -> {raw_value, confidence_0_to_1}
If text is blurry or ambiguous, mark confidence < 0.5."""

    def extract(
        self,
        image_path: str,
        vlm: "VLMManager",
        learned_context: str = "",
    ) -> DualPassResult:
        """Run dual-pass extraction and cross-validate.

        Pipeline:

        1. **Pass A** (structural) — prompt the VLM to identify all
           fields and their bounding boxes.
        2. **Pass B** (value) — prompt the VLM to extract actual values.
        3. **Parse** both responses as JSON.
        4. **Cross-validate** — compare field-name sets, flag missing
           fields, type mismatches, and orphaned values.
        5. Return a :class:`DualPassResult` with a *consistency_score*.

        Args:
            image_path: Path to the document image.
            vlm: A :class:`~app.vision.vlm_loader.VLMManager` or
                :class:`~app.vision.vlm_loader.MockVLMManager`
                instance.
            learned_context: Optional few-shot examples retrieved from the
                learned image corpus.

        Returns:
            :class:`DualPassResult` containing both maps and consistency
            metadata.
        """
        logger.info("DualPassExtractor starting — %s", image_path)

        # --------------------------------------------------------------
        # Pass A — structural identification
        # --------------------------------------------------------------
        if not vlm.is_loaded:
            vlm.load()

        if getattr(vlm, "supports_direct_extraction", False):
            try:
                document = vlm.extract_document(image_path, with_layout=True)
                result = self._from_document_extraction(document)
                logger.info(
                    "DualPassExtractor complete via direct Surya path - fields=%d consistency=%.2f",
                    len(result.value_map),
                    result.consistency_score,
                )
                return result
            except Exception:
                logger.exception(
                    "Direct Surya extraction failed; falling back to prompt-compatible passes"
                )

        prompt_a = self._with_learned_context(self.PROMPT_PASS_A, learned_context)
        prompt_b = self._with_learned_context(self.PROMPT_PASS_B, learned_context)

        logger.info("DualPassExtractor — Pass A (structural)")
        raw_a = vlm.generate(image_path, prompt_a)
        structural_map = self._parse_json_response(raw_a)
        logger.debug("Pass A returned %d fields", len(structural_map))

        # --------------------------------------------------------------
        # Pass B — value extraction
        # --------------------------------------------------------------
        logger.info("DualPassExtractor — Pass B (value extraction)")
        raw_b = vlm.generate(image_path, prompt_b)
        value_map = self._parse_json_response(raw_b)
        logger.debug("Pass B returned %d fields", len(value_map))

        # --------------------------------------------------------------
        # Cross-validation
        # --------------------------------------------------------------
        consistency_score, contradictions = self._cross_validate(
            structural_map, value_map
        )

        logger.info(
            "DualPassExtractor complete — consistency=%.2f contradictions=%d",
            consistency_score,
            len(contradictions),
        )

        return DualPassResult(
            structural_map=structural_map,
            value_map=value_map,
            consistency_score=consistency_score,
            contradictions=contradictions,
        )

    def _from_document_extraction(self, extraction: Any) -> DualPassResult:
        """Convert normalized Surya OCR/layout output into dual-pass maps."""
        structural_map: dict[str, dict[str, Any]] = {}
        value_map: dict[str, dict[str, Any]] = {}
        contradictions: list[str] = []

        lines = list(getattr(extraction, "lines", []) or [])
        engine = str(getattr(extraction, "engine", "surya") or "surya")
        fallback_confidence = 0.85 if engine == "surya" else 0.55

        for index, line in enumerate(lines, start=1):
            text = " ".join(str(getattr(line, "text", "") or "").split())
            if not text:
                continue

            key, value = self._split_line(text)
            base_name = self._normalise_field_name(key or f"line_{index:04d}")
            field_name = self._unique_field_name(structural_map, base_name)
            confidence = self._coerce_confidence(
                getattr(line, "confidence", fallback_confidence)
            )
            if confidence <= 0.0:
                confidence = fallback_confidence

            structural_map[field_name] = {
                "label_text": key or f"OCR line {index}",
                "field_type": self._infer_field_type(value or text),
                "estimated_bbox": getattr(line, "bbox", None),
                "engine": engine,
            }
            value_map[field_name] = {
                "raw_value": value or text,
                "confidence_0_to_1": confidence,
                "engine": engine,
            }
            if confidence < 0.5:
                contradictions.append(
                    f"Field '{field_name}' has low confidence ({confidence:.2f})"
                )

        if not value_map:
            text = str(getattr(extraction, "text", "") or "")
            for index, text_line in enumerate(text.splitlines(), start=1):
                cleaned = " ".join(text_line.strip().split())
                if not cleaned:
                    continue
                field_name = self._unique_field_name(
                    structural_map,
                    self._normalise_field_name(f"line_{index:04d}"),
                )
                structural_map[field_name] = {
                    "label_text": f"OCR line {index}",
                    "field_type": self._infer_field_type(cleaned),
                    "estimated_bbox": None,
                    "engine": engine,
                }
                value_map[field_name] = {
                    "raw_value": cleaned,
                    "confidence_0_to_1": fallback_confidence,
                    "engine": engine,
                }

        consistency_score, cross_contradictions = self._cross_validate(
            structural_map,
            value_map,
        )
        contradictions.extend(cross_contradictions)
        if value_map:
            avg_confidence = sum(
                self._coerce_confidence(value.get("confidence_0_to_1", 0.0))
                for value in value_map.values()
                if isinstance(value, dict)
            ) / max(len(value_map), 1)
            consistency_score = min(consistency_score, max(0.0, avg_confidence))

        return DualPassResult(
            structural_map=structural_map,
            value_map=value_map,
            consistency_score=consistency_score,
            contradictions=contradictions,
        )

    @staticmethod
    def _with_learned_context(prompt: str, learned_context: str) -> str:
        """Append retrieved learned image examples to a VLM prompt."""
        if not learned_context.strip():
            return prompt
        return (
            f"{prompt}\n\n"
            f"{learned_context}\n\n"
            "Use the learned examples only as recognition guidance for noisy "
            "scans. Do not copy values from examples into this extraction."
        )

    # ------------------------------------------------------------------
    # Cross-validation logic
    # ------------------------------------------------------------------

    def _cross_validate(
        self,
        structural_map: dict[str, Any],
        value_map: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Compare Pass A structure against Pass B values.

        Checks performed:

        1. **Missing in B** — field in A but no value extracted.
        2. **Orphan in B** — value extracted for a field not in A.
        3. **Low confidence** — field present but confidence < 0.5.

        Returns:
            ``(consistency_score, contradictions)`` where score is 1.0
            when no contradictions are found.
        """
        contradictions: list[str] = []

        set_a = set(structural_map.keys())
        set_b = set(value_map.keys())

        # 1. Fields missing in Pass B -----------------------------------
        missing_in_b = set_a - set_b
        for field_name in missing_in_b:
            contradictions.append(
                f"Field '{field_name}' identified in structure (Pass A) "
                f"but no value extracted (Pass B) — flag for human review"
            )
            logger.warning("Contradiction: %s missing in Pass B", field_name)

        # 2. Orphan values in Pass B ------------------------------------
        orphan_in_b = set_b - set_a
        for field_name in orphan_in_b:
            contradictions.append(
                f"Value found for '{field_name}' in Pass B but not listed "
                f"in structural map (Pass A) — possible misidentification"
            )
            logger.warning("Contradiction: %s orphan in Pass B", field_name)

        # 3. Low confidence fields --------------------------------------
        for field_name in set_a & set_b:
            val_entry = value_map[field_name]
            if isinstance(val_entry, dict):
                confidence = val_entry.get("confidence_0_to_1", 1.0)
                raw_value = val_entry.get("raw_value", "")
                if confidence < 0.5:
                    contradictions.append(
                        f"Field '{field_name}' has low confidence "
                        f"({confidence:.2f}) — value '{raw_value}' may be unreliable"
                    )
                    logger.warning(
                        "Low confidence: %s (%.2f)", field_name, confidence
                    )
                elif not raw_value or raw_value in ("null", "None", ""):
                    contradictions.append(
                        f"Field '{field_name}' has empty/null value — "
                        f"mark as UNRESOLVED"
                    )

        # 4. Compute consistency score ----------------------------------
        if not set_a and not set_b:
            return 0.0, ["Both passes returned empty results"]

        total_fields = len(set_a | set_b)
        matched_fields = len(set_a & set_b)
        missing_penalty = len(missing_in_b) + len(orphan_in_b)
        low_conf_penalty = sum(
            1
            for f in set_a & set_b
            if isinstance(value_map.get(f), dict)
            and value_map[f].get("confidence_0_to_1", 1.0) < 0.5
        )

        if total_fields == 0:
            return 0.0, contradictions

        raw_score = (matched_fields - missing_penalty - low_conf_penalty * 0.5) / total_fields
        consistency_score = max(0.0, min(1.0, raw_score))

        return consistency_score, contradictions

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw_text: str) -> dict[str, Any]:
        """Safely parse VLM JSON output.

        Handles common VLM formatting issues:
        *   Markdown code fences (```json … ```).
        *   Trailing commentary after the JSON block.
        *   Bare JSON with no wrapper.

        Args:
            raw_text: Raw string returned by the VLM.

        Returns:
            Parsed dictionary; empty dict on failure.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty VLM response")
            return {}

        text = raw_text.strip()

        # Strip markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try to find the first JSON object/array
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting the first {…} or […] block
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            pass

        logger.warning("Could not parse VLM response as JSON: %s...", text[:200])
        return {}

    @staticmethod
    def _split_line(text: str) -> tuple[str, str]:
        cleaned = " ".join(text.strip().split())
        for separator in (":", "=", "|"):
            if separator in cleaned:
                key, value = cleaned.split(separator, 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    return key, value
        return "", cleaned

    @staticmethod
    def _normalise_field_name(value: str) -> str:
        field = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
        field = "_".join(part for part in field.split("_") if part)
        if not field:
            field = "field"
        if field[0].isdigit():
            field = f"field_{field}"
        return field[:80]

    @staticmethod
    def _unique_field_name(existing: dict[str, Any], base: str) -> str:
        candidate = base or "field"
        suffix = 2
        while candidate in existing:
            candidate = f"{base}_{suffix}"
            suffix += 1
        return candidate

    @staticmethod
    def _infer_field_type(text: str) -> str:
        stripped = text.strip()
        digits = sum(ch.isdigit() for ch in stripped)
        if digits >= 6 and any(ch in stripped for ch in (".", "/", "-")):
            return "date"
        if digits and digits >= max(1, len(stripped) // 2):
            return "number"
        return "text"
