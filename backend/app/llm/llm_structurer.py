"""
Optional local-LLM structuring (llama.cpp, Apple Metal accelerated).

After layout-aware OCR mapping, some fields may still be missing or
low-confidence. This module asks a *local* GGUF model to fill ONLY those
fields, strictly from the OCR text already extracted — never inventing data.

Design guarantees:
  * **Off by default** — the orchestrator only constructs this when
    ``ENABLE_LLM_STRUCTURING`` is set and the model file exists.
  * **Anti-fabrication** — the system prompt forbids guessing; absent fields
    must come back ``null``; temperature defaults to 0.0.
  * **Sub-threshold confidence** — LLM-proposed values are returned at low
    confidence so they stay flagged for human review and still pass through
    symbolic validation.
  * **Metal** — ``n_gpu_layers=-1`` offloads all layers to the Apple GPU.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.models.enums import DocumentType

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a strict data extraction assistant for German bureaucratic "
    "documents. You are given OCR text fragments already read from one "
    "document and a list of fields that still need values. Return a single "
    "JSON object mapping each requested field to its value found in the OCR "
    "text. Rules: (1) Use ONLY text present in the provided OCR context. "
    "(2) Never guess, complete, translate, or invent a value. (3) If a field "
    "is not clearly present, set it to null. (4) Output JSON only, no prose."
)

# Human-readable hints per field so the model knows what to look for.
_FIELD_HINTS: dict[str, str] = {
    "familienname": "family name / surname",
    "vorname": "first name(s)",
    "geburtsdatum": "date of birth (DD.MM.YYYY)",
    "geburtsort": "place of birth",
    "staatsangehoerigkeit": "nationality",
    "strasse": "street name",
    "hausnummer": "house number",
    "postleitzahl": "5-digit postal code",
    "wohnort": "city of residence",
    "einzugsdatum": "move-in date (DD.MM.YYYY)",
    "vorherige_anschrift": "previous address",
    "steueridentifikationsnummer": "11-digit tax ID",
    "veranlagungszeitraum": "tax year (YYYY)",
    "zu_versteuerndes_einkommen": "taxable income (number)",
    "festgesetzte_steuer": "assessed tax (number)",
    "arbeitgeber": "employer name",
    "brutto_lohn": "gross salary (number)",
    "netto_lohn": "net salary (number)",
    "abrechnungszeitraum": "pay period (MM.YYYY)",
    "steuerklasse": "tax class (I-VI)",
    "dokumentnummer": "document number",
    "gueltig_bis": "expiry date (DD.MM.YYYY)",
    "ausstellungsdatum": "issue date (DD.MM.YYYY)",
    "ausstellende_behoerde": "issuing authority",
}


class LLMStructurer:
    """Thin, guarded llama.cpp wrapper that fills missing schema fields."""

    LLM_CONFIDENCE = 0.55  # below the 0.70 review threshold, by design

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 8192,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self._llm: Any = None

    def _load(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama

        logger.info(
            "Loading GGUF model for structuring (n_gpu_layers=%s, Metal)",
            self.n_gpu_layers,
        )
        # n_gpu_layers=-1 -> offload every transformer layer to the Apple Metal GPU.
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def close(self) -> None:
        """Release the model (frees Metal buffers)."""
        self._llm = None

    def fill_fields(
        self,
        document_type: DocumentType,
        canonical: dict[str, Any],
        missing: list[str],
    ) -> dict[str, tuple[Any, float]]:
        """Return ``{field_name: (value, confidence)}`` for fields it could fill.

        Args:
            document_type: classified document type (for prompt context).
            canonical: mapping of field name → MappedField (already-read values
                provide the OCR context the model may draw from).
            missing: field names the mapper could not confidently resolve.
        """
        if not missing:
            return {}
        self._load()

        known_context = {
            name: mapped_field.raw_text or mapped_field.value
            for name, mapped_field in canonical.items()
            if getattr(mapped_field, "value", None) not in (None, "")
        }
        prompt = self._build_prompt(document_type, known_context, missing)

        completion = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        text = completion["choices"][0]["message"]["content"]
        data = self._parse_json(text)

        result: dict[str, tuple[Any, float]] = {}
        for name in missing:
            value = data.get(name)
            if value not in (None, "", "null", "None"):
                result[name] = (value, self.LLM_CONFIDENCE)
        return result

    @staticmethod
    def _build_prompt(
        document_type: DocumentType,
        known_context: dict[str, Any],
        missing: list[str],
    ) -> str:
        lines = [f"Document type: {document_type.value}", "", "OCR context already read:"]
        if known_context:
            for name, value in known_context.items():
                lines.append(f"  - {name}: {value}")
        else:
            lines.append("  (no fields read yet)")
        lines.append("")
        lines.append("Fill these fields (use null if not present in the OCR context):")
        for name in missing:
            hint = _FIELD_HINTS.get(name, name)
            lines.append(f"  - {name} ({hint})")
        lines.append("")
        lines.append("Return a single JSON object with exactly these keys.")
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        if not text or not text.strip():
            return {}
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        try:
            data = json.loads(cleaned)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass
        # Best-effort: grab the first {...} block.
        try:
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            data = json.loads(cleaned[start:end])
            return data if isinstance(data, dict) else {}
        except (ValueError, json.JSONDecodeError):
            return {}
