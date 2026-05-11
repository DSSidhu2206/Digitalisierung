"""
Correction Capture -- Higher-level helper for the learning loop.

Validates and stores user corrections in ChromaDB, then formats
retrieved few-shot examples as prompt-ready text sections.  Includes
token estimation to enforce the ``MAX_FEW_SHOT_CORRECTIONS`` budget
(spec Section 12).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token budget (mirrors spec Section 12 and config.py)
# ---------------------------------------------------------------------------
MAX_FEW_SHOT_CORRECTIONS: int = 5  # hard cap per query

# Rough heuristic: 1 token ~ 0.75 English characters.  We use a
# conservative multiplier so we *under*-estimate rather than overflow.
_CHAR_PER_TOKEN_ESTIMATE: float = 0.65

# Header / boilerplate overhead (in tokens) for the few-shot section.
_HEADER_OVERHEAD_TOKENS: int = 8

# Per-correction line overhead (label text, punctuation, etc).
_LINE_OVERHEAD_TOKENS: int = 12


class CorrectionCapture:
    """Higher-level wrapper around :class:`ChromaManager` for the ABE learning loop.

    Responsibilities:
    1. **Validation** -- sanity-check correction data before persistence.
    2. **Capture** -- store validated corrections atomically in ChromaDB.
    3. **Few-shot formatting** -- retrieve relevant corrections and format
       them as a string ready for prompt injection.
    4. **Token safety** -- ensure the formatted section never exceeds the
       ``MAX_FEW_SHOT_CORRECTIONS`` budget.
    """

    def __init__(self, chroma_manager: Optional[Any] = None) -> None:
        """Initialise the capture helper.

        Args:
            chroma_manager: An existing :class:`ChromaManager` instance.
                If ``None``, a new one is created with default settings.
        """
        if chroma_manager is None:
            from app.database.chroma_manager import ChromaManager
            chroma_manager = ChromaManager()
        self._chroma = chroma_manager

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(
        self,
        extraction_id: str,
        field_name: str,
        original_value: str,
        corrected_value: str,
        reason: str = "",
        document_type: str = "Unbekannt",
    ) -> Optional[str]:
        """Validate and store a user correction.

        The correction is stored atomically: embedding is computed *before*
        any DB mutation, so either both steps succeed or neither has any
        effect.

        Args:
            extraction_id: UUID of the parent extraction (for audit trail).
            field_name: Schema field that was corrected (e.g. ``"plz"``).
            original_value: The value produced by the extraction pipeline.
            corrected_value: The user-supplied corrected value.
            reason: Optional explanation for the correction.

        Returns:
            The correction UUID if storage succeeded, ``None`` if validation
            failed.

        Raises:
            RuntimeError: If ChromaDB persistence fails after embedding
                succeeded (should not happen in normal operation).
        """
        # --- Validation ---
        if not field_name or not field_name.strip():
            logger.warning("capture rejected: empty field_name")
            return None
        if corrected_value is None:
            logger.warning("capture rejected: corrected_value is None")
            return None

        field = field_name.strip()
        original = str(original_value) if original_value is not None else ""
        corrected = str(corrected_value)
        doc_type = document_type

        try:
            correction_id: str = self._chroma.add_correction(
                original_text=original,
                corrected_value=corrected,
                field_name=field,
                document_type=doc_type,
                reason=reason,
            )
        except RuntimeError:
            logger.exception("capture failed for field=%s", field)
            raise

        logger.info(
            "Correction captured: id=%s extraction=%s field=%s",
            correction_id,
            extraction_id,
            field,
        )
        return correction_id

    # ------------------------------------------------------------------
    # Few-shot prompt formatting
    # ------------------------------------------------------------------

    def get_few_shot_prompt_section(
        self,
        field_name: str,
        current_text: str,
        document_type: str = "Unbekannt",
        max_corrections: Optional[int] = None,
    ) -> str:
        """Retrieve relevant corrections and format them as a prompt section.

        Token estimation is applied *before* formatting so that we never
        produce a section exceeding the prompt budget.  Corrections are
        ordered by descending similarity.

        Args:
            field_name: The field currently being processed.
            current_text: The raw text extracted for that field.
            document_type: Document type discriminator.
            max_corrections: Hard cap on number of corrections.  If ``None``,
                :data:`MAX_FEW_SHOT_CORRECTIONS` is used.  Silently clamped
                to the global maximum.

        Returns:
            Formatted prompt section string, or empty string if no relevant
            corrections were found.
        """
        cap = self._clamp_max(max_corrections)

        # Retrieve from vector store (already capped at MAX_CORRECTIONS_PER_QUERY)
        corrections: list[dict[str, Any]] = self._chroma.retrieve_relevant(
            field_name=field_name,
            current_text=current_text,
            document_type=document_type,
            n_results=cap,
        )

        if not corrections:
            return ""

        # Token-aware culling: drop corrections until the estimated total
        # fits within the budget.
        corrections = self._apply_token_budget(corrections)

        if not corrections:
            return ""

        return self._format_corrections(corrections)

    # ------------------------------------------------------------------
    # Token estimation helpers
    # ------------------------------------------------------------------

    def _apply_token_budget(
        self, corrections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Drop corrections (from the end) until the token budget is met.

        We iterate from the highest-similarity (front of list) and keep
        adding while the running total stays under budget.
        """
        kept: list[dict[str, Any]] = []
        running_tokens: int = _HEADER_OVERHEAD_TOKENS

        for corr in corrections:
            line_tokens = self._estimate_correction_tokens(corr)
            if running_tokens + line_tokens > self._max_tokens():
                break
            running_tokens += line_tokens
            kept.append(corr)

        return kept

    @staticmethod
    def _estimate_correction_tokens(correction: dict[str, Any]) -> int:
        """Rough token count for a single formatted correction line.

        Uses a conservative character-per-token heuristic.
        Typical German text: ~4 characters per token.
        """
        original = str(correction.get("original", ""))
        corrected = str(correction.get("corrected", ""))
        reason = str(correction.get("reason", ""))

        content_chars = len(original) + len(corrected) + len(reason)
        # Conservative: ~3.5 chars per token for German mixed text
        content_tokens = int(content_chars / 3.5)
        return _LINE_OVERHEAD_TOKENS + content_tokens

    @staticmethod
    def _max_tokens() -> int:
        """Maximum tokens allocated to the few-shot section."""
        return MAX_FEW_SHOT_CORRECTIONS * 50  # ~250 tokens budget

    @staticmethod
    def _clamp_max(max_corrections: Optional[int]) -> int:
        """Clamp user-provided cap to the global maximum."""
        if max_corrections is None:
            return MAX_FEW_SHOT_CORRECTIONS
        return min(max(0, max_corrections), MAX_FEW_SHOT_CORRECTIONS)

    @staticmethod
    def _format_corrections(corrections: list[dict[str, Any]]) -> str:
        """Build the prompt injection string from a list of corrections.

        Mirrors the format shown in spec Section 7.3 (PromptBuilder).
        """
        lines: list[str] = [
            "",
            "LEARNED CORRECTIONS (apply these patterns):",
            "",
        ]
        for i, corr in enumerate(corrections, start=1):
            original = corr.get("original", "")
            corrected = corr.get("corrected", "")
            reason = corr.get("reason", "")
            line = f"  {i}. Similar document: '{original}' was corrected to '{corrected}'."
            if reason:
                line += f" Reason: {reason}"
            lines.append(line)
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pass-through helpers
    # ------------------------------------------------------------------

    def get_correction_by_id(self, correction_id: str) -> Optional[dict[str, Any]]:
        """Direct lookup of a correction by UUID."""
        return self._chroma.get_correction_by_id(correction_id)

    def delete_correction(self, correction_id: str) -> bool:
        """Remove a correction from the learning loop."""
        return self._chroma.delete_correction(correction_id)

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate correction statistics."""
        return self._chroma.get_stats()
