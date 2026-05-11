"""Instructor client for structured LLM output with Pydantic schema forcing.

The ``InstructorClient`` wraps an ``LLMManager`` and uses the ``instructor``
library to force every generation into a strict Pydantic model.  If schema
forcing fails, the field is marked ``UNRESOLVED`` rather than hallucinated.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from app.models.enums import FieldStatus
from app.models.extraction_models import FieldResult

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# Attempt to import instructor; if unavailable we fall back to manual JSON.
# ---------------------------------------------------------------------------
try:
    import instructor

    _INSTRUCTOR_AVAILABLE = True
except ImportError:
    _INSTRUCTOR_AVAILABLE = False
    logger.warning(
        "instructor library not available — InstructorClient will use "
        "manual JSON parsing as fallback. Install with: pip install instructor"
    )


class InstructorClient:
    """Client that forces LLM output into a Pydantic schema via instructor.

    Temperature is **always** 0.0.  If schema enforcement fails the
    extraction falls back to ``UNRESOLVED`` — never fabricated.

    Attributes:
        llm_manager: The ``LLMManager`` (or ``MockLLMManager``) that owns
            the loaded GGUF model.
    """

    def __init__(self, llm_manager: "LLMManager") -> None:  # type: ignore[name-defined]
        """Patch the underlying LLM with instructor.

        Args:
            llm_manager: An *already loaded* ``LLMManager`` instance.
        """
        self.llm_manager = llm_manager
        self.client: Optional[Any] = None

        if _INSTRUCTOR_AVAILABLE and llm_manager.is_loaded():
            try:
                self.client = instructor.patch(llm_manager.llm)
                logger.info("InstructorClient: instructor patch applied.")
            except Exception as exc:
                logger.warning("InstructorClient: failed to patch instructor: %s", exc)
                self.client = None
        elif not _INSTRUCTOR_AVAILABLE:
            logger.info("InstructorClient: running in manual-JSON fallback mode.")
        else:
            logger.warning(
                "InstructorClient: LLM not loaded — schema forcing unavailable."
            )

    # ------------------------------------------------------------------
    # Core extraction helpers
    # ------------------------------------------------------------------

    def _temperature(self) -> float:
        """Return the mandatory temperature of 0.0."""
        return 0.0

    def _create_unresolved_field(self, field_name: str, reason: str) -> FieldResult:
        """Create an UNRESOLVED FieldResult when extraction fails.

        Args:
            field_name: The name of the field that could not be extracted.
            reason: Human-readable explanation of the failure.

        Returns:
            A ``FieldResult`` with status ``FieldStatus.UNRESOLVED``.
        """
        return FieldResult(
            field_name=field_name,
            value=None,
            status=FieldStatus.UNRESOLVED,
            confidence=0.0,
            validation_message=reason,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_structured(
        self,
        text: str,
        response_model: Type[T],
        few_shot_examples: list[dict] = [],
    ) -> T:
        """Force LLM output into the given Pydantic schema.

        Injects *few_shot_examples* into the prompt and runs inference at
        temperature 0.0.  If instructor is available the response is
        schema-validated automatically; otherwise a manual JSON parse +
        Pydantic validation is performed.

        Args:
            text: The prompt text to send to the LLM.
            response_model: Pydantic model class that defines the expected
                output structure.
            few_shot_examples: Optional list of correction dicts with keys
                ``original``, ``corrected``, ``reason``.

        Returns:
            An instance of *response_model*.

        Raises:
            RuntimeError: If the LLM is not loaded.
            ValidationError: If the LLM output cannot be parsed into the schema
                (callers should catch this and fall back to UNRESOLVED).
        """
        if not self.llm_manager.is_loaded():
            raise RuntimeError("LLM not loaded. Call load() first.")

        temperature = self._temperature()

        # Append few-shot examples into the prompt if provided
        augmented_prompt = self._inject_few_shot(text, few_shot_examples)

        if self.client is not None and _INSTRUCTOR_AVAILABLE:
            return self._extract_with_instructor(
                augmented_prompt, response_model, temperature
            )

        return self._extract_with_manual_json(
            augmented_prompt, response_model, temperature
        )

    def generate_json(self, prompt: str, schema: Type[T]) -> T:
        """Alternative direct JSON generation without few-shot context.

        Convenience wrapper around :meth:`extract_structured` that skips
        few-shot injection.

        Args:
            prompt: The prompt text.
            schema: Pydantic model class for the expected output.

        Returns:
            An instance of *schema*.
        """
        return self.extract_structured(
            text=prompt,
            response_model=schema,
            few_shot_examples=[],
        )

    # ------------------------------------------------------------------
    # Internal implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_few_shot(
        prompt: str,
        examples: list[dict],
    ) -> str:
        """Append few-shot correction examples to the prompt.

        Each example dict must contain:
        - ``original``: the original (incorrect) value
        - ``corrected``: the corrected value
        - ``reason``: why the correction was made
        """
        if not examples:
            return prompt

        lines = [prompt, "", "=== LEARNED CORRECTIONS (apply these patterns) ==="]
        for idx, ex in enumerate(examples, 1):
            lines.append(
                f"{idx}. Field '{ex.get('field', 'unknown')}': "
                f"'{ex.get('original', '')}' → '{ex.get('corrected', '')}' "
                f"(Reason: {ex.get('reason', 'N/A')})"
            )
        lines.append("=== END CORRECTIONS ===")
        return "\n".join(lines)

    def _extract_with_instructor(
        self,
        prompt: str,
        response_model: Type[T],
        temperature: float,
    ) -> T:
        """Use instructor's structured-completion API.

        Falls back to manual JSON parsing on any exception.
        """
        try:
            response = self.client.chat.completions.create(
                model="local",  # dummy — instructor uses the patched Llama
                messages=[
                    {"role": "system", "content": "You are a deterministic extraction engine. Output ONLY valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_model=response_model,
                temperature=temperature,
            )
            logger.debug("Instructor extraction succeeded for %s", response_model.__name__)
            return response
        except (ValidationError, Exception) as exc:
            logger.warning(
                "Instructor schema forcing failed: %s — "
                "falling back to manual JSON parsing.",
                exc,
            )
            return self._extract_with_manual_json(
                prompt, response_model, temperature
            )

    def _extract_with_manual_json(
        self,
        prompt: str,
        response_model: Type[T],
        temperature: float,
    ) -> T:
        """Manual JSON extraction + Pydantic validation fallback.

        This path is used when instructor is not installed or when
        instructor's own validation fails.
        """
        # Add explicit JSON instructions to the prompt
        json_prompt = (
            f"{prompt}\n\n"
            "IMPORTANT: Respond with ONLY a valid JSON object. "
            "Do not include markdown code fences or any explanatory text. "
            "If any field is unclear or missing, set its value to null."
        )

        raw_text = self.llm_manager.generate(
            json_prompt,
            max_tokens=4096,
            temperature=temperature,
        )

        # Try to extract JSON from the response (handle code fences)
        json_str = self._extract_json_block(raw_text)

        try:
            data = json.loads(json_str)
            instance = response_model.model_validate(data)
            logger.debug("Manual JSON parsing succeeded for %s", response_model.__name__)
            return instance
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.error(
                "Manual JSON parsing also failed for %s: %s",
                response_model.__name__,
                exc,
            )
            raise ValidationError.from_exception_data(
                title=response_model.__name__,
                line_errors=[
                    {
                        "type": "value_error",
                        "loc": ("__root__",),
                        "msg": f"Could not parse LLM output into {response_model.__name__}: {exc}",
                        "input": raw_text[:500],
                    }
                ],
            )

    @staticmethod
    def _extract_json_block(text: str) -> str:
        """Extract a JSON object from text that may contain markdown fences.

        Handles:
        - `` ```json ... ``` ``  fences
        - `` ``` ... ``` ``      fences (no language tag)
        - Bare JSON starting with ``{`` or ``[``

        Returns:
            The cleaned JSON string.
        """
        text = text.strip()

        # Look for JSON inside code fences
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    return candidate
                # Handle ```json\n{...}
                if candidate.startswith("json"):
                    inner = candidate[4:].strip()
                    if inner.startswith("{") or inner.startswith("["):
                        return inner

        # Look for the first { or [ and try to parse from there
        for start_char in ("{", "["):
            if start_char in text:
                start_idx = text.index(start_char)
                # For objects, find the matching end brace
                if start_char == "{":
                    # Simple brace counting
                    depth = 0
                    for i, ch in enumerate(text[start_idx:], start_idx):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                return text[start_idx : i + 1]
                return text[start_idx:]

        return text
