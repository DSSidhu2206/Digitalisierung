"""LLM engine package for Digitalisierung ABE."""

from app.llm.llm_loader import LLMManager, MockLLMManager
from app.llm.instructor_client import InstructorClient
from app.llm.prompt_builder import (
    MAX_CONTEXT_TOKENS,
    MAX_FEW_SHOT_CORRECTIONS,
    PromptBuilder,
    TOKENS_PER_WORD_ESTIMATE,
)
from app.llm.schema_adapter import (
    SchemaAdapter,
    build_field_guide,
    get_field_description,
    schema_to_prompt,
)

__all__ = [
    "LLMManager",
    "MockLLMManager",
    "InstructorClient",
    "PromptBuilder",
    "SchemaAdapter",
    "MAX_CONTEXT_TOKENS",
    "MAX_FEW_SHOT_CORRECTIONS",
    "TOKENS_PER_WORD_ESTIMATE",
    "build_field_guide",
    "get_field_description",
    "schema_to_prompt",
]
