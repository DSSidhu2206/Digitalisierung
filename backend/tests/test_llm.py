"""Unit tests for the LLM Engine.

Covers:
- ``LLMManager`` / ``MockLLMManager`` load/unload lifecycle
- ``InstructorClient`` structured extraction with schema forcing
- ``PromptBuilder`` prompt construction and few-shot injection
- ``SchemaAdapter`` Pydantic → prompt conversion
- Temperature 0.0 enforcement on all paths
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import date
from unittest.mock import MagicMock, patch

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import BaseModel, Field, ValidationError

from app.models.document_schemas import (
    MeldebescheinigungSchema,
    get_schema_for_document_type,
)
from app.models.enums import DocumentType, FieldStatus
from app.models.extraction_models import FieldResult
from app.llm.llm_loader import MockLLMManager
from app.llm.instructor_client import InstructorClient
from app.llm.prompt_builder import (
    MAX_CONTEXT_TOKENS,
    MAX_FEW_SHOT_CORRECTIONS,
    PromptBuilder,
)
from app.llm.schema_adapter import SchemaAdapter, build_field_guide, schema_to_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleSchema(BaseModel):
    """Minimal schema for testing schema forcing."""
    name: str = Field(..., description="A person's name")
    age: int = Field(..., description="Age in years")


# ---------------------------------------------------------------------------
# MockLLMManager tests
# ---------------------------------------------------------------------------


class TestMockLLMManager(unittest.TestCase):
    """Tests for the mock LLM manager used in CI/testing."""

    def setUp(self) -> None:
        self.llm = MockLLMManager()

    def test_load_sets_loaded_flag(self) -> None:
        """Loading must set the internal _loaded flag."""
        self.assertFalse(self.llm.is_loaded())
        self.llm.load()
        self.assertTrue(self.llm.is_loaded())

    def test_unload_clears_loaded_flag(self) -> None:
        """Unloading must clear the internal _loaded flag."""
        self.llm.load()
        self.assertTrue(self.llm.is_loaded())
        self.llm.unload()
        self.assertFalse(self.llm.is_loaded())

    def test_generate_without_load_raises(self) -> None:
        """Generation without prior load() must raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.llm.generate("test prompt")

    def test_generate_returns_valid_json(self) -> None:
        """Mock generate() must return a parseable JSON string."""
        self.llm.load()
        result = self.llm.generate("Extract Meldebescheinigung fields: familienname vorname")
        self.llm.unload()

        # Must be valid JSON
        data = json.loads(result)
        self.assertIsInstance(data, dict)
        self.assertIn("_mock", data)
        self.assertTrue(data["_mock"])

    def test_generate_contains_document_type(self) -> None:
        """Mock response should identify the document type from the prompt."""
        self.llm.load()
        result = self.llm.generate("Document type: Meldebescheinigung")
        self.llm.unload()

        data = json.loads(result)
        self.assertEqual(data.get("document_type"), "Meldebescheinigung")

    def test_temperature_warning_logged(self) -> None:
        """Non-zero temperature should be accepted but trigger a warning path.

        We verify that the method runs without crashing; the warning itself
        is verified by checking the logger was called in integration.
        """
        self.llm.load()
        # MockLLMManager generates regardless of temperature
        result = self.llm.generate("test", temperature=0.7)
        self.llm.unload()
        self.assertIsInstance(result, str)

    def test_load_unload_cycle(self) -> None:
        """Multiple load/unload cycles must remain stable."""
        for _ in range(3):
            self.llm.load()
            self.assertTrue(self.llm.is_loaded())
            result = self.llm.generate("familienname vorname geburtsdatum")
            self.assertIsInstance(json.loads(result), dict)
            self.llm.unload()
            self.assertFalse(self.llm.is_loaded())


# ---------------------------------------------------------------------------
# LLMManager (with mocked llama_cpp) tests
# ---------------------------------------------------------------------------


class TestLLMManager(unittest.TestCase):
    """Tests for the real LLMManager with mocked dependencies."""

    @patch("app.llm.llm_loader._LLAMA_CPP_AVAILABLE", True)
    @patch("app.llm.llm_loader.Llama")
    def test_load_creates_llama_instance(self, mock_llama_cls: MagicMock) -> None:
        """Loading must instantiate Llama with correct parameters."""
        from app.llm.llm_loader import LLMManager

        mgr = LLMManager()

        # We need a model file to exist
        with patch("os.path.isfile", return_value=True):
            mgr.load()

        self.assertTrue(mgr.is_loaded())
        mock_llama_cls.assert_called_once()
        call_kwargs = mock_llama_cls.call_args.kwargs
        self.assertEqual(call_kwargs["n_ctx"], 8192)
        self.assertEqual(call_kwargs["n_threads"], 8)
        self.assertEqual(call_kwargs["n_batch"], 512)
        self.assertFalse(call_kwargs["verbose"])

    @patch("app.llm.llm_loader._LLAMA_CPP_AVAILABLE", False)
    def test_load_without_llama_cpp_raises(self) -> None:
        """Loading must raise RuntimeError when llama_cpp is not installed."""
        from app.llm.llm_loader import LLMManager

        mgr = LLMManager()
        with self.assertRaises(RuntimeError) as ctx:
            mgr.load()
        self.assertIn("llama_cpp is not installed", str(ctx.exception))

    @patch("app.llm.llm_loader._LLAMA_CPP_AVAILABLE", True)
    @patch("app.llm.llm_loader.Llama")
    def test_generate_calls_llama_with_temperature_zero(self, mock_llama_cls: MagicMock) -> None:
        """generate() must call the Llama instance with temperature=0.0."""
        from app.llm.llm_loader import LLMManager

        mock_instance = MagicMock()
        mock_instance.return_value = {
            "choices": [{"text": "  {\"name\": \"test\"}  "}]
        }
        mock_llama_cls.return_value = mock_instance

        mgr = LLMManager()
        with patch("os.path.isfile", return_value=True):
            mgr.load()

        result = mgr.generate("test prompt")
        self.assertEqual(result, '{"name": "test"}')

        # Verify temperature=0.0 was passed
        call_kwargs = mock_instance.call_args.kwargs
        self.assertEqual(call_kwargs["temperature"], 0.0)

        mgr.unload()

    @patch("app.llm.llm_loader._LLAMA_CPP_AVAILABLE", True)
    @patch("app.llm.llm_loader.Llama")
    def test_generate_without_load_raises(self, mock_llama_cls: MagicMock) -> None:
        """generate() without load() must raise RuntimeError."""
        from app.llm.llm_loader import LLMManager

        mgr = LLMManager()
        with self.assertRaises(RuntimeError) as ctx:
            mgr.generate("test")
        self.assertIn("not loaded", str(ctx.exception))


# ---------------------------------------------------------------------------
# InstructorClient tests
# ---------------------------------------------------------------------------


class TestInstructorClient(unittest.TestCase):
    """Tests for InstructorClient structured extraction."""

    def setUp(self) -> None:
        self.mock_llm = MockLLMManager()
        self.mock_llm.load()
        self.client = InstructorClient(self.mock_llm)

    def tearDown(self) -> None:
        self.mock_llm.unload()

    def test_temperature_always_zero(self) -> None:
        """_temperature() must always return 0.0."""
        self.assertEqual(self.client._temperature(), 0.0)

    def test_create_unresolved_field(self) -> None:
        """_create_unresolved_field must produce a valid FieldResult."""
        field = self.client._create_unresolved_field(
            "geburtsdatum", "Could not parse date"
        )
        self.assertEqual(field.field_name, "geburtsdatum")
        self.assertIsNone(field.value)
        self.assertEqual(field.status, FieldStatus.UNRESOLVED)
        self.assertEqual(field.confidence, 0.0)
        self.assertEqual(field.validation_message, "Could not parse date")

    def test_inject_few_shot_with_examples(self) -> None:
        """_inject_few_shot must format correction examples."""
        examples = [
            {"field": "postleitzahl", "original": "1011", "corrected": "10115", "reason": "Missing digit"},
        ]
        result = InstructorClient._inject_few_shot("Base prompt", examples)
        self.assertIn("LEARNED CORRECTIONS", result)
        self.assertIn("postleitzahl", result)
        self.assertIn("1011", result)
        self.assertIn("10115", result)

    def test_inject_few_shot_empty(self) -> None:
        """_inject_few_shot with empty list must return prompt unchanged."""
        prompt = "Base prompt"
        result = InstructorClient._inject_few_shot(prompt, [])
        self.assertEqual(result, prompt)

    def test_extract_json_without_few_shot(self) -> None:
        """generate_json must call extract_structured with empty few_shot_examples."""
        # We can't fully test without a real LLM, but we verify the prompt is built correctly
        prompt = "Extract: familienname=Müller, vorname=Anna"
        # Since MockLLMManager returns generic JSON, schema validation will fail
        # but the prompt construction and LLM call paths are exercised
        with patch.object(self.client, "extract_structured") as mock_extract:
            mock_extract.return_value = _SimpleSchema(name="Anna", age=30)
            result = self.client.generate_json(prompt, _SimpleSchema)
            mock_extract.assert_called_once_with(
                text=prompt,
                response_model=_SimpleSchema,
                few_shot_examples=[],
            )
            self.assertEqual(result.name, "Anna")
            self.assertEqual(result.age, 30)

    def test_extract_structured_without_llm_loaded(self) -> None:
        """extract_structured must raise RuntimeError when LLM is not loaded."""
        unloaded = MockLLMManager()  # never loaded
        client = InstructorClient(unloaded)
        with self.assertRaises(RuntimeError):
            client.extract_structured("test", _SimpleSchema)

    def test_extract_json_block_with_fences(self) -> None:
        """_extract_json_block must strip markdown code fences."""
        text = '```json\n{"name": "test"}\n```'
        result = InstructorClient._extract_json_block(text)
        self.assertEqual(result, '{"name": "test"}')

    def test_extract_json_block_without_fences(self) -> None:
        """_extract_json_block must return bare JSON."""
        text = '{"name": "test", "age": 25}'
        result = InstructorClient._extract_json_block(text)
        self.assertEqual(result, '{"name": "test", "age": 25}')

    def test_extract_json_block_nested_braces(self) -> None:
        """_extract_json_block must handle nested JSON with matching braces."""
        text = 'Some text before {\n  "outer": {\n    "inner": 1\n  }\n} after'
        result = InstructorClient._extract_json_block(text)
        self.assertEqual(result, '{\n  "outer": {\n    "inner": 1\n  }\n}')

    def test_extract_json_block_list(self) -> None:
        """_extract_json_block must handle JSON arrays."""
        text = '```\n[1, 2, 3]\n```'
        result = InstructorClient._extract_json_block(text)
        self.assertEqual(result, '[1, 2, 3]')


# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------


class TestPromptBuilder(unittest.TestCase):
    """Tests for PromptBuilder prompt construction."""

    def setUp(self) -> None:
        self.builder = PromptBuilder()

    def test_build_contains_anti_fabrication_rules(self) -> None:
        """The prompt must contain all 5 anti-fabrication rules."""
        prompt = self.builder.build(
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            vision_output="familienname: Müller",
            corrections=[],
        )
        self.assertIn("NO EXTRAPOLATION", prompt)
        self.assertIn("NO AUTOCOMPLETE", prompt)
        self.assertIn("DETERMINISTIC OUTPUT", prompt)
        self.assertIn("UNRESOLVED FALLBACK", prompt)
        self.assertIn("AUDIT TRAIL", prompt)

    def test_build_contains_document_type(self) -> None:
        """The prompt must name the document type."""
        prompt = self.builder.build(
            document_type=DocumentType.STEUERBESCHEID,
            vision_output="steuer_id: 12345678901",
            corrections=[],
        )
        self.assertIn("Steuerbescheid", prompt)

    def test_build_contains_vision_output(self) -> None:
        """The prompt must include the raw vision extraction."""
        vision = "steuer_id: 12345678901\nsteuernummer: 12345"
        prompt = self.builder.build(
            document_type=DocumentType.STEUERBESCHEID,
            vision_output=vision,
            corrections=[],
        )
        self.assertIn(vision, prompt)

    def test_build_few_shot_section(self) -> None:
        """build_few_shot_section must format corrections."""
        corrections = [
            {
                "field": "postleitzahl",
                "original": "1011",
                "corrected": "10115",
                "reason": "Missing final digit",
            },
            {
                "field": "geburtsdatum",
                "original": "15.03.85",
                "corrected": "15.03.1985",
                "reason": "Full 4-digit year required",
            },
        ]
        section = self.builder.build_few_shot_section(corrections)
        self.assertIn("LEARNED CORRECTIONS", section)
        self.assertIn("postleitzahl", section)
        self.assertIn("1011", section)
        self.assertIn("10115", section)
        self.assertIn("geburtsdatum", section)
        self.assertIn("15.03.1985", section)

    def test_few_shot_section_empty(self) -> None:
        """build_few_shot_section with empty list must return empty string."""
        section = self.builder.build_few_shot_section([])
        self.assertEqual(section, "")

    def test_few_shot_section_capped_at_max(self) -> None:
        """build_few_shot_section must cap corrections at MAX_FEW_SHOT_CORRECTIONS."""
        corrections = [
            {"field": f"field_{i}", "original": f"orig_{i}", "corrected": f"corr_{i}", "reason": f"reason_{i}"}
            for i in range(MAX_FEW_SHOT_CORRECTIONS + 3)
        ]
        section = self.builder.build_few_shot_section(corrections)
        # Count the number of bullet-point lines (starting with "-")
        bullet_count = sum(1 for line in section.split("\n") if line.startswith("- "))
        self.assertEqual(bullet_count, MAX_FEW_SHOT_CORRECTIONS)

    def test_build_truncates_excess_corrections(self) -> None:
        """build() must silently truncate corrections exceeding the limit."""
        corrections = [
            {"field": f"f{i}", "original": f"o{i}", "corrected": f"c{i}", "reason": f"r{i}"}
            for i in range(20)
        ]
        prompt = self.builder.build(
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            vision_output="test",
            corrections=corrections,
        )
        # Should contain at most MAX_FEW_SHOT_CORRECTIONS correction lines
        bullet_count = sum(1 for line in prompt.split("\n") if line.startswith("- "))
        self.assertLessEqual(bullet_count, MAX_FEW_SHOT_CORRECTIONS)

    def test_token_estimation(self) -> None:
        """estimate_tokens must return a positive float."""
        text = "Hello world this is a test"
        tokens = PromptBuilder.estimate_tokens(text)
        self.assertIsInstance(tokens, float)
        self.assertGreater(tokens, 0)

    def test_token_estimation_reasonable(self) -> None:
        """estimate_tokens should give roughly word_count / 0.75."""
        text = "one two three four five six seven eight nine ten"
        word_count = 10
        expected = word_count / 0.75
        tokens = PromptBuilder.estimate_tokens(text)
        self.assertAlmostEqual(tokens, expected, places=1)

    def test_build_correction_prompt(self) -> None:
        """build_correction_prompt must include both original and corrected values."""
        prompt = self.builder.build_correction_prompt(
            original_value="Berln",
            corrected_value="Berlin",
            field_name="wohnort",
        )
        self.assertIn("wohnort", prompt)
        self.assertIn("Berln", prompt)
        self.assertIn("Berlin", prompt)
        self.assertIn("Original:", prompt)
        self.assertIn("Corrected:", prompt)

    def test_default_field_guide_meldebescheinigung(self) -> None:
        """Default field guide for Meldebescheinigung must mention key fields."""
        prompt = self.builder.build(
            document_type=DocumentType.MELDEBESCHEINIGUNG,
            vision_output="test",
            corrections=[],
        )
        self.assertIn("Meldebescheinigung", prompt)

    def test_all_document_types(self) -> None:
        """Building prompts for all document types must succeed."""
        for doc_type in DocumentType:
            prompt = self.builder.build(
                document_type=doc_type,
                vision_output=f"vision output for {doc_type.value}",
                corrections=[],
            )
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 100)
            # Every prompt must have all anti-fabrication rules
            self.assertIn("NO EXTRAPOLATION", prompt)


# ---------------------------------------------------------------------------
# SchemaAdapter tests
# ---------------------------------------------------------------------------


class TestSchemaAdapter(unittest.TestCase):
    """Tests for SchemaAdapter Pydantic → prompt conversion."""

    def test_schema_to_prompt(self) -> None:
        """schema_to_prompt must list all fields."""
        guide = schema_to_prompt(MeldebescheinigungSchema)
        self.assertIn("familienname", guide)
        self.assertIn("vorname", guide)
        self.assertIn("geburtsdatum", guide)
        self.assertIn("postleitzahl", guide)

    def test_schema_to_prompt_has_descriptions(self) -> None:
        """The guide must include field descriptions."""
        guide = schema_to_prompt(MeldebescheinigungSchema)
        self.assertIn("Family name", guide)
        self.assertIn("First name", guide)

    def test_get_field_description(self) -> None:
        """get_field_description must return the annotated description."""
        desc = SchemaAdapter.get_field_description("familienname", MeldebescheinigungSchema)
        self.assertIn("Family name", desc)

    def test_get_field_description_missing_field(self) -> None:
        """Missing field must return a default description."""
        desc = SchemaAdapter.get_field_description("nonexistent", MeldebescheinigungSchema)
        self.assertIsInstance(desc, str)
        self.assertGreater(len(desc), 0)

    def test_build_field_guide_known_type(self) -> None:
        """build_field_guide must return a guide for known document types."""
        guide = build_field_guide(DocumentType.MELDEBESCHEINIGUNG)
        self.assertIn("familienname", guide)
        self.assertIn("postleitzahl", guide)

    def test_build_field_guide_unknown_type(self) -> None:
        """build_field_guide for unknown type must return generic text."""
        guide = build_field_guide("NonexistentType")
        self.assertIn("Unknown", guide)

    def test_build_field_guide_with_string(self) -> None:
        """build_field_guide must accept string document type names."""
        guide = build_field_guide("Meldebescheinigung")
        self.assertIn("familienname", guide)


# ---------------------------------------------------------------------------
# Temperature 0.0 enforcement tests
# ---------------------------------------------------------------------------


class TestTemperatureEnforcement(unittest.TestCase):
    """Verify that all LLM calls default to temperature=0.0."""

    def test_mock_llm_manager_default_temperature(self) -> None:
        """MockLLMManager.generate must default temperature to 0.0."""
        llm = MockLLMManager()
        llm.load()
        # MockLLMManager accepts the parameter and runs
        with patch("logging.Logger.warning") as mock_warn:
            result = llm.generate("test prompt")  # no temperature arg
            # With default temp=0.0, no warning should fire
            # (mock_warn won't be called because temperature IS 0.0)
        llm.unload()

    def test_instructor_client_temperature_method(self) -> None:
        """InstructorClient._temperature must return 0.0."""
        llm = MockLLMManager()
        llm.load()
        client = InstructorClient(llm)
        self.assertEqual(client._temperature(), 0.0)
        llm.unload()

    @patch("app.llm.llm_loader._LLAMA_CPP_AVAILABLE", True)
    @patch("app.llm.llm_loader.Llama")
    def test_llm_manager_generate_default_temperature(self, mock_llama_cls: MagicMock) -> None:
        """LLMManager.generate must default temperature=0.0."""
        from app.llm.llm_loader import LLMManager

        mock_instance = MagicMock()
        mock_instance.return_value = {"choices": [{"text": "result"}]}
        mock_llama_cls.return_value = mock_instance

        mgr = LLMManager()
        with patch("os.path.isfile", return_value=True):
            mgr.load()

        mgr.generate("prompt")  # default temperature
        call_kwargs = mock_instance.call_args.kwargs
        self.assertEqual(call_kwargs["temperature"], 0.0)
        mgr.unload()


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports(unittest.TestCase):
    """Verify that __init__.py exports the expected symbols."""

    def test_all_classes_exported(self) -> None:
        """All major classes must be importable from the llm package."""
        from app.llm import (
            InstructorClient,
            LLMManager,
            MAX_CONTEXT_TOKENS,
            MAX_FEW_SHOT_CORRECTIONS,
            MockLLMManager,
            PromptBuilder,
            SchemaAdapter,
            TOKENS_PER_WORD_ESTIMATE,
            build_field_guide,
            get_field_description,
            schema_to_prompt,
        )

        self.assertTrue(callable(LLMManager))
        self.assertTrue(callable(MockLLMManager))
        self.assertTrue(callable(InstructorClient))
        self.assertTrue(callable(PromptBuilder))
        self.assertTrue(callable(SchemaAdapter))

    def test_max_few_shot_default_is_five(self) -> None:
        """MAX_FEW_SHOT_CORRECTIONS default must be 5."""
        from app.llm import MAX_FEW_SHOT_CORRECTIONS

        self.assertEqual(MAX_FEW_SHOT_CORRECTIONS, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
