"""Helper for converting Pydantic schemas to LLM-friendly prompt descriptions.

``SchemaAdapter`` introspects Pydantic models and produces human-readable
field guides that can be inserted into extraction prompts.  This bridges
the gap between strict schema definitions and the free-text context that
helps an LLM produce correct output.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

from app.models.document_schemas import (
    DOCUMENT_SCHEMAS,
    get_schema_for_document_type,
)
from app.models.enums import DocumentType

logger = logging.getLogger(__name__)

# Mapping of Python types to human-readable descriptions
_TYPE_DESCRIPTIONS: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _unwrap_optional(annotation: Any) -> tuple[type, bool]:
    """Determine the inner type and optionality of an annotation.

    Handles ``Optional[T]``, ``Union[T, None]``, and bare ``T``.

    Args:
        annotation: A type annotation (possibly from Pydantic FieldInfo).

    Returns:
        Tuple of (inner_type, is_optional).
    """
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if type(None) in args and non_none:
            return non_none[0], True
        if non_none:
            return non_none[0], False
        return str, False
    if origin is Optional:
        args = get_args(annotation)
        if args:
            return args[0], True
    # For date fields, unwrap to date
    if annotation == "date":
        from datetime import date as _date
        return _date, False
    return annotation, False


class SchemaAdapter:
    """Converts Pydantic schemas into LLM-friendly prompt text."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def schema_to_prompt(schema_class: Type[BaseModel]) -> str:
        """Convert a Pydantic model to a human-readable field listing.

        Each field is rendered as::

            - field_name (type, optional?): description [pattern]

        Args:
            schema_class: A Pydantic ``BaseModel`` subclass.

        Returns:
            A multi-line string describing all model fields.
        """
        lines: list[str] = []
        title = getattr(schema_class, "model_config", {}).get("title", schema_class.__name__)
        lines.append(f"Schema: {title}")
        lines.append("-" * 40)

        for field_name, field_info in schema_class.model_fields.items():
            description = SchemaAdapter.get_field_description(field_name, schema_class)
            field_line = SchemaAdapter._format_field(field_name, field_info, description)
            lines.append(field_line)

        return "\n".join(lines)

    @staticmethod
    def get_field_description(
        field_name: str,
        schema_class: Type[BaseModel],
    ) -> str:
        """Get the description annotation for a single field.

        Args:
            field_name: Name of the field in the schema.
            schema_class: The Pydantic model class.

        Returns:
            The field's description string, or a default description
            derived from the field name if none is set.
        """
        field_info = schema_class.model_fields.get(field_name)
        if field_info is None:
            return SchemaAdapter._default_description(field_name)

        # Pydantic v2 stores description in field_info.description
        if hasattr(field_info, "description") and field_info.description:
            return field_info.description

        # Also check json_schema_extra
        if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
            extra = field_info.json_schema_extra
            if isinstance(extra, dict):
                desc = extra.get("description", "")
                if desc:
                    return desc

        return SchemaAdapter._default_description(field_name)

    @staticmethod
    def build_field_guide(document_type: Union[DocumentType, str]) -> str:
        """Build a human-readable field guide for a document type.

        Looks up the registered schema for the document type and converts
        it to a prompt-friendly field listing.

        Args:
            document_type: A ``DocumentType`` enum member or its string value.

        Returns:
            A formatted field guide string. If the document type is unknown,
            returns a generic extraction guide.
        """
        if isinstance(document_type, DocumentType):
            type_name = document_type.value
        else:
            type_name = document_type

        schema_class = get_schema_for_document_type(type_name)
        if schema_class is None:
            logger.warning("No schema registered for document type: %s", type_name)
            return (
                f"Unknown document type '{type_name}'. "
                "Extract any clearly legible fields. "
                "Mark unclear fields as UNRESOLVED."
            )

        return SchemaAdapter.schema_to_prompt(schema_class)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_field(
        field_name: str,
        field_info: FieldInfo,
        description: str,
    ) -> str:
        """Format a single field as a human-readable line.

        Args:
            field_name: The field name.
            field_info: Pydantic FieldInfo for the field.
            description: The field description.

        Returns:
            Formatted line like ``- field_name (string, optional): description [pattern]``
        """
        # Determine type and optionality
        annotation = field_info.annotation
        inner_type, is_optional = _unwrap_optional(annotation)

        type_name = _TYPE_DESCRIPTIONS.get(inner_type, str(inner_type.__name__))

        parts: list[str] = [f"- {field_name} ({type_name}"]
        if is_optional:
            parts[0] += ", optional"
        parts[0] += ")"

        if description:
            parts.append(f": {description}")

        # Include pattern if present
        if hasattr(field_info, "json_schema_extra") and field_info.json_schema_extra:
            extra = field_info.json_schema_extra
            if isinstance(extra, dict) and "pattern" in extra:
                parts.append(f" [pattern: {extra['pattern']}]")

        # Include pattern from Field metadata (Pydantic v2 pattern attribute)
        if hasattr(field_info, "pattern") and field_info.pattern:
            parts.append(f" [pattern: {field_info.pattern}]")

        return "".join(parts)

    @staticmethod
    def _default_description(field_name: str) -> str:
        """Generate a default description from a field name.

        Converts snake_case to spaced words and title-cases.

        Args:
            field_name: The field name (e.g. ``geburtsdatum``).

        Returns:
            A human-readable description.
        """
        # Convert snake_case to human readable
        words = field_name.replace("_", " ").split()
        return " ".join(w.capitalize() for w in words)


# Convenience module-level functions

def schema_to_prompt(schema_class: Type[BaseModel]) -> str:
    """Module-level convenience wrapper for :meth:`SchemaAdapter.schema_to_prompt`."""
    return SchemaAdapter.schema_to_prompt(schema_class)


def get_field_description(field_name: str, schema_class: Type[BaseModel]) -> str:
    """Module-level convenience wrapper for :meth:`SchemaAdapter.get_field_description`."""
    return SchemaAdapter.get_field_description(field_name, schema_class)


def build_field_guide(document_type: Union[DocumentType, str]) -> str:
    """Module-level convenience wrapper for :meth:`SchemaAdapter.build_field_guide`."""
    return SchemaAdapter.build_field_guide(document_type)
