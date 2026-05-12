"""
Best-effort text extraction for non-image document uploads.

The vision pipeline remains the preferred path for images.  This extractor
handles text-bearing files so the system can return useful data for PDFs,
plain text, CSV/TSV, JSON, XML/HTML, Markdown, and DOCX without fabricating.
Optional dependencies are used when present; otherwise the extractor falls back
to safe standard-library parsing.
"""

from __future__ import annotations

import csv
import json
import logging
import mimetypes
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedFileText:
    """Text and metadata extracted from an uploaded file."""

    text: str
    file_type: str
    parser: str


class FileTextExtractor:
    """Extract text from common non-image file types."""

    TEXT_EXTENSIONS = {
        ".txt",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".md",
        ".log",
    }

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def is_image(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.IMAGE_EXTENSIONS

    def extract(self, path: str) -> ExtractedFileText:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        mime, _ = mimetypes.guess_type(file_path.name)

        if suffix == ".pdf":
            return ExtractedFileText(
                text=self._extract_pdf(file_path),
                file_type=mime or "application/pdf",
                parser="pdf",
            )
        if suffix == ".docx":
            return ExtractedFileText(
                text=self._extract_docx(file_path),
                file_type=mime or "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                parser="docx",
            )
        if suffix in (".csv", ".tsv"):
            return ExtractedFileText(
                text=self._extract_table(file_path, delimiter="\t" if suffix == ".tsv" else ","),
                file_type=mime or "text/csv",
                parser="csv",
            )
        if suffix == ".json":
            return ExtractedFileText(
                text=self._extract_json(file_path),
                file_type=mime or "application/json",
                parser="json",
            )
        if suffix in self.TEXT_EXTENSIONS:
            return ExtractedFileText(
                text=self._read_text(file_path),
                file_type=mime or "text/plain",
                parser="text",
            )

        return ExtractedFileText(
            text=self._read_text(file_path),
            file_type=mime or "application/octet-stream",
            parser="binary-text-fallback",
        )

    def _extract_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(page for page in pages if page.strip())
        except Exception as exc:
            logger.warning("PDF text extraction failed for %s: %s", path, exc)
            return self._read_text(path)

    def _extract_docx(self, path: Path) -> str:
        try:
            with zipfile.ZipFile(path) as archive:
                xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
            text = re.sub(r"<[^>]+>", " ", xml)
            return re.sub(r"\s+", " ", text).strip()
        except Exception as exc:
            logger.warning("DOCX text extraction failed for %s: %s", path, exc)
            return self._read_text(path)

    def _extract_table(self, path: Path, delimiter: str) -> str:
        rows: list[str] = []
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            for row in reader:
                rows.append(" | ".join(cell.strip() for cell in row))
        return "\n".join(rows)

    def _extract_json(self, path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            return self._read_text(path)

    def _read_text(self, path: Path) -> str:
        data = path.read_bytes()
        if b"\x00" in data[:4096]:
            data = data.replace(b"\x00", b" ")
        return data.decode("utf-8", errors="ignore")[:200_000]
