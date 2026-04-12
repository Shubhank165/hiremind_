"""
Resume parsing utility.
Supports PDF and plain text resume files.
Extracts raw text for the Profile Extraction Agent.
"""

import io
from PyPDF2 import PdfReader


def parse_resume(file_bytes: bytes, filename: str) -> str:
    """
    Parse a resume file and extract text content.

    Args:
        file_bytes: Raw bytes of the uploaded file
        filename: Original filename (used to detect format)

    Returns:
        Extracted text content from the resume
    """
    if filename.lower().endswith(".pdf"):
        return _parse_pdf(file_bytes)
    else:
        return _parse_text(file_bytes)


def _parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text.strip())

    full_text = "\n\n".join(text_parts)

    # Clean up common PDF artifacts
    full_text = full_text.replace("\x00", "")
    # Collapse excessive whitespace but preserve paragraph breaks
    lines = full_text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

    return "\n".join(cleaned_lines)


def _parse_text(file_bytes: bytes) -> str:
    """Extract text from a plain text file."""
    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return file_bytes.decode("latin-1").strip()
