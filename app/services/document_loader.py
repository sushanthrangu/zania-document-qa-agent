import json
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


class DocumentLoadError(ValueError):
    """Raised when an uploaded document cannot be loaded."""


SUPPORTED_DOCUMENT_TYPES = {".pdf", ".json"}


def load_document(
    content: bytes,
    filename: str,
) -> list[Document]:
    """Load an uploaded PDF or JSON document into LangChain Documents."""

    if not filename or not filename.strip():
        raise DocumentLoadError(
            "Document filename is required."
        )

    if not content:
        raise DocumentLoadError(
            "Document file cannot be empty."
        )

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_DOCUMENT_TYPES:
        raise DocumentLoadError(
            "Unsupported document type. Only PDF and JSON are supported."
        )

    if extension == ".pdf":
        return _load_pdf(content, filename)

    return _load_json(content, filename)


def _load_pdf(
    content: bytes,
    filename: str,
) -> list[Document]:
    """Extract text from a PDF while preserving page metadata."""

    documents: list[Document] = []

    try:
        with pymupdf.open(stream=content, filetype="pdf") as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_number,
                        },
                    )
                )

    except Exception as exc:
        raise DocumentLoadError(
            "Unable to read the PDF document."
        ) from exc

    if not documents:
        raise DocumentLoadError(
            "PDF does not contain extractable text."
        )

    return documents

def _flatten_json(
    data: object,
    parent_key: str = "",
) -> list[str]:
    """Flatten nested JSON into searchable key-value text lines."""

    lines: list[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_key = f"{parent_key}.{key}" if parent_key else str(key)
            lines.extend(_flatten_json(value, current_key))

    elif isinstance(data, list):
        for index, value in enumerate(data):
            current_key = f"{parent_key}[{index}]"
            lines.extend(_flatten_json(value, current_key))

    else:
        if data is None:
            value = "null"
        elif isinstance(data, bool):
            value = str(data).lower()
        else:
            value = str(data)

        key = parent_key or "value"
        lines.append(f"{key}: {value}")

    return lines

def _load_json(
    content: bytes,
    filename: str,
) -> list[Document]:
    """Load and normalize JSON content into a LangChain Document."""

    try:
        data = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DocumentLoadError(
            "Unable to read the JSON document."
        ) from exc

    lines = _flatten_json(data)

    if not lines:
        raise DocumentLoadError(
            "JSON document does not contain usable content."
        )

    text = "\n".join(lines)

    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": None,
            },
        )
    ]