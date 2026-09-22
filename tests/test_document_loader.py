import json

import pymupdf
import pytest

from app.services.document_loader import (
    DocumentLoadError,
    load_document,
)


def create_test_pdf(pages: list[str]) -> bytes:
    """Create an in-memory PDF for testing."""

    pdf = pymupdf.open()

    for text in pages:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            text,
        )

    content = pdf.tobytes()
    pdf.close()

    return content


def test_load_json_document():
    content = json.dumps(
        {
            "company": "Nave",
            "cloud": {
                "provider": "GCP",
                "region": "US Central",
            },
        }
    ).encode("utf-8")

    documents = load_document(
        content=content,
        filename="document.json",
    )

    assert len(documents) == 1

    document = documents[0]

    assert "company: Nave" in document.page_content
    assert "cloud.provider: GCP" in document.page_content
    assert "cloud.region: US Central" in document.page_content

    assert document.metadata["source"] == "document.json"
    assert document.metadata["page"] is None


def test_load_json_array():
    content = json.dumps(
        {
            "providers": [
                "GCP",
                "AWS",
            ]
        }
    ).encode("utf-8")

    documents = load_document(
        content=content,
        filename="document.json",
    )

    text = documents[0].page_content

    assert "providers[0]: GCP" in text
    assert "providers[1]: AWS" in text


def test_load_nested_json_preserves_paths():
    content = json.dumps(
        {
            "security": {
                "encryption": {
                    "at_rest": "AES-256",
                }
            }
        }
    ).encode("utf-8")

    documents = load_document(
        content=content,
        filename="security.json",
    )

    assert (
        "security.encryption.at_rest: AES-256"
        in documents[0].page_content
    )


def test_invalid_json_raises_error():
    with pytest.raises(
        DocumentLoadError,
        match="Unable to read the JSON document",
    ):
        load_document(
            content=b"{invalid json}",
            filename="document.json",
        )


@pytest.mark.parametrize(
    "content",
    [
        b"{}",
        b"[]",
    ],
)
def test_empty_json_raises_error(content):
    with pytest.raises(
        DocumentLoadError,
        match="does not contain usable content",
    ):
        load_document(
            content=content,
            filename="document.json",
        )


def test_load_pdf_preserves_pages():
    content = create_test_pdf(
        [
            "Nave uses Google Cloud Platform.",
            "Backups are performed daily.",
        ]
    )

    documents = load_document(
        content=content,
        filename="report.pdf",
    )

    assert len(documents) == 2

    assert (
        "Nave uses Google Cloud Platform"
        in documents[0].page_content
    )
    assert documents[0].metadata["source"] == "report.pdf"
    assert documents[0].metadata["page"] == 1

    assert (
        "Backups are performed daily"
        in documents[1].page_content
    )
    assert documents[1].metadata["page"] == 2


def test_blank_pdf_raises_error():
    pdf = pymupdf.open()
    pdf.new_page()

    content = pdf.tobytes()
    pdf.close()

    with pytest.raises(
        DocumentLoadError,
        match="does not contain extractable text",
    ):
        load_document(
            content=content,
            filename="blank.pdf",
        )


def test_corrupt_pdf_raises_error():
    with pytest.raises(
        DocumentLoadError,
        match="Unable to read the PDF document",
    ):
        load_document(
            content=b"This is not a real PDF.",
            filename="report.pdf",
        )


def test_unsupported_document_type_raises_error():
    with pytest.raises(
        DocumentLoadError,
        match="Only PDF and JSON are supported",
    ):
        load_document(
            content=b"hello",
            filename="document.txt",
        )


def test_empty_document_raises_error():
    with pytest.raises(
        DocumentLoadError,
        match="cannot be empty",
    ):
        load_document(
            content=b"",
            filename="document.pdf",
        )


def test_missing_filename_raises_error():
    with pytest.raises(
        DocumentLoadError,
        match="filename is required",
    ):
        load_document(
            content=b"content",
            filename="",
        )