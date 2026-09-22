from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.models.schemas import Question
from app.services.qa_service import (
    GroundedAnswer,
    build_answer_result,
    format_context,
    generate_answer,
)


def test_format_context_with_page():
    documents = [
        Document(
            page_content="Nave uses Google Cloud Platform.",
            metadata={
                "source": "report.pdf",
                "page": 17,
            },
        )
    ]

    context = format_context(documents)

    assert "[Context 0]" in context
    assert "[Source: report.pdf | Page: 17]" in context
    assert "Nave uses Google Cloud Platform." in context


def test_format_context_without_page():
    documents = [
        Document(
            page_content="cloud.provider: GCP",
            metadata={
                "source": "document.json",
                "page": None,
            },
        )
    ]

    context = format_context(documents)

    assert "[Context 0]" in context
    assert "[Source: document.json]" in context
    assert "cloud.provider: GCP" in context


def test_generate_answer_rejects_blank_question():
    with pytest.raises(
        ValueError,
        match="Question cannot be empty",
    ):
        generate_answer(
            question="   ",
            documents=[],
        )


def test_generate_answer_without_documents_returns_not_found():
    result = generate_answer(
        question="Which cloud provider is used?",
        documents=[],
    )

    assert result.answer == "Data-Not-Found"
    assert result.confidence == "low"
    assert result.evidence_indices == []


@patch("app.services.qa_service.get_chat_model")
def test_generate_grounded_answer(mock_get_chat_model):
    structured_model = MagicMock()

    structured_model.invoke.return_value = GroundedAnswer(
        answer="Google Cloud Platform (GCP).",
        confidence="high",
        evidence_indices=[0],
    )

    chat_model = MagicMock()
    chat_model.with_structured_output.return_value = structured_model
    mock_get_chat_model.return_value = chat_model

    documents = [
        Document(
            page_content=(
                "Nave uses Google Cloud Platform "
                "for cloud hosting."
            ),
            metadata={
                "source": "report.pdf",
                "page": 17,
            },
        )
    ]

    result = generate_answer(
        question="Which cloud provider is used?",
        documents=documents,
    )

    assert result.answer == "Google Cloud Platform (GCP)."
    assert result.confidence == "high"
    assert result.evidence_indices == [0]

    structured_model.invoke.assert_called_once()


@patch("app.services.qa_service.get_chat_model")
def test_generate_exact_data_not_found_is_normalized(
    mock_get_chat_model,
):
    structured_model = MagicMock()

    structured_model.invoke.return_value = GroundedAnswer(
        answer="Data-Not-Found",
        confidence="medium",
        evidence_indices=[0],
    )

    chat_model = MagicMock()
    chat_model.with_structured_output.return_value = structured_model
    mock_get_chat_model.return_value = chat_model

    documents = [
        Document(
            page_content="Unrelated document content.",
            metadata={
                "source": "report.pdf",
                "page": 1,
            },
        )
    ]

    result = generate_answer(
        question="What is the notification SLA?",
        documents=documents,
    )

    assert result.answer == "Data-Not-Found"
    assert result.confidence == "low"
    assert result.evidence_indices == []


@patch("app.services.qa_service.get_chat_model")
def test_partial_answer_is_not_destroyed_by_not_found_text(
    mock_get_chat_model,
):
    structured_model = MagicMock()

    structured_model.invoke.return_value = GroundedAnswer(
        answer=(
            "Nave uses GCP backups across multiple availability "
            "zones. The exact geographic backup location was not "
            "identified in the provided context."
        ),
        confidence="medium",
        evidence_indices=[0],
    )

    chat_model = MagicMock()
    chat_model.with_structured_output.return_value = structured_model
    mock_get_chat_model.return_value = chat_model

    documents = [
        Document(
            page_content=(
                "Nave maintains multi-availability zone deployment "
                "on GCP."
            ),
            metadata={
                "source": "report.pdf",
                "page": 22,
            },
        )
    ]

    result = generate_answer(
        question="Where are primary and backup systems located?",
        documents=documents,
    )

    assert result.answer != "Data-Not-Found"
    assert "GCP" in result.answer
    assert result.confidence == "medium"
    assert result.evidence_indices == [0]


@patch("app.services.qa_service.get_chat_model")
def test_blank_model_answer_returns_not_found(
    mock_get_chat_model,
):
    structured_model = MagicMock()

    structured_model.invoke.return_value = GroundedAnswer(
        answer="   ",
        confidence="high",
        evidence_indices=[0],
    )

    chat_model = MagicMock()
    chat_model.with_structured_output.return_value = structured_model
    mock_get_chat_model.return_value = chat_model

    documents = [
        Document(
            page_content="Some document content.",
            metadata={
                "source": "report.pdf",
                "page": 1,
            },
        )
    ]

    result = generate_answer(
        question="Test question?",
        documents=documents,
    )

    assert result.answer == "Data-Not-Found"
    assert result.confidence == "low"
    assert result.evidence_indices == []


def test_build_answer_result_uses_only_evidence_indices():
    question = Question(
        id="q1",
        question="Which cloud provider is used?",
    )

    documents = [
        Document(
            page_content="GCP is the cloud provider.",
            metadata={
                "source": "report.pdf",
                "page": 17,
            },
        ),
        Document(
            page_content="Unrelated content.",
            metadata={
                "source": "report.pdf",
                "page": 40,
            },
        ),
    ]

    grounded_answer = GroundedAnswer(
        answer="Google Cloud Platform (GCP).",
        confidence="high",
        evidence_indices=[0],
    )

    result = build_answer_result(
        question=question,
        grounded_answer=grounded_answer,
        documents=documents,
    )

    assert len(result.sources) == 1
    assert result.sources[0].source == "report.pdf"
    assert result.sources[0].page == 17


def test_build_answer_result_deduplicates_sources():
    question = Question(
        id="q1",
        question="Which cloud provider is used?",
    )

    documents = [
        Document(
            page_content="GCP hosting.",
            metadata={
                "source": "report.pdf",
                "page": 17,
            },
        ),
        Document(
            page_content="More GCP information.",
            metadata={
                "source": "report.pdf",
                "page": 17,
            },
        ),
    ]

    grounded_answer = GroundedAnswer(
        answer="Google Cloud Platform (GCP).",
        confidence="high",
        evidence_indices=[0, 1],
    )

    result = build_answer_result(
        question=question,
        grounded_answer=grounded_answer,
        documents=documents,
    )

    assert len(result.sources) == 1
    assert result.sources[0].page == 17


def test_build_answer_result_ignores_invalid_evidence_indices():
    question = Question(
        id="q1",
        question="Which cloud provider is used?",
    )

    documents = [
        Document(
            page_content="GCP hosting.",
            metadata={
                "source": "report.pdf",
                "page": 17,
            },
        )
    ]

    grounded_answer = GroundedAnswer(
        answer="Google Cloud Platform (GCP).",
        confidence="high",
        evidence_indices=[0, 99, -1],
    )

    result = build_answer_result(
        question=question,
        grounded_answer=grounded_answer,
        documents=documents,
    )

    assert len(result.sources) == 1
    assert result.sources[0].page == 17


def test_data_not_found_has_no_sources():
    question = Question(
        id="q1",
        question="What is the notification SLA?",
    )

    documents = [
        Document(
            page_content="Some unrelated content.",
            metadata={
                "source": "report.pdf",
                "page": 10,
            },
        )
    ]

    grounded_answer = GroundedAnswer(
        answer="Data-Not-Found",
        confidence="low",
        evidence_indices=[0],
    )

    result = build_answer_result(
        question=question,
        grounded_answer=grounded_answer,
        documents=documents,
    )

    assert result.sources == []