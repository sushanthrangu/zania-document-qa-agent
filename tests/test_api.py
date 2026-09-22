from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.services.qa_service import GroundedAnswer


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
    }


@patch("app.api.routes.build_vector_store")
@patch("app.api.routes.retrieve_documents")
@patch("app.api.routes.generate_answer")
def test_qa_endpoint_success(
    mock_generate_answer,
    mock_retrieve_documents,
    mock_build_vector_store,
):
    mock_vector_store = object()
    mock_build_vector_store.return_value = mock_vector_store

    retrieved_document = Document(
        page_content="Nave uses Google Cloud Platform.",
        metadata={
            "source": "document.json",
            "page": None,
        },
    )

    mock_retrieve_documents.return_value = [
        retrieved_document,
    ]

    mock_generate_answer.return_value = GroundedAnswer(
        answer="Google Cloud Platform (GCP).",
        confidence="high",
        evidence_indices=[0],
    )

    questions_content = b"""
    {
        "questions": [
            {
                "id": "q1",
                "question": "Which cloud provider is used?"
            }
        ]
    }
    """

    document_content = b"""
    {
        "cloud": {
            "provider": "Google Cloud Platform"
        }
    }
    """

    response = client.post(
        "/api/v1/qa",
        files={
            "questions_file": (
                "questions.json",
                questions_content,
                "application/json",
            ),
            "document_file": (
                "document.json",
                document_content,
                "application/json",
            ),
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["results"]) == 1

    result = data["results"][0]

    assert result["id"] == "q1"
    assert result["question"] == "Which cloud provider is used?"
    assert result["answer"] == "Google Cloud Platform (GCP)."
    assert result["confidence"] == "high"

    assert result["sources"] == [
        {
            "source": "document.json",
            "page": None,
        }
    ]

    mock_build_vector_store.assert_called_once()
    mock_retrieve_documents.assert_called_once()
    mock_generate_answer.assert_called_once()


def test_qa_endpoint_invalid_questions_json():
    response = client.post(
        "/api/v1/qa",
        files={
            "questions_file": (
                "questions.json",
                b"{invalid json}",
                "application/json",
            ),
            "document_file": (
                "document.json",
                b'{"cloud": "GCP"}',
                "application/json",
            ),
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "Questions file must contain valid UTF-8 JSON."
    )


def test_qa_endpoint_unsupported_document_type():
    questions_content = b"""
    {
        "questions": [
            {
                "id": "q1",
                "question": "Which cloud provider is used?"
            }
        ]
    }
    """

    response = client.post(
        "/api/v1/qa",
        files={
            "questions_file": (
                "questions.json",
                questions_content,
                "application/json",
            ),
            "document_file": (
                "document.txt",
                b"Nave uses GCP.",
                "text/plain",
            ),
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == (
            "Unsupported document type. "
            "Only PDF and JSON are supported."
        )
    )


def test_qa_endpoint_empty_questions_list():
    response = client.post(
        "/api/v1/qa",
        files={
            "questions_file": (
                "questions.json",
                b'{"questions": []}',
                "application/json",
            ),
            "document_file": (
                "document.json",
                b'{"cloud": "GCP"}',
                "application/json",
            ),
        },
    )

    assert response.status_code == 400

    assert (
        response.json()["detail"]
        == "Questions list cannot be empty."
    )


def test_qa_endpoint_requires_questions_file():
    response = client.post(
        "/api/v1/qa",
        files={
            "document_file": (
                "document.json",
                b'{"cloud": "GCP"}',
                "application/json",
            ),
        },
    )

    assert response.status_code == 422


def test_qa_endpoint_requires_document_file():
    response = client.post(
        "/api/v1/qa",
        files={
            "questions_file": (
                "questions.json",
                b'{"questions": ["Test question?"]}',
                "application/json",
            ),
        },
    )

    assert response.status_code == 422