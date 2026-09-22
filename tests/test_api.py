import asyncio
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
@patch("app.api.routes.generate_answer_async")
def test_qa_endpoint_success(
    mock_generate_answer_async,
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

    mock_generate_answer_async.return_value = GroundedAnswer(
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
    mock_generate_answer_async.assert_awaited_once()


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

def test_qa_endpoint_rejects_too_many_questions():
    questions = [
        {
            "id": f"q{index}",
            "question": f"Question {index}?",
        }
        for index in range(1, 52)
    ]

    import json

    questions_content = json.dumps(
        {"questions": questions}
    ).encode("utf-8")

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
                b'{"cloud": "GCP"}',
                "application/json",
            ),
        },
    )

    assert response.status_code == 413

    assert response.json()["detail"] == (
        "Too many questions. Maximum allowed is "
        "50 questions per request."
    )

def test_qa_endpoint_rejects_oversized_questions_file():
    oversized_questions = b"x" * (1024 * 1024 + 1)

    response = client.post(
        "/api/v1/qa",
        files={
            "questions_file": (
                "questions.json",
                oversized_questions,
                "application/json",
            ),
            "document_file": (
                "document.json",
                b'{"cloud": "GCP"}',
                "application/json",
            ),
        },
    )

    assert response.status_code == 413

    assert response.json()["detail"] == (
        "Questions file is too large. "
        "Maximum allowed size is 1 MB."
    )

def test_qa_endpoint_rejects_oversized_document_file():
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

    oversized_document = b"x" * (20 * 1024 * 1024 + 1)

    response = client.post(
        "/api/v1/qa",
        files={
            "questions_file": (
                "questions.json",
                questions_content,
                "application/json",
            ),
            "document_file": (
                "document.pdf",
                oversized_document,
                "application/pdf",
            ),
        },
    )

    assert response.status_code == 413

    assert response.json()["detail"] == (
        "Document file is too large. "
        "Maximum allowed size is 20 MB."
    )

def test_qa_endpoint_processes_questions_concurrently():
    active_calls = 0
    max_active_calls = 0
    lock = asyncio.Lock()

    async def mock_generate_answer_async(
        question,
        documents,
    ):
        nonlocal active_calls, max_active_calls

        async with lock:
            active_calls += 1
            max_active_calls = max(
                max_active_calls,
                active_calls,
            )

        # Give the other question tasks time to start.
        await asyncio.sleep(0.05)

        async with lock:
            active_calls -= 1

        return GroundedAnswer(
            answer=f"Answer for {question}",
            confidence="high",
            evidence_indices=[0],
        )

    retrieved_document = Document(
        page_content="Nave uses Google Cloud Platform.",
        metadata={
            "source": "document.json",
            "page": None,
        },
    )

    questions_content = b"""
    {
        "questions": [
            {
                "id": "q1",
                "question": "Question one?"
            },
            {
                "id": "q2",
                "question": "Question two?"
            },
            {
                "id": "q3",
                "question": "Question three?"
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

    with (
        patch(
            "app.api.routes.build_vector_store",
            return_value=object(),
        ),
        patch(
            "app.api.routes.retrieve_documents",
            return_value=[retrieved_document],
        ),
        patch(
            "app.api.routes.generate_answer_async",
            side_effect=mock_generate_answer_async,
        ),
    ):
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

    results = response.json()["results"]

    # asyncio.gather must preserve the original input order.
    assert [result["id"] for result in results] == [
        "q1",
        "q2",
        "q3",
    ]

    # More than one active LLM call proves concurrent processing.
    assert max_active_calls > 1