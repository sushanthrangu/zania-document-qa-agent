from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import QAResponse
from app.services.chunking import chunk_documents
from app.services.document_loader import (
    DocumentLoadError,
    load_document,
)
from app.services.qa_service import (
    build_answer_result,
    generate_answer,
)
from app.services.question_parser import (
    QuestionParseError,
    parse_questions,
)
from app.services.retriever import retrieve_documents
from app.services.vector_store import build_vector_store


router = APIRouter(
    prefix="/api/v1",
    tags=["Document QA"],
)


@router.post("/qa", response_model=QAResponse)
async def answer_questions(
    questions_file: UploadFile = File(...),
    document_file: UploadFile = File(...),
) -> QAResponse:
    """Answer questions using the uploaded document."""

    questions_content = await questions_file.read()
    document_content = await document_file.read()

    try:
        questions = parse_questions(questions_content)

        documents = load_document(
            content=document_content,
            filename=document_file.filename or "",
        )

    except (QuestionParseError, DocumentLoadError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    chunks = chunk_documents(documents)
    vector_store = build_vector_store(chunks)

    results = []

    for question in questions:
        retrieved_documents = retrieve_documents(
            vector_store=vector_store,
            query=question.question,
        )

        grounded_answer = generate_answer(
            question=question.question,
            documents=retrieved_documents,
        )

        answer_result = build_answer_result(
            question=question,
            grounded_answer=grounded_answer,
            documents=retrieved_documents,
        )

        results.append(answer_result)

    return QAResponse(
        results=results,
    )