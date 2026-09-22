import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.schemas import QAResponse
from app.services.chunking import chunk_documents
from app.services.document_loader import (
    DocumentLoadError,
    load_document,
)
from app.services.qa_service import (
    build_answer_result,
    generate_answer_async,
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


async def read_upload_with_limit(
    upload: UploadFile,
    max_size_mb: int,
    label: str,
) -> bytes:
    """Read an uploaded file without exceeding the configured size limit."""

    max_bytes = max_size_mb * 1024 * 1024

    # Read only one byte beyond the allowed limit.
    # This prevents loading arbitrarily large uploads into memory.
    content = await upload.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{label} is too large. Maximum allowed size is "
                f"{max_size_mb} MB."
            ),
        )

    return content


async def process_question(
    question,
    vector_store,
    semaphore: asyncio.Semaphore,
):
    """Process one question while respecting the concurrency limit."""

    async with semaphore:
        # FAISS retrieval is synchronous, so move it off the event loop.
        retrieved_documents = await asyncio.to_thread(
            retrieve_documents,
            vector_store,
            question.question,
        )

        # Use LangChain's native async LLM interface.
        grounded_answer = await generate_answer_async(
            question=question.question,
            documents=retrieved_documents,
        )

        return build_answer_result(
            question=question,
            grounded_answer=grounded_answer,
            documents=retrieved_documents,
        )


@router.post("/qa", response_model=QAResponse)
async def answer_questions(
    questions_file: UploadFile = File(...),
    document_file: UploadFile = File(...),
) -> QAResponse:
    """Answer questions using the uploaded document."""

    settings = get_settings()

    # Protect the service from excessively large uploads.
    questions_content = await read_upload_with_limit(
        upload=questions_file,
        max_size_mb=settings.max_questions_file_size_mb,
        label="Questions file",
    )

    document_content = await read_upload_with_limit(
        upload=document_file,
        max_size_mb=settings.max_document_file_size_mb,
        label="Document file",
    )

    try:
        questions = parse_questions(questions_content)

        # Prevent excessive retrieval and LLM calls in one request.
        if len(questions) > settings.max_questions:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Too many questions. Maximum allowed is "
                    f"{settings.max_questions} questions per request."
                ),
            )

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

    # Embedding/vector-store construction performs blocking work,
    # so move it off the event loop.
    vector_store = await asyncio.to_thread(
        build_vector_store,
        chunks,
    )

    # Apply backpressure instead of launching every question at once.
    semaphore = asyncio.Semaphore(
        settings.max_concurrent_questions
    )

    tasks = [
        process_question(
            question=question,
            vector_store=vector_store,
            semaphore=semaphore,
        )
        for question in questions
    ]

    # Questions run concurrently while gather preserves input order.
    results = await asyncio.gather(*tasks)

    return QAResponse(
        results=results,
    )