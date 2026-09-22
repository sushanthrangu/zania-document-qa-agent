import logging
import time
import uuid

from fastapi import FastAPI, Request

from app.api.routes import router
from app.core.logging import configure_logging


configure_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Document QA Agent",
    description=(
        "Question-answering API for PDF and JSON documents "
        "using retrieval-augmented generation."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    """Log request metadata and latency without sensitive content."""

    request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )

        return response

    except Exception:
        duration_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        logger.exception(
            "request_failed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": 500,
                "duration_ms": duration_ms,
            },
        )

        raise


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the health status of the API."""

    return {"status": "ok"}