from fastapi import FastAPI

from app.api.routes import router


app = FastAPI(
    title="Document QA Agent",
    description=(
        "Question-answering API for PDF and JSON documents "
        "using retrieval-augmented generation."
    ),
    version="1.0.0",
)

app.include_router(router)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the health status of the API."""

    return {"status": "ok"}