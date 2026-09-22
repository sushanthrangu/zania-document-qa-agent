from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings


def retrieve_documents(
    vector_store: FAISS,
    query: str,
) -> list[Document]:
    """Retrieve the most relevant document chunks for a query."""

    if not query or not query.strip():
        raise ValueError(
            "Retrieval query cannot be empty."
        )

    settings = get_settings()

    return vector_store.similarity_search(
        query=query.strip(),
        k=settings.top_k,
    )