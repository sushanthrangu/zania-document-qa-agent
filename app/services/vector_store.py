from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings


def get_embedding_model() -> OpenAIEmbeddings:
    """Create the OpenAI embedding model used for vectorization."""

    settings = get_settings()

    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


def build_vector_store(
    documents: list[Document],
) -> FAISS:
    """Build a FAISS vector store from document chunks."""

    if not documents:
        raise ValueError(
            "Cannot build vector store from an empty document list."
        )

    embeddings = get_embedding_model()

    return FAISS.from_documents(
        documents=documents,
        embedding=embeddings,
    )