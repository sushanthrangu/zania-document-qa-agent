from typing import Literal

from pydantic import BaseModel, Field


class Question(BaseModel):
    """A single question supplied by the user."""

    id: str
    question: str = Field(min_length=1)


class SourceReference(BaseModel):
    """Document location used as evidence for an answer."""

    source: str
    page: int | None = None


class AnswerResult(BaseModel):
    """Structured answer returned for a single question."""

    id: str
    question: str
    answer: str
    confidence: Literal["high", "medium", "low"]
    sources: list[SourceReference] = Field(default_factory=list)


class QAResponse(BaseModel):
    """Top-level API response containing all generated answers."""

    results: list[AnswerResult]