import json

from pydantic import ValidationError

from app.models.schemas import Question


class QuestionParseError(ValueError):
    """Raised when a questions file cannot be parsed or validated."""


def parse_questions(content: bytes) -> list[Question]:
    """Parse and validate questions from JSON file content."""

    try:
        data = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuestionParseError(
            "Questions file must contain valid UTF-8 JSON."
        ) from exc

    # Format A:
    # {
    #   "questions": [
    #       {"id": "q1", "question": "..."}
    #   ]
    # }
    if isinstance(data, dict):
        if "questions" not in data:
            raise QuestionParseError(
                "JSON object must contain a 'questions' field."
            )

        data = data["questions"]

    if not isinstance(data, list):
        raise QuestionParseError(
            "Questions must be provided as a JSON list."
        )

    if not data:
        raise QuestionParseError(
            "Questions list cannot be empty."
        )

    questions: list[Question] = []

    for index, item in enumerate(data, start=1):
        try:
            # Format B:
            # [
            #   "Question one?",
            #   "Question two?"
            # ]
            if isinstance(item, str):
                question = Question(
                    id=f"q{index}",
                    question=item.strip(),
                )

            # Structured question:
            # {"id": "q1", "question": "..."}
            elif isinstance(item, dict):
                question = Question.model_validate(item)

            else:
                raise QuestionParseError(
                    f"Question at position {index} must be "
                    "a string or an object."
                )

        except ValidationError as exc:
            raise QuestionParseError(
                f"Invalid question at position {index}."
            ) from exc

        questions.append(question)

    return questions