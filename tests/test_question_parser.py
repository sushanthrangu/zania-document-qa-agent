import json

import pytest

from app.services.question_parser import (
    QuestionParseError,
    parse_questions,
)


def test_parse_structured_questions():
    content = json.dumps(
        {
            "questions": [
                {
                    "id": "q1",
                    "question": "Which cloud provider do you use?",
                },
                {
                    "id": "q2",
                    "question": "Where is the data stored?",
                },
            ]
        }
    ).encode("utf-8")

    questions = parse_questions(content)

    assert len(questions) == 2
    assert questions[0].id == "q1"
    assert questions[0].question == "Which cloud provider do you use?"
    assert questions[1].id == "q2"


def test_parse_question_string_list():
    content = json.dumps(
        [
            "Which cloud provider do you use?",
            "Where is the data stored?",
        ]
    ).encode("utf-8")

    questions = parse_questions(content)

    assert len(questions) == 2
    assert questions[0].id == "q1"
    assert questions[1].id == "q2"
    assert questions[0].question == "Which cloud provider do you use?"


def test_parse_top_level_question_objects():
    content = json.dumps(
        [
            {
                "id": "abc123",
                "question": "Which cloud provider do you use?",
                "answer": "Expected answer from evaluation file",
                "comments": "Test comment",
                "confidence": "high",
            }
        ]
    ).encode("utf-8")

    questions = parse_questions(content)

    assert len(questions) == 1
    assert questions[0].id == "abc123"
    assert questions[0].question == "Which cloud provider do you use?"


def test_invalid_json_raises_error():
    content = b"{invalid json}"

    with pytest.raises(
        QuestionParseError,
        match="valid UTF-8 JSON",
    ):
        parse_questions(content)


def test_missing_questions_field_raises_error():
    content = json.dumps(
        {
            "items": [
                {
                    "id": "q1",
                    "question": "Test question",
                }
            ]
        }
    ).encode("utf-8")

    with pytest.raises(
        QuestionParseError,
        match="'questions' field",
    ):
        parse_questions(content)


def test_empty_question_list_raises_error():
    content = json.dumps(
        {
            "questions": [],
        }
    ).encode("utf-8")

    with pytest.raises(
        QuestionParseError,
        match="cannot be empty",
    ):
        parse_questions(content)


def test_blank_question_raises_error():
    content = json.dumps(
        {
            "questions": [
                {
                    "id": "q1",
                    "question": "",
                }
            ]
        }
    ).encode("utf-8")

    with pytest.raises(
        QuestionParseError,
        match="Invalid question",
    ):
        parse_questions(content)


def test_invalid_question_type_raises_error():
    content = json.dumps(
        {
            "questions": [
                123,
            ]
        }
    ).encode("utf-8")

    with pytest.raises(
        QuestionParseError,
        match="must be a string or an object",
    ):
        parse_questions(content)