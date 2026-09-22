# Document QA Agent

A production-oriented Question-Answering application that uses Retrieval-Augmented Generation (RAG) to answer questions grounded in uploaded PDF or JSON documents.

The application accepts:

- A JSON file containing a list of questions
- A source document in PDF or JSON format

It retrieves relevant evidence from the source document and uses an OpenAI language model to generate structured, grounded answers with confidence levels and source references.

---

## Features

- PDF and JSON document ingestion
- Multiple supported question JSON formats
- Page-aware PDF extraction
- Recursive text chunking with overlap
- OpenAI embeddings
- FAISS vector similarity search
- Retrieval-Augmented Generation (RAG)
- GPT-4o-mini grounded answer generation
- Structured question/answer output
- Source/page references
- Confidence classification
- `Data-Not-Found` handling
- Partial-answer support for multi-part questions
- FastAPI REST API
- Streamlit user interface
- Docker support
- Bounded concurrent question processing
- Configurable question-count and upload-size limits
- OpenAI request timeout protection
- Structured JSON request logging
- Request IDs and response latency tracking
- Automated test suite with mocked LLM calls

---

## Architecture

```text
┌─────────────────────┐
│    Streamlit UI     │
│                     │
│ Questions JSON      │
│ Document PDF/JSON   │
└──────────┬──────────┘
           │
           │ multipart/form-data
           ▼
┌─────────────────────┐
│      FastAPI        │
│    POST /api/v1/qa  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Question Parser     │
│ Document Loader     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Text Chunking       │
│ 1000 chars          │
│ 150 overlap         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ OpenAI Embeddings   │
│ text-embedding-     │
│ 3-small             │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FAISS Vector Store  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Similarity Retriever│
│ Top K = 8           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ GPT-4o-mini         │
│ Grounded QA         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Structured Response │
│ Answer              │
│ Confidence          │
│ Sources             │
└─────────────────────┘
```

---

## RAG Processing Flow

For every request:

1. The questions JSON file is parsed and validated.
2. The source PDF or JSON document is loaded.
3. Document text is normalized into LangChain `Document` objects.
4. Text is split into overlapping chunks.
5. OpenAI embeddings are generated for the chunks.
6. The chunks are indexed in an in-memory FAISS vector store.
7. For each question, the most relevant chunks are retrieved.
8. Retrieved evidence is passed to GPT-4o-mini.
9. The model generates a structured grounded response.
10. Evidence indices are validated and converted into source/page references.
11. The API returns all question/answer pairs as structured JSON.

The full document is **not** sent to the language model. Only retrieved evidence is included in the QA prompt.

---

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| API | FastAPI |
| UI | Streamlit |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| PDF Parsing | PyMuPDF |
| Validation | Pydantic |
| Testing | pytest |
| Containerization | Docker |

---

## Project Structure

```text
document-qa-agent/
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── exceptions.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── question_parser.py
│       ├── document_loader.py
│       ├── chunking.py
│       ├── vector_store.py
│       ├── retriever.py
│       └── qa_service.py
│
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_question_parser.py
│   ├── test_document_loader.py
│   └── test_qa_service.py
│
├── sample_data/
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.12+
- OpenAI API key
- Docker (optional)

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/sushanthrangu/zania-document-qa-agent.git
cd zania-document-qa-agent
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example configuration:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`.

Example:

```env
OPENAI_API_KEY=your_openai_api_key_here

OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=8

MAX_QUESTIONS=50
MAX_QUESTIONS_FILE_SIZE_MB=1
MAX_DOCUMENT_FILE_SIZE_MB=20
MAX_CONCURRENT_QUESTIONS=5

OPENAI_TIMEOUT_SECONDS=60
```

Never commit `.env` or API keys to source control.

---

## Running the Application

The application consists of a FastAPI backend and a Streamlit frontend.

### Start FastAPI

From the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Start Streamlit

Open another terminal, activate the virtual environment, and run:

```bash
streamlit run ui/streamlit_app.py
```

Streamlit normally starts at:

```text
http://localhost:8501
```

Upload:

1. A questions JSON file
2. A PDF or JSON source document

Then select **Generate Answers**.

---

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Generate Answers

```http
POST /api/v1/qa
```

The endpoint accepts `multipart/form-data` containing:

- `questions_file` — JSON file containing questions
- `document_file` — PDF or JSON source document

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/qa" \
  -F "questions_file=@questions.json" \
  -F "document_file=@document.pdf"
```

Example response:

```json
{
  "results": [
    {
      "id": 1,
      "question": "Which cloud providers do you rely on?",
      "answer": "The document identifies Google Cloud Platform (GCP) as the cloud hosting provider.",
      "confidence": "high",
      "sources": [
        {
          "source": "document.pdf",
          "page": 16
        }
      ]
    }
  ]
}
```

---

## Supported Question Formats

The question parser supports multiple common JSON structures.

### Wrapped questions

```json
{
  "questions": [
    {
      "id": 1,
      "question": "Which cloud providers do you rely on?"
    }
  ]
}
```

### Top-level list

```json
[
  {
    "id": 1,
    "question": "Which cloud providers do you rely on?"
  }
]
```

### List of strings

```json
[
  "Which cloud providers do you rely on?",
  "What monitoring capabilities are documented?"
]
```

Additional fields in question objects can be present without being passed to the QA model.

---

## Supported Document Formats

### PDF

PDF documents are processed page-by-page using PyMuPDF.

Page metadata is preserved so generated answers can reference the pages containing supporting evidence.

### JSON

JSON documents are recursively flattened into textual representations while preserving key paths as metadata.

This allows nested JSON structures to participate in semantic retrieval.

---

## Grounded Answering

The model is explicitly instructed to answer using only retrieved document context.

The QA layer distinguishes between:

- Directly supported information
- Partially or indirectly supported information
- Information that cannot be established from the retrieved evidence

The system does not intentionally fill missing document details using the model's general knowledge.

### Partial Answers

For multi-part questions, the system preserves useful evidence even when the complete answer is unavailable.

For example, if a document identifies a cloud provider but does not identify the geographic region, the system can return the documented provider/infrastructure information while clearly stating that the requested region was not identified.

This avoids discarding useful evidence simply because one portion of a question cannot be answered.

### Data-Not-Found

When no meaningful part of a question can be supported by retrieved evidence, the system returns:

```text
Data-Not-Found
```

A `Data-Not-Found` result has:

- `confidence = low`
- No source references

---

## Confidence Levels

Confidence represents the degree to which the retrieved document evidence supports the generated answer.

| Confidence | Meaning |
|---|---|
| `high` | Evidence directly and materially answers the question |
| `medium` | Evidence is meaningful but partial or indirect |
| `low` | Evidence is insufficient and the answer is `Data-Not-Found` |

Confidence is generated as part of the model's structured output and should be interpreted as an evidence-support indicator rather than a mathematical probability.

---

## Source Validation

The model returns evidence indices referring to retrieved context chunks.

Before exposing sources to the client, the application:

1. Validates that each evidence index exists.
2. Ignores invalid indices.
3. Maps valid evidence back to document metadata.
4. Deduplicates repeated source/page combinations.
5. Returns no sources for `Data-Not-Found`.

This prevents the model from directly inventing arbitrary page references.

---

## Retrieval Configuration

Current defaults:

```text
Chunk size:     1000 characters
Chunk overlap:   150 characters
Top K:             8 chunks
```

`TOP_K=8` was selected after retrieval testing against the supplied sample document. A smaller retrieval window could miss relevant evidence for questions whose supporting information was distributed across the document.

---

## Testing

Run the complete automated test suite:

```bash
pytest -v
```

Current test result:

```text
43 passed, 7 warnings
```

The tests cover:

- Question JSON parsing
- Valid and invalid question structures
- PDF loading
- JSON document loading
- Unsupported document types
- Empty/scanned PDF behavior
- Grounded QA response handling
- Partial answers
- `Data-Not-Found`
- Evidence validation
- Source deduplication
- FastAPI health endpoint
- QA endpoint validation
- Successful API orchestration
- Error handling
- Maximum question-count validation
- Questions/document upload-size validation
- Concurrent multi-question processing with result-order preservation
- API robustness controls

LLM behavior is mocked in automated tests, so running the test suite does not require consuming OpenAI API credits.

---

## Docker

Build the image:

```bash
docker build -t document-qa-agent .
```

Run the FastAPI container:

```bash
docker run --rm \
  --name document-qa-agent \
  -p 8000:8000 \
  --env-file .env \
  document-qa-agent
```

Verify the running container:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

The `.env` file is excluded from the Docker build context and is supplied only at container runtime.

---

## Design Decisions

### RAG instead of sending the entire document

Only relevant chunks are sent to the LLM. This reduces token usage, improves scalability, and keeps the answer focused on relevant evidence.

### FAISS

FAISS provides efficient local vector similarity search without requiring an external vector database service, which keeps the assignment easy to run and review.

### Structured LLM Output

The model returns a validated schema containing:

- Answer
- Confidence
- Evidence indices

This is more reliable for API consumers than parsing free-form model text.

### Page-Aware PDF Processing

PDF pages are preserved as metadata throughout ingestion, chunking, retrieval, and answer generation, enabling source traceability.

### Deterministic Generation

The chat model uses:

```text
temperature = 0
```

to reduce unnecessary response variation.

### Separation of Concerns

Parsing, document loading, chunking, indexing, retrieval, answer generation, API routing, and UI rendering are implemented as separate components.

This makes the application easier to test and maintain.

---

## Complexity

Let:

- `N` = number of characters in the source document
- `C` = number of generated chunks
- `Q` = number of questions
- `K` = number of retrieved chunks (`8`)

Document parsing and chunking are approximately:

```text
Time:  O(N)
Space: O(N)
```

Embedding/index construction scales approximately with the number of chunks:

```text
O(C)
```

at the application level, while actual embedding computation is performed by the external embedding service.

Each question performs vector retrieval plus one LLM generation request.

At the application level:

```text
Q × (retrieval + generation)
```

The dominant real-world latency is generally network/model inference rather than local Python processing.

---

## Security Considerations

- OpenAI API keys are loaded through environment variables.
- `.env` is excluded from Git.
- `.env` is excluded from the Docker build context.
- API keys are not included in source code.
- The QA prompt explicitly instructs the model to remain grounded in supplied document context.
- Model-provided evidence references are validated before being returned.

For a public production deployment, authentication, rate limiting, request-size limits, and additional content/security controls should also be added.

---

## Current Limitations

### Scanned PDFs

The current implementation extracts text directly with PyMuPDF.

Image-only or scanned PDFs without an embedded text layer require OCR and are currently rejected when no usable text can be extracted.

### In-Memory Vector Store

The FAISS index is rebuilt for each request and is not persisted.

For repeated queries against the same large document, a production system could persist and reuse document indexes.

### Semantic Retrieval

The current implementation uses dense similarity retrieval.

Some complex or multi-part questions may benefit from:

- Hybrid lexical + semantic retrieval
- Query decomposition
- Reranking
- MMR retrieval
- Metadata filtering

### Request Processing

Questions within a request are processed concurrently using asynchronous execution.

The application:

- Uses `asyncio.gather()` to process multiple questions concurrently.
- Uses an `asyncio.Semaphore` to bound concurrency.
- Processes at most 5 questions concurrently by default.
- Offloads synchronous FAISS retrieval work using `asyncio.to_thread()`.
- Uses asynchronous LLM calls for answer generation.
- Preserves the original question order in the final response.
- Applies configurable OpenAI request timeouts to embedding and chat-model operations.

The concurrency limit can be configured with:

```text
MAX_CONCURRENT_QUESTIONS=5
```

The OpenAI timeout can be configured with:

```text
OPENAI_TIMEOUT_SECONDS=60
```

These controls improve throughput while preventing unbounded concurrent model requests.


### Production API Controls

The API includes configurable safeguards for request size and workload:

- Maximum questions per request: **50**
- Maximum questions JSON file size: **1 MB**
- Maximum document file size: **20 MB**
- Maximum concurrent questions: **5**
- OpenAI request timeout: **60 seconds**

Requests that exceed the configured question-count or upload-size limits are rejected with HTTP `413` responses and clear error messages.

These limits can be configured through environment variables without changing application code.

Authentication and rate limiting are not included in this assignment implementation and would be appropriate additions for a public internet-facing deployment.
---

## Future Improvements

Potential improvements include:

- OCR support for scanned PDFs
- Persistent vector indexes
- Document fingerprinting and index caching
- Hybrid lexical + semantic retrieval
- Query decomposition for complex questions
- Cross-encoder reranking
- Retrieval evaluation metrics
- Authentication and authorization
- API rate limiting
- Persistent answer history
- Distributed task processing for larger workloads
- Persistent metrics dashboards and distributed tracing

---
## Observability

The FastAPI application includes structured request-level observability using Python's standard logging library.

For every HTTP request, the middleware records:

- UTC timestamp
- Log level
- Logger name
- Request ID
- HTTP method
- Request path
- Response status code
- Request duration in milliseconds

Logs are emitted as structured JSON, making them suitable for ingestion by centralized logging platforms.

Example:

```json
{
  "timestamp": "2026-09-22T14:34:25.203734+00:00",
  "level": "INFO",
  "logger": "app.main",
  "message": "request_completed",
  "request_id": "f31b019e-8ce2-4061-a881-68b6163ba1ea",
  "method": "GET",
  "path": "/health",
  "status_code": 200,
  "duration_ms": 2.02
}
```

Each response also includes an `X-Request-ID` header for request correlation. If the client supplies an `X-Request-ID`, the application preserves it; otherwise, a UUID is generated.

Request bodies, uploaded document contents, questions, answers, and API keys are not intentionally included in request logs.

---

## Summary

This project demonstrates an end-to-end document Question-Answering system using Retrieval-Augmented Generation.

The implementation focuses on:

- Grounded answers
- Evidence traceability
- Clear handling of missing information
- Modular architecture
- Structured API responses
- Automated testing
- Reproducible local and Docker execution

The result is a practical foundation that can be extended into a larger document intelligence or knowledge retrieval platform.
