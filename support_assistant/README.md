# Module 3 — Zepto Support Assistant

## 1. Project Overview

This project implements an offline customer-support assistant for Zepto using a Retrieval-Augmented Generation (RAG) architecture.

The system uses:

* 8 Zepto policy documents in TXT format
* `all-MiniLM-L6-v2` for local text embeddings
* ChromaDB as the vector database
* LangGraph for workflow orchestration
* Pydantic for structured response validation
* FastAPI for the `/ask` API endpoint
* Docker for containerization
* `MOCK_LLM=1` as the default graded baseline

The system can distinguish between policy-related and general questions. Policy questions are answered using retrieved policy documents, while general questions receive a direct mock response.

---

## 2. Architecture

```text
                         User Query
                             |
                             v
                    +------------------+
                    | classify_intent  |
                    +------------------+
                       /            \
                      /              \
                     v                v
          policy_question       general_question
                  |                    |
                  v                    v
       +---------------------+   +---------------+
       | retrieve_and_answer |   | direct_answer |
       +---------------------+   +---------------+
                  |
                  v
          Query Embedding
                  |
                  v
             ChromaDB
                  |
                  v
             Top-3 Sources
                  |
                  v
          Structured Response
                  |
                  v
              FastAPI
               /ask
```

### RAG Pipeline

```text
TXT Documents
      |
      v
Document Ingestion
      |
      v
all-MiniLM-L6-v2
      |
      v
Embeddings
      |
      v
ChromaDB
      |
      v
Query Embedding
      |
      v
Top-3 Retrieval
      |
      v
Answer Generation
```

---

## 3. Project Structure

```text
Pranay-capstone/
│
├── README.md
├── Dockerfile
├── .dockerignore
│
└── support_assistant/
    │
    ├── __init__.py
    ├── main.py
    ├── ingest.py
    ├── retrieve.py
    ├── prompt.py
    ├── requirements.txt
    │
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    │
    └── data/
        └── chroma/
```

---

## 4. Policy Corpus

The knowledge base contains eight policy documents:

| Document     | Policy                    |
| ------------ | ------------------------- |
| `doc_01.txt` | Delivery Policy           |
| `doc_02.txt` | Returns & Refunds         |
| `doc_03.txt` | Membership Tiers          |
| `doc_04.txt` | Order Tracking            |
| `doc_05.txt` | Order Cancellation Policy |
| `doc_06.txt` | Damaged or Missing Items  |
| `doc_07.txt` | Gift Cards                |
| `doc_08.txt` | Customer Support Hours    |

Each document contains the policy text used by the retrieval system.

---

## 5. Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r support_assistant\requirements.txt
```

---

## 6. Document Ingestion

Run the ingestion pipeline:

```powershell
python support_assistant\ingest.py
```

The ingestion pipeline:

1. Loads the eight TXT documents.
2. Loads `all-MiniLM-L6-v2`.
3. Generates embeddings locally.
4. Creates the ChromaDB collection.
5. Stores the documents and embeddings.

Expected result:

```text
Loaded 8 documents.
Generating embeddings...
Creating ChromaDB collection...

Ingestion complete!
Collection: zepto_policies
Documents stored: 8
```

---

## 7. Testing Retrieval

The retrieval test is implemented in:

```text
support_assistant/retrieve.py
```

Run:

```powershell
python support_assistant\retrieve.py
```

Example query:

```text
How much does Zepto charge for delivery?
```

The system successfully retrieved:

```text
Rank: 1
ID: doc_01
Source: doc_01
```

`doc_01` contains the delivery policy and therefore provides the correct source for the query.

---

## 8. Structured Prompt

The structured prompt is implemented in:

```text
support_assistant/prompt.py
```

The prompt contains:

* Role
* Context
* Task
* Format
* Length
* Negative constraint
* Few-shot example

The negative constraint prevents the assistant from inventing policy information that is not present in the retrieved context.

The expected response structure is:

```json
{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}
```

---

## 9. LangGraph Workflow

The LangGraph workflow is implemented in:

```text
support_assistant/main.py
```

It contains three required nodes:

### Node 1 — `classify_intent`

Determines whether the query is:

```text
policy_question
```

or:

```text
general_question
```

### Node 2 — `retrieve_and_answer`

For policy questions:

1. Embeds the query.
2. Searches ChromaDB.
3. Retrieves the top three documents.
4. Builds the retrieved context.
5. Generates the response.

### Node 3 — `direct_answer`

For general questions, the system returns the direct mock response:

```text
I can only answer questions about Zepto policies right now.
```

### Conditional Routing

```text
classify_intent
      |
      +---- policy_question ----> retrieve_and_answer
      |
      +---- general_question --> direct_answer
```

---

## 10. Mock LLM Mode

The default configuration uses:

```text
MOCK_LLM=1
```

This mode does not require an external LLM API or API key.

When a policy question is received, the system returns an answer based on the retrieved context.

For example:

```text
How much does delivery cost?
```

produces a response beginning with:

```text
Based on the retrieved context:
```

For a general question:

```text
What is the capital of India?
```

the system returns:

```text
I can only answer questions about Zepto policies right now.
```

---

## 11. Pydantic Validation and Retry Logic

The response schema is defined using Pydantic:

```python
class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

The confidence value is constrained between `0.0` and `1.0`.

The code also contains validation and corrective retry logic for the optional real-LLM path.

If an LLM response fails validation:

```text
LLM Response
     |
     v
Pydantic Validation
     |
   failure
     |
     v
Corrective Retry
     |
     v
Validation
     |
   failure
     |
     v
Second Retry
     |
     v
Validation
     |
   failure
     |
     v
Clear Error Response
```

The graded baseline continues to use `MOCK_LLM=1`.

---

## 12. FastAPI

The application exposes:

```text
POST /ask
```

The request format is:

```json
{
  "query": "How much does delivery cost?"
}
```

Start the API locally:

```powershell
python -m uvicorn support_assistant.main:app --reload
```

The service runs on:

```text
http://127.0.0.1:8000
```

---

## 13. Local API Test — Policy Question

PowerShell:

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"How much does delivery cost?"}'
$response | ConvertTo-Json
```

Successful response:

```json
{
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
    "sources": [
        "doc_01",
        "doc_05",
        "doc_02"
    ],
    "confidence": 1.0
}
```

The first source is:

```text
doc_01
```

which is the Delivery Policy document.

---

## 14. Local API Test — General Question

PowerShell:

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the capital of India?"}'
$response | ConvertTo-Json
```

Successful response:

```json
{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}
```

This demonstrates the `direct_answer` route.

---

## 15. Docker

The application is containerized using Docker.

The Docker image uses:

```text
python:3.11-slim
```

The container:

1. Installs the Python dependencies.
2. Copies the support assistant application.
3. Runs the ingestion pipeline.
4. Creates the ChromaDB collection inside the container.
5. Starts Uvicorn.

### Build the image

From the project root:

```powershell
docker build -t zepto-support .
```

### Run the container

```powershell
docker run --rm -p 7860:7860 zepto-support
```

The API is then available at:

```text
http://127.0.0.1:7860
```

The container was successfully verified with:

```text
Loaded 8 documents.
Documents stored: 8
Application startup complete.
Uvicorn running on http://0.0.0.0:7860
```

---

## 16. Docker API Test — Policy Question

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body '{"query":"How much does delivery cost?"}'
$response | ConvertTo-Json
```

Successful response:

```json
{
    "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
    "sources": [
        "doc_01",
        "doc_05",
        "doc_02"
    ],
    "confidence": 1.0
}
```

---

## 17. Docker API Test — General Question

```powershell
$response = Invoke-RestMethod -Uri "http://127.0.0.1:7860/ask" -Method Post -ContentType "application/json" -Body '{"query":"What is the capital of India?"}'
$response | ConvertTo-Json
```

Successful response:

```json
{
    "answer": "I can only answer questions about Zepto policies right now.",
    "sources": [],
    "confidence": 1.0
}
```

---

## 18. Docker Optimization

A `.dockerignore` file is included to prevent unnecessary files from being sent to Docker.

Excluded content includes:

```text
.venv/
__pycache__/
*.pyc
.git/
.ipynb_checkpoints/
.vscode/
*.log
support_assistant/data/chroma/
```

The Docker build context was reduced from approximately 1.54 GB to approximately 1 KB.

The ChromaDB database is intentionally excluded from the image because the container creates its own database by running:

```text
support_assistant/ingest.py
```

during container startup.

---

## 19. End-to-End Workflow

```text
                    +----------------+
                    | 8 Policy TXT   |
                    |   Documents    |
                    +-------+--------+
                            |
                            v
                    +---------------+
                    |   ingest.py   |
                    +-------+-------+
                            |
                            v
                 +---------------------+
                 | all-MiniLM-L6-v2   |
                 | Local Embeddings   |
                 +---------+-----------+
                           |
                           v
                    +-------------+
                    |  ChromaDB   |
                    +------+------+
                           |
                     User Query
                           |
                           v
                 +-------------------+
                 | classify_intent   |
                 +---------+---------+
                           |
              +------------+------------+
              |                         |
              v                         v
       Policy Question            General Question
              |                         |
              v                         v
   retrieve_and_answer            direct_answer
              |
              v
       Top-3 Retrieval
              |
              v
        Structured Response
              |
              v
           FastAPI
              |
              v
          POST /ask
              |
              v
           Docker
```

---

## 20. Key Technologies

| Component            | Technology       |
| -------------------- | ---------------- |
| Programming Language | Python 3.11      |
| Embeddings           | all-MiniLM-L6-v2 |
| Vector Database      | ChromaDB         |
| Workflow             | LangGraph        |
| Validation           | Pydantic         |
| API                  | FastAPI          |
| Server               | Uvicorn          |
| Containerization     | Docker           |
| Baseline LLM Mode    | MOCK_LLM         |
| Document Format      | TXT              |

---

## 21. Completion Summary

The Module 3 implementation provides:

* 8-document Zepto policy corpus
* Local MiniLM embeddings
* ChromaDB vector retrieval
* Top-3 retrieval
* Structured prompt template
* LangGraph `StateGraph`
* `classify_intent` node
* `retrieve_and_answer` node
* `direct_answer` node
* Conditional routing
* Mock LLM baseline
* Pydantic structured responses
* Validation and retry logic
* FastAPI `/ask` endpoint
* Dockerized deployment
* Successful local Docker build
* Successful Docker container startup
* Successful Docker API testing
* Policy and general-question demonstrations

The complete system therefore implements the required offline RAG customer-support assistant workflow.
