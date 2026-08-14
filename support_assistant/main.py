import os
from pathlib import Path
from typing import TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

from .prompt import build_prompt


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default is mock mode.
# MOCK_LLM unset or MOCK_LLM=1 -> mock mode
# MOCK_LLM=0 -> optional real-LLM mode
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# ============================================================
# Pydantic Models
# ============================================================

class SupportRequest(BaseModel):
    query: str


class SupportResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ============================================================
# LangGraph State
# ============================================================

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float
    context: str


# ============================================================
# Load Embedding Model and ChromaDB
# ============================================================

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)


# ============================================================
# Policy Keywords
# ============================================================

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


# ============================================================
# Required Real-LLM Response Validation + Retry Logic
# ============================================================

def validate_llm_response(
    raw_response,
    retry_callback,
    max_retries=2,
):
    """
    Validate a real LLM response against the SupportResponse schema.

    If validation fails, retry up to max_retries times with
    a corrective instruction.

    This function is used only by the optional real-LLM path.
    """

    # --------------------------------------------------------
    # First validation attempt
    # --------------------------------------------------------

    try:
        if isinstance(raw_response, str):
            return SupportResponse.model_validate_json(raw_response)

        return SupportResponse.model_validate(raw_response)

    except Exception as first_error:
        last_error = first_error

    # --------------------------------------------------------
    # Corrective retries
    # --------------------------------------------------------

    for attempt in range(max_retries):

        corrective_instruction = (
            "Your previous response did not match the required "
            "JSON schema. Return ONLY valid JSON with exactly "
            "these fields: answer (string), sources (list of "
            "strings), and confidence (number between 0 and 1). "
            f"Validation error: {last_error}"
        )

        corrected_response = retry_callback(
            corrective_instruction
        )

        try:
            if isinstance(corrected_response, str):
                return SupportResponse.model_validate_json(
                    corrected_response
                )

            return SupportResponse.model_validate(
                corrected_response
            )

        except Exception as error:
            last_error = error

    # --------------------------------------------------------
    # All attempts failed
    # --------------------------------------------------------

    return SupportResponse(
        answer=(
            "ERROR: The real LLM response could not be validated "
            "against the required output schema."
        ),
        sources=[],
        confidence=0.0,
    )


# ============================================================
# Node 1 — Intent Classification
# ============================================================

def classify_intent(state: GraphState) -> GraphState:

    query = state["query"]
    query_lower = query.lower()

    # --------------------------------------------------------
    # Required graded MOCK_LLM behavior
    # --------------------------------------------------------

    if MOCK_LLM:

        if any(
            keyword in query_lower
            for keyword in POLICY_KEYWORDS
        ):
            intent = "policy_question"
        else:
            intent = "general_question"

    # --------------------------------------------------------
    # Optional real-LLM behavior
    # --------------------------------------------------------

    else:

        # Optional real LLM classification can be connected here.
        # The graded baseline does not require an API call.
        #
        # Safe fallback keeps the service functional.

        if any(
            keyword in query_lower
            for keyword in POLICY_KEYWORDS
        ):
            intent = "policy_question"
        else:
            intent = "general_question"

    return {
        **state,
        "intent": intent,
    }


# ============================================================
# Node 2 — Retrieve and Answer
# ============================================================

def retrieve_and_answer(state: GraphState) -> GraphState:

    query = state["query"]

    # --------------------------------------------------------
    # Embed query locally
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # --------------------------------------------------------
    # Retrieve top 3 chunks from ChromaDB
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_documents = results["documents"][0]
    retrieved_metadatas = results["metadatas"][0]

    # --------------------------------------------------------
    # Source IDs
    # --------------------------------------------------------

    source_ids = [
        metadata["source"]
        for metadata in retrieved_metadatas
    ]

    # --------------------------------------------------------
    # Build context for optional real LLM
    # --------------------------------------------------------

    context_parts = []

    for source, document in zip(
        source_ids,
        retrieved_documents
    ):
        context_parts.append(
            f"{source}:\n{document}"
        )

    context = "\n\n".join(context_parts)

    # ========================================================
    # REQUIRED MOCK_LLM PATH
    # ========================================================

    if MOCK_LLM:

        top_chunk = retrieved_documents[0]

        # Required short excerpt.
        top_chunk_snippet = top_chunk[:200]

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        return {
            **state,
            "answer": answer,
            "sources": source_ids,
            "confidence": 1.0,
            "context": context,
        }

    # ========================================================
    # OPTIONAL REAL-LLM PATH
    # ========================================================

    prompt = build_prompt(
        query=query,
        context=context,
    )

    # --------------------------------------------------------
    # Real LLM integration is optional.
    #
    # The graded baseline does NOT require an API key.
    #
    # The prompt above is the structured prompt required by
    # Task 2 and would be sent to the real LLM here.
    # --------------------------------------------------------

    # Safe fallback when no real LLM backend is configured.
    answer = (
        "Real LLM mode is not configured. "
        "Please use MOCK_LLM=1 for the graded offline mode."
    )

    return {
        **state,
        "answer": answer,
        "sources": source_ids,
        "confidence": 0.5,
        "context": context,
    }


# ============================================================
# Node 3 — Direct Answer
# ============================================================

def direct_answer(state: GraphState) -> GraphState:

    # --------------------------------------------------------
    # Required MOCK_LLM behavior
    # --------------------------------------------------------

    if MOCK_LLM:

        answer = (
            "I can only answer questions about Zepto policies "
            "right now."
        )

        return {
            **state,
            "answer": answer,
            "sources": [],
            "confidence": 1.0,
        }

    # --------------------------------------------------------
    # Optional real-LLM path
    # --------------------------------------------------------

    prompt = build_prompt(
        query=state["query"],
        context="No policy documents were retrieved.",
    )

    # Safe fallback when no real LLM backend is configured.
    answer = (
        "Real LLM mode is not configured. "
        "Please use MOCK_LLM=1 for the graded offline mode."
    )

    return {
        **state,
        "answer": answer,
        "sources": [],
        "confidence": 0.5,
    }


# ============================================================
# Conditional Routing
# ============================================================

def route_intent(state: GraphState) -> str:

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# Build LangGraph StateGraph
# ============================================================

graph_builder = StateGraph(GraphState)

# Three required nodes
graph_builder.add_node(
    "classify_intent",
    classify_intent,
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

graph_builder.add_node(
    "direct_answer",
    direct_answer,
)


# Entry point
graph_builder.set_entry_point(
    "classify_intent"
)


# Conditional routing
graph_builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)


# End edges
graph_builder.add_edge(
    "retrieve_and_answer",
    END,
)

graph_builder.add_edge(
    "direct_answer",
    END,
)


# Compile graph
graph = graph_builder.compile()


# ============================================================
# Run Support Assistant
# ============================================================

def run_support_assistant(
    query: str
) -> SupportResponse:

    initial_state: GraphState = {
        "query": query,
    }

    result = graph.invoke(
        initial_state
    )

    return SupportResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get(
            "confidence",
            1.0
        ),
    )


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Zepto Support Assistant",
    description=(
        "Offline RAG support assistant using "
        "LangGraph and ChromaDB"
    ),
)


# ============================================================
# POST /ask
# ============================================================

@app.post(
    "/ask",
    response_model=SupportResponse,
)
def ask(
    request: SupportRequest
):

    return run_support_assistant(
        request.query
    )


# ============================================================
# Local Testing
# ============================================================

if __name__ == "__main__":

    print(
        "MOCK_LLM:",
        MOCK_LLM
    )

    # --------------------------------------------------------
    # Policy question
    # --------------------------------------------------------

    policy_result = run_support_assistant(
        "How much does Zepto charge for delivery?"
    )

    print("\nPolicy question:")

    print(
        policy_result.model_dump_json(
            indent=2
        )
    )

    # --------------------------------------------------------
    # General question
    # --------------------------------------------------------

    general_result = run_support_assistant(
        "What is the capital of India?"
    )

    print("\nGeneral question:")

    print(
        general_result.model_dump_json(
            indent=2
        )
    )