from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# -----------------------------
# Load documents
# -----------------------------

def load_documents():
    documents = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if text:
            documents.append({
                "id": file_path.stem,
                "text": text
            })

    return documents


# -----------------------------
# Main ingestion pipeline
# -----------------------------

def main():
    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 documents, but found {len(documents)}."
        )

    # Load local embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings
    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    # Create ChromaDB client
    print("Creating ChromaDB collection...")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Zepto customer support policy documents"
        }
    )

    # Store documents and embeddings
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source": doc["id"],
                "file": f"{doc['id']}.txt"
            }
            for doc in documents
        ]
    )

    print("\nIngestion complete!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents stored: {collection.count()}")
    print(f"Database location: {CHROMA_DIR}")


if __name__ == "__main__":
    main()