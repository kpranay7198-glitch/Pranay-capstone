from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def main():
    # Load embedding model
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to existing ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    # Test policy question
    query = "How much does Zepto charge for delivery?"

    # Embed query
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Retrieve top 3
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    print("\nQuery:")
    print(query)

    print("\nTop retrieved documents:")

    for i in range(len(results["ids"][0])):
        print("\n-----------------------------")
        print(f"Rank: {i + 1}")
        print(f"ID: {results['ids'][0][i]}")
        print(f"Distance: {results['distances'][0][i]:.4f}")
        print(f"Source: {results['metadatas'][0][i]['source']}")
        print(f"Text: {results['documents'][0][i][:300]}")


if __name__ == "__main__":
    main()