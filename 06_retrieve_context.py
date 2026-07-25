from importlib import import_module

import chromadb
import numpy as np
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

vectors = import_module("04_vector_representation")
store = import_module("05_create_chroma_store")
preprocessing = import_module("02_preprocessing")

model = vectors.model
ALPHA = vectors.ALPHA

_client = chromadb.PersistentClient(
    path=str(store.DB_PATH),
    settings=Settings(anonymized_telemetry=False),
)
_collection = _client.get_or_create_collection(store.COLLECTION_NAME)
if _collection.count() == 0:
    _collection = store.create_vector_store()


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def hybrid_search(question, k=4):
    """Retrieve top chunks from ChromaDB using hybrid BM25 + stored embedding similarity."""
    clean_query = preprocessing.preprocess_text(question)

    stored = _collection.get(include=["documents", "metadatas"])
    ids, documents, metadatas = stored["ids"], stored["documents"], stored["metadatas"]

    tokenized_chunks = [preprocessing.preprocess_text(doc).split() for doc in documents]
    bm25_scores = BM25Okapi(tokenized_chunks).get_scores(clean_query.split())

    query_embedding = model.encode([clean_query], convert_to_numpy=True, normalize_embeddings=True)
    chroma_results = _collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=len(ids),
        include=["distances"],
    )
    distance_by_id = dict(zip(chroma_results["ids"][0], chroma_results["distances"][0]))
    embedding_scores = np.array([1 - distance_by_id.get(chunk_id, 1.0) for chunk_id in ids])

    hybrid_scores = ((1 - ALPHA) * min_max_normalize(bm25_scores)) + (
        ALPHA * min_max_normalize(embedding_scores)
    )

    ranking = np.argsort(hybrid_scores)[::-1][:k]
    return [
        {
            "chunk_id": ids[i],
            "document_id": metadatas[i]["document_id"],
            "title": metadatas[i]["title"],
            "is_current": metadatas[i]["is_current"] == "True",
            "chunk_text": documents[i],
            "score": hybrid_scores[i],
        }
        for i in ranking
    ]


def build_context(question, k=4, max_sources=3):
    """Retrieve top chunks for a question and package them into a citable context string."""
    if not question or not question.strip():
        return "", []

    rows = hybrid_search(question, k=k)
    rows = sorted(rows, key=lambda row: (row["is_current"], row["score"]), reverse=True)

    selected = []
    seen_documents = set()

    for row in rows:
        if row["score"] <= 0:
            continue
        if row["document_id"] in seen_documents:
            continue
        selected.append(row)
        seen_documents.add(row["document_id"])
        if len(selected) == max_sources:
            break

    context = ""
    for source_number, row in enumerate(selected, start=1):
        status = "CURRENT" if row["is_current"] else "OUTDATED"
        context += f"[Source {source_number}] {row['title']} ({status})\n{row['chunk_text']}\n\n"

    return context.strip(), selected

