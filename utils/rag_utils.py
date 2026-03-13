"""RAG utility functions for document ingestion and retrieval."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import faiss
import numpy as np
from pypdf import PdfReader

from models.embeddings import create_embeddings


def load_documents_from_folder(folder_path: str) -> list[dict[str, str]]:
    """Load supported text content from a folder recursively."""

    base_path = Path(folder_path)
    if not base_path.exists():
        return []

    supported_suffixes = {".txt", ".md", ".pdf"}
    documents: list[dict[str, str]] = []

    for file_path in base_path.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_suffixes:
            continue

        try:
            if file_path.suffix.lower() == ".pdf":
                pdf_reader = PdfReader(str(file_path))
                text = "\n".join((page.extract_text() or "") for page in pdf_reader.pages)
            else:
                text = file_path.read_text(encoding="utf-8", errors="ignore")

            text = text.strip()
            if text:
                documents.append({"source": str(file_path), "text": text})
        except Exception:
            continue

    return documents


def split_documents_into_chunks(
    documents: list[dict[str, str]], chunk_size: int = 700, overlap: int = 120
) -> list[dict[str, Any]]:
    """Split raw documents into overlapping chunks for vector search."""

    chunks: list[dict[str, Any]] = []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    for doc in documents:
        text = doc["text"]
        source = doc["source"]

        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "chunk_id": chunk_index,
                    }
                )

            if end == len(text):
                break

            start += chunk_size - overlap
            chunk_index += 1

    return chunks


def create_vector_store(chunks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Create a FAISS index from text chunks and their embeddings."""

    if not chunks:
        return None

    chunk_texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(chunk_texts)

    if embeddings.size == 0:
        return None

    if embeddings.ndim != 2:
        raise ValueError("Expected 2D embeddings for vector store creation.")

    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    return {
        "index": index,
        "chunks": chunks,
    }


def retrieve_relevant_context(
    query: str,
    vector_store: dict[str, Any] | None,
    top_k: int = 4,
    min_similarity: float = 0.25,
) -> list[dict[str, Any]]:
    """Retrieve top relevant chunks from FAISS for a query."""

    if not vector_store or "index" not in vector_store or "chunks" not in vector_store:
        return []

    index = vector_store["index"]
    chunks = vector_store["chunks"]
    if index.ntotal == 0:
        return []

    query_vector = create_embeddings(query)
    if query_vector.ndim != 1:
        raise ValueError("Expected a single vector for query embedding.")

    query_vector = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
    distances, indices = index.search(query_vector, min(top_k, index.ntotal))

    results: list[dict[str, Any]] = []
    for distance, idx in zip(distances[0], indices[0], strict=False):
        if idx < 0 or idx >= len(chunks):
            continue

        similarity = 1.0 / (1.0 + float(distance))
        if similarity >= min_similarity:
            chunk = chunks[idx]
            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "similarity": similarity,
                }
            )

    return results
