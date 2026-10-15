"""Simple RAG (Retrieval-Augmented Generation) over the onboarding PDFs.

1. Indexing:   split PDFs into chunks and turn each chunk into an embedding.
2. Retrieval:  embed the question and find the most similar chunks.
The agent then answers using only those chunks (generation).
"""
import os
from pathlib import Path

import numpy as np
from pypdf import PdfReader

from google import genai
from google.genai import types

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHUNK_SIZE = 500  # max characters per chunk
TOP_K = 3         # how many chunks to return per question

_index = None  # built on first use: {"chunks": [...], "vectors": np.ndarray}


def split_into_chunks(text: str, source: str) -> list[dict]:
    """Pack lines into chunks of up to CHUNK_SIZE characters.
    The last line of each chunk is repeated in the next one (overlap),
    so information at a chunk border is not lost."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks, current = [], []
    for line in lines:
        if current and len(" ".join(current + [line])) > CHUNK_SIZE:
            chunks.append(" ".join(current))
            current = [current[-1]]  # overlap
        current.append(line)
    if current:
        chunks.append(" ".join(current))
    return [{"text": c, "source": source} for c in chunks]


def load_chunks() -> list[dict]:
    chunks = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        chunks.extend(split_into_chunks(text, pdf_path.name))
    return chunks


def embed(texts: list[str], task_type: str) -> np.ndarray:
    """Turn texts into embedding vectors with the Gemini embedding model."""


    client = genai.Client()  # reads GOOGLE_API_KEY from the environment
    model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
    vectors = []
    for i in range(0, len(texts), 50):  # send in batches
        result = client.models.embed_content(
            model=model,
            contents=texts[i:i + 50],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in result.embeddings)
    return np.array(vectors)


def build_index(embed_fn=embed) -> dict:
    chunks = load_chunks()
    vectors = embed_fn([c["text"] for c in chunks], "RETRIEVAL_DOCUMENT")
    return {"chunks": chunks, "vectors": vectors}


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return (matrix @ vector) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector))


def search_documents(query: str) -> dict:
    """Searches the HR onboarding documents for passages relevant to the query.

    Args:
        query: The employee's question or the topic to search for.

    Returns:
        A dict with the most relevant document passages and their source file.
    """
    global _index
    if _index is None:
        _index = build_index()

    query_vector = embed([query], "RETRIEVAL_QUERY")[0]
    scores = cosine_similarity(_index["vectors"], query_vector)
    best = np.argsort(scores)[::-1][:TOP_K]
    return {
        "results": [
            {
                "source": _index["chunks"][i]["source"],
                "text": _index["chunks"][i]["text"],
                "score": round(float(scores[i]), 3),
            }
            for i in best
        ]
    }
