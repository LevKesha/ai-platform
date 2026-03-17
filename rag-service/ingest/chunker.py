from typing import List

CHUNK_SIZE = 500      # characters
CHUNK_OVERLAP = 50   # characters


def chunk_text(text: str) -> List[str]:
    """
    Split text into overlapping fixed-size character chunks.
    Simple but effective for RAG — swap for sentence/token-aware
    chunking later if retrieval quality needs improvement.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
