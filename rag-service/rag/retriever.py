import structlog
from db.models import SessionLocal, Chunk
from ingest.embedder import embed_texts
from sqlalchemy import text

log = structlog.get_logger()


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    """
    Embed the question and retrieve top-k most similar chunks
    from pgvector using cosine distance.
    """
    query_embedding = embed_texts([question])[0]

    session = SessionLocal()
    try:
        results = (
            session.query(Chunk)
            .order_by(Chunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
            .all()
        )
        chunks = [
            {"chunk_id": c.id, "doc_id": c.doc_id, "content": c.content}
            for c in results
        ]
        log.info("retriever.done", top_k=top_k, retrieved=len(chunks))
        return chunks
    finally:
        session.close()
