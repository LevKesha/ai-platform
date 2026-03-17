from db.models import SessionLocal, Document, Chunk


def store_chunks(doc_id: str, chunks: list[str], embeddings: list[list[float]], metadata: dict):
    session = SessionLocal()
    try:
        # Upsert document
        doc = session.get(Document, doc_id)
        if not doc:
            doc = Document(doc_id=doc_id, metadata_=metadata)
            session.add(doc)

        # Delete old chunks for this doc (re-ingest case)
        session.query(Chunk).filter(Chunk.doc_id == doc_id).delete()

        for i, (text, emb) in enumerate(zip(chunks, embeddings)):
            session.add(Chunk(
                doc_id=doc_id,
                chunk_index=i,
                content=text,
                embedding=emb,
            ))
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
