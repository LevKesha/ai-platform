import uuid
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any

from db.models import init_db
from ingest.chunker import chunk_text
from ingest.embedder import embed_texts
from db.store import store_chunks
from rag.retriever import retrieve
from rag.generator import generate_answer

log = structlog.get_logger()

app = FastAPI(title="rag-service", version="0.1.0")


@app.on_event("startup")
async def startup():
    init_db()
    log.info("rag-service started")


# ---------- schemas ----------

class IngestRequest(BaseModel):
    doc_id: str
    content: str
    metadata: dict[str, Any] = {}


class IngestResponse(BaseModel):
    status: str
    chunks_created: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    request_id: str


# ---------- endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    log.info("ingest.start", doc_id=req.doc_id)
    chunks = chunk_text(req.content)
    embeddings = embed_texts(chunks)
    store_chunks(req.doc_id, chunks, embeddings, req.metadata)
    log.info("ingest.done", doc_id=req.doc_id, chunks=len(chunks))
    return IngestResponse(status="ok", chunks_created=len(chunks))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    request_id = str(uuid.uuid4())
    log.info("query.start", request_id=request_id, question=req.question)
    chunks = retrieve(req.question, req.top_k)
    answer, sources = generate_answer(req.question, chunks, request_id)
    log.info("query.done", request_id=request_id, sources=len(sources))
    return QueryResponse(answer=answer, sources=sources, request_id=request_id)
