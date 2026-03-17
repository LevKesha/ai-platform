# rag-service

FastAPI RAG service — ingests documents, stores embeddings in PostgreSQL + pgvector (RDS), and answers queries using Claude (Bedrock) with source citations.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Chunk, embed, and store a document |
| POST | `/query` | Semantic search + Claude grounded answer |
| GET | `/health` | Liveness check |

## Module Structure

```
rag-service/
  app/main.py          ← FastAPI app, endpoints, startup
  db/models.py         ← SQLAlchemy models + pgvector column
  db/store.py          ← Write chunks to RDS
  ingest/chunker.py    ← Fixed-size overlapping text chunker
  ingest/embedder.py   ← Bedrock Titan Embed wrapper
  rag/retriever.py     ← pgvector cosine similarity search
  rag/generator.py     ← Prompt construction + Claude call
  tests/               ← Unit tests
  Dockerfile
  requirements.txt
```

## Local Dev

```bash
cp .env.example .env
# fill in DB_PASSWORD and real AWS creds

pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Ingest Example

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"doc_id": "doc1", "content": "AWS ECS is a container orchestration service...", "metadata": {"source": "aws-docs"}}'
```

## Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is ECS?", "top_k": 3}'
```
