import os
import json
import time
import boto3
import structlog

log = structlog.get_logger()

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])
    return _client


def generate_answer(question: str, chunks: list[dict], request_id: str) -> tuple[str, list[dict]]:
    """
    Build a grounded prompt from retrieved chunks and call Claude via Bedrock.
    Returns (answer_text, sources_list).
    """
    model_id = os.environ.get("BEDROCK_LLM_MODEL_ID", "anthropic.claude-haiku-20240307-v1:0")

    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(f"[{i+1}] (doc: {c['doc_id']}, chunk_id: {c['chunk_id']})\n{c['content']}")
    context = "\n\n".join(context_parts)

    prompt = (
        f"You are a helpful assistant. Answer the question using ONLY the context below. "
        f"Cite sources as [1], [2], etc. based on the numbered context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    })

    t0 = time.time()
    response = _get_client().invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    latency_ms = int((time.time() - t0) * 1000)

    result = json.loads(response["body"].read())
    answer = result["content"][0]["text"]

    sources = [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]} for c in chunks]

    log.info(
        "generator.done",
        request_id=request_id,
        model=model_id,
        latency_ms=latency_ms,
        prompt_chars=len(prompt),
        chunks_used=len(chunks),
    )

    return answer, sources
