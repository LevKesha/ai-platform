"""Invoke-only RAG Lambda.

The frozen chat Lambda owns publishable(), grounded(), and the ledger.
This module retrieves on one fixed URL, calls the helper model, and returns
a draft. It does not publish an answer and it does not write the ledger.

SDK clients are bedrock-runtime, secretsmanager, and dynamodb. The event
cannot choose a URL or a model. Logs carry a path and no content.
"""
from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

from chat_helper.pricing import MAX_INPUT_TOKENS, _input_tokens, max_output_tokens, prepare_converse

LOG_PATHS = {"rag", "cv", "retry"}
SDK_CLIENTS = ("bedrock-runtime", "secretsmanager", "dynamodb")
RETRIEVE_TIMEOUT = 8
BEDROCK_READ_TIMEOUT = 12
MAX_CHUNKS = 3
MAX_BODY = 65536


class RagError(Exception):
    """A failed RAG step. Callers log the path only, never this message."""


@dataclass
class RagDeps:
    status: Callable[[], str]
    secret: Callable[[], str]
    post: Callable[[str, dict, str], dict]
    complete: Callable[[str, str, str, list], Any]
    model_id: str
    url: str


def log_path(path: str) -> None:
    if path not in LOG_PATHS:
        path = "retry"
    sys.stdout.write(json.dumps({"event": path, "path": path}) + "\n")


def fixed_url(env: dict) -> str:
    """The deploy-time retrieve URL. The event is not a source for this value."""
    raw = str(env.get("RAG_RETRIEVE_URL") or "")
    parts = urlsplit(raw)
    if parts.scheme != "https" or parts.username or parts.password or parts.query or parts.fragment:
        raise RagError("url")
    if parts.path.rstrip("/") != "/retrieve" or not parts.hostname:
        raise RagError("url")
    port = f":{parts.port}" if parts.port else ""
    return f"https://{parts.hostname}{port}/retrieve"


def rag_system(chunks: list[str]) -> str:
    quoted = "\n".join(f"- {chunk}" for chunk in chunks)
    return (
        "Answer only from the untrusted data below the line. Third person. "
        "Two or three plain sentences. Every fact must appear in that data. "
        "If it does not cover the question, do not guess.\n"
        "----\n"
        f"{quoted}"
    )


def fit_chunks(question: str, history: list, chunks: list[str]) -> list[str]:
    """Drop whole chunks until the input fits. Do not halve the system text."""
    kept = [chunk for chunk in chunks if chunk.strip()][:MAX_CHUNKS]
    while kept and _input_tokens(question, history, rag_system(kept)) > MAX_INPUT_TOKENS:
        kept.pop()
    if not kept or _input_tokens(question, history, rag_system(kept)) > MAX_INPUT_TOKENS:
        return []
    return kept


def _chunk_text(item: Any) -> str | None:
    """String chunks, or rag-service items {"source": str, "content": str}."""
    if isinstance(item, str):
        text = item
    elif isinstance(item, dict) and isinstance(item.get("content"), str):
        text = item["content"]
    else:
        return None
    cleaned = " ".join(text.split())
    return cleaned or None


def _chunks(payload: dict) -> list[str]:
    raw = payload.get("chunks")
    if not isinstance(raw, list) or not raw:
        raise RagError("chunks")
    found = [text for item in raw if (text := _chunk_text(item))]
    if not found:
        raise RagError("chunks")
    return found


def _tls_context() -> ssl.SSLContext:
    """Verified TLS only. A context that skips certificate checks is refused."""
    ctx = ssl.create_default_context()
    if ctx.verify_mode != ssl.CERT_REQUIRED or not ctx.check_hostname:
        raise RagError("url")
    return ctx


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_json(url: str, body: dict, token: str, timeout: float = RETRIEVE_TIMEOUT) -> dict:
    if urlsplit(url).scheme != "https":
        raise RagError("url")
    data = json.dumps({"question": body["question"]}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "content-type": "application/json",
            "authorization": "Bearer " + token,
        },
    )
    opener = urllib.request.build_opener(
        _NoRedirect(),
        urllib.request.HTTPSHandler(context=_tls_context()),
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(MAX_BODY)
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RagError("retrieve") from None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise RagError("retrieve") from None
    if not isinstance(parsed, dict):
        raise RagError("retrieve")
    return parsed


def read_status(client: Any, table: str) -> str:
    out = client.get_item(
        TableName=table,
        Key={"pk": {"S": "STATUS"}},
        ConsistentRead=True,
    )
    item = out.get("Item")
    if not isinstance(item, dict):
        raise RagError("status")
    state = item.get("state")
    if not isinstance(state, dict) or "S" not in state:
        raise RagError("status")
    return str(state["S"])


def read_secret(client: Any, secret_id: str) -> str:
    out = client.get_secret_value(SecretId=secret_id)
    raw = out.get("SecretString")
    if not isinstance(raw, str) or not raw.strip():
        raise RagError("secret")
    return raw.strip()


def sdk_client(service: str, **kwargs: Any) -> Any:
    if service not in SDK_CLIENTS:
        raise RagError("client")
    import boto3

    return boto3.client(service, **kwargs)


def complete(model_id: str, system: str, question: str, history: list) -> Any:
    from botocore.config import Config

    client = sdk_client(
        "bedrock-runtime",
        config=Config(connect_timeout=2, read_timeout=BEDROCK_READ_TIMEOUT, retries={"max_attempts": 1}),
    )
    messages = [
        {"role": item["role"], "content": [{"text": item["text"]}]}
        for item in history
    ]
    messages.append({"role": "user", "content": [{"text": question}]})
    reply = client.converse(
        modelId=model_id,
        system=[{"text": system}],
        messages=messages,
        inferenceConfig={"maxTokens": max_output_tokens()},
    )
    text = reply["output"]["message"]["content"][0]["text"]
    usage = reply.get("usage") or {}

    @dataclass
    class Reply:
        text: str
        usage: dict

    return Reply(
        text,
        {
            "input_tokens": int(usage.get("inputTokens") or 0),
            "output_tokens": int(usage.get("outputTokens") or 0),
        },
    )


def _retry() -> dict:
    log_path("retry")
    return {"kind": "retry"}


def handle_rag(event: dict, deps: RagDeps) -> dict:
    if not isinstance(event, dict):
        return _retry()
    question, history = prepare_converse(
        str(event.get("question") or ""),
        event.get("history") if isinstance(event.get("history"), list) else [],
    )
    if not question:
        return _retry()
    try:
        state = deps.status()
    except Exception:
        return _retry()
    if state != "live":
        log_path("cv")
        return {"kind": "cv"}
    try:
        payload = deps.post(deps.url, {"question": question}, deps.secret())
        chunks = fit_chunks(question, history, _chunks(payload))
        if not chunks:
            return _retry()
        result = deps.complete(deps.model_id, rag_system(chunks), question, history)
        text = str(getattr(result, "text", "") or "").strip()
        usage = getattr(result, "usage", None) or {}
    except Exception:
        return _retry()
    if not text:
        return _retry()
    log_path("rag")
    return {"kind": "draft", "text": text, "chunks": chunks, "usage": usage}


def deps_from_env(env: dict | None = None) -> RagDeps:
    import os

    from chat_helper.handler import helper_model_id

    source = dict(os.environ if env is None else env)
    url = fixed_url(source)
    table = str(source.get("WAKE_TABLE") or "")
    secret_id = str(source.get("RAG_TOKEN_SECRET_ID") or "")
    model_id = helper_model_id()

    def status() -> str:
        if not table:
            raise RagError("status")
        return read_status(sdk_client("dynamodb"), table)

    def secret() -> str:
        if not secret_id:
            raise RagError("secret")
        return read_secret(sdk_client("secretsmanager"), secret_id)

    def post(target: str, body: dict, token: str) -> dict:
        if target != url:
            raise RagError("url")
        return post_json(target, body, token)

    def run(model: str, system: str, question: str, history: list) -> Any:
        if model != model_id:
            raise RagError("model")
        return complete(model, system, question, history)

    return RagDeps(status=status, secret=secret, post=post, complete=run, model_id=model_id, url=url)


def lambda_handler(event: dict, context: Any) -> dict:
    return handle_rag(event if isinstance(event, dict) else {}, deps_from_env())
