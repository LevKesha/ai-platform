"""Chat Lambda. Policy and the helper slot are read from their owners.

No rate limiter lives here. The edge WAF owns request limits.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from chat_helper.corpus import (
    classify,
    compose,
    grounded,
    load_lines,
    sentences,
    system_prompt,
)
from chat_helper.ledger import LedgerError, MemoryLedger, ledger_from_env
from chat_helper.policy import load_policy
from chat_helper.pricing import (
    MAX_INPUT_TOKENS,
    actual_micro,
    bound_question,
    fit_input,
    max_output_tokens,
    prepare_converse,
    worst_case_micro,
)

LOG_EVENTS = {"request", "answer", "abstain", "closed", "denied", "retry"}
BANNED = ("asleep", "waking", "money", "error", "dollar")


class ModelTimeout(Exception):
    pass


class ModelDenied(Exception):
    pass


@dataclass
class Deps:
    policy: dict
    corpus: list
    ledger: MemoryLedger
    haiku: Any
    theseus: Any | None
    bedrock_enabled: bool
    now: Callable[[], datetime]
    helper_model_id: str
    live_model_id: str
    rag_enabled: bool = False
    rag_invoke: Callable[[str, list], dict] | None = None


def log_event(name: str, path: str | None = None) -> None:
    if name not in LOG_EVENTS:
        name = "denied"
    body: dict[str, str] = {"event": name}
    if path in {"rag", "cv", "retry"}:
        body["path"] = path
    sys.stdout.write(json.dumps(body) + "\n")


def payload_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def helper_model_id() -> str:
    from platform_common.registry import load_model_mix

    return load_model_mix()["helper_bedrock_id"]


def live_model_id() -> str:
    from platform_common.registry import load_model_mix

    return load_model_mix()["default_bedrock_id"]


def _link(policy: dict) -> dict[str, str]:
    return {"text": policy["link_text"], "href": policy["link_href"]}


def closed_body(policy: dict) -> dict:
    return {
        "kind": "closed",
        "text": policy["closed"],
        "label": "",
        "link": _link(policy),
        "switch": False,
        "input_enabled": False,
    }


def abstain_body(policy: dict, kind: str) -> dict:
    key = "off_topic" if kind == "off_topic" else "not_in_cv"
    return {
        "kind": kind,
        "text": policy[key],
        "label": "",
        "link": _link(policy),
        "switch": False,
        "input_enabled": True,
    }


def answer_body(policy: dict, text: str, label: str, switch: bool) -> dict:
    return {
        "kind": "answer",
        "text": text,
        "label": label,
        "link": None,
        "switch": switch,
        "input_enabled": True,
    }


def _http(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body),
    }


def _headers(event: dict) -> dict[str, str]:
    raw = event.get("headers") or {}
    return {str(key).lower(): str(value) for key, value in raw.items()}


def _raw_body(event: dict) -> str:
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        import base64

        raw = base64.b64decode(raw).decode("utf-8")
    return raw


def publishable(text: str, lines: list[dict[str, str]]) -> bool:
    cleaned = text.strip()
    if not cleaned or not grounded(cleaned, lines):
        return False
    folded = cleaned.lower()
    if any(word in folded for word in BANNED):
        return False
    count = len(sentences(cleaned))
    return 1 <= count <= 3


def _converse(backend, model_id: str, system: str, question: str, history: list) -> Any:
    question, history, system = fit_input(question, history, system)
    return backend.converse(
        model_id=model_id,
        system=system,
        question=question,
        history=history,
        max_tokens=max_output_tokens(),
    )


def retry_body(policy: dict) -> dict:
    return {"kind": "retry", "text": policy["retry"], "input_enabled": True}


def _finish(status: int, body: dict, event_name: str, path: str | None = None) -> dict:
    log_event(event_name, path)
    return _http(status, body)


def _rag_invoker(source: dict) -> Callable[[str, list], dict]:
    name = str(source.get("RAG_FUNCTION_NAME") or "")

    def invoke(question: str, history: list) -> dict:
        if not name:
            return {"kind": "retry"}
        import boto3

        try:
            raw = boto3.client("lambda").invoke(
                FunctionName=name,
                InvocationType="RequestResponse",
                Payload=json.dumps({"question": question, "history": history}).encode("utf-8"),
            )
        except Exception:
            return {"kind": "retry"}
        if raw.get("FunctionError"):
            return {"kind": "retry"}
        payload = raw.get("Payload")
        try:
            parsed = json.loads(payload.read() if hasattr(payload, "read") else payload or b"{}")
        except Exception:
            return {"kind": "retry"}
        if not isinstance(parsed, dict):
            return {"kind": "retry"}
        return parsed

    return invoke


def _rag_branch(deps: Deps, question: str, history: list, chat_id: str, idem: str, now: datetime, policy: dict) -> dict | None:
    """RAG draft, retry, or None when the cluster is not live (CV fallthrough).

    The hold uses a separate idem so an asleep fallthrough can still reserve
    the visitor idem. publishable() and the ledger stay here.
    """
    month = f"SPEND#{now.strftime('%Y-%m')}"
    rag_idem = idem + "#rag"
    worst = worst_case_micro(question, history, " " * (MAX_INPUT_TOKENS * 4), "helper")
    try:
        hold = deps.ledger.reserve(month, rag_idem, worst)
    except LedgerError:
        return _finish(200, closed_body(policy), "closed")
    if not hold.ok:
        deps.ledger.mark_closed(chat_id, now)
        return _finish(200, closed_body(policy), "closed")
    if hold.settled and hold.response:
        if hold.response.get("kind") == "cv":
            return None
        event = "answer" if hold.response.get("kind") == "answer" else "retry"
        path = "rag" if event == "answer" else "retry"
        return _finish(200, hold.response, event, path)
    invoke = deps.rag_invoke
    try:
        outcome = invoke(question, history) if invoke is not None else {"kind": "retry"}
    except Exception:
        outcome = {"kind": "retry"}
    if not isinstance(outcome, dict):
        outcome = {"kind": "retry"}
    if outcome.get("kind") == "cv":
        deps.ledger.settle(month, rag_idem, 0, {"kind": "cv"})
        return None
    if outcome.get("kind") != "draft":
        body = retry_body(policy)
        deps.ledger.settle(month, rag_idem, 0, body)
        return _finish(200, body, "retry", "retry")
    raw_chunks = outcome.get("chunks")
    lines = []
    if isinstance(raw_chunks, list):
        for index, chunk in enumerate(raw_chunks):
            text = str(chunk).strip()
            if text:
                lines.append({"id": str(index), "source": "rag", "section": "platform", "text": text})
    text = str(outcome.get("text") or "").strip()
    if not publishable(text, lines):
        body = retry_body(policy)
        deps.ledger.settle(month, rag_idem, 0, body)
        return _finish(200, body, "retry", "retry")
    usage = outcome.get("usage") if isinstance(outcome.get("usage"), dict) else {}
    actual = actual_micro(
        {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
        },
        "helper",
    )
    body = answer_body(policy, text, policy["label_rag"], False)
    deps.ledger.settle(month, rag_idem, actual, body)
    deps.ledger.bump(chat_id, now)
    return _finish(200, body, "answer", "rag")


def handle(event: dict, deps: Deps) -> dict:
    raw = _raw_body(event)
    headers = _headers(event)
    # Read in memory only. Never pass this value to the ledger, logs, or a response.
    _viewer = headers.get("x-viewer-ip")
    del _viewer
    if headers.get("x-amz-content-sha256") != payload_hash(raw):
        return _finish(400, {"kind": "rejected"}, "denied")
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return _finish(400, {"kind": "rejected"}, "denied")
    if not isinstance(payload, dict):
        return _finish(400, {"kind": "rejected"}, "denied")

    question = bound_question(str(payload.get("question") or ""))
    history_in = payload.get("history") if isinstance(payload.get("history"), list) else []
    chat_id = str(payload.get("chatId") or "anon")[:64]
    idem = str(payload.get("idempotencyKey") or "")[:80]
    if not question or not idem:
        return _finish(400, {"kind": "rejected"}, "denied")
    question, history = prepare_converse(question, history_in)

    policy = deps.policy
    now = deps.now()
    limit = int(policy["turn_limit"])
    if deps.ledger.chat_closed(chat_id, now) or deps.ledger.read_turns(chat_id, now) >= limit:
        deps.ledger.mark_closed(chat_id, now)
        return _finish(200, closed_body(policy), "closed")
    if not deps.bedrock_enabled:
        deps.ledger.mark_closed(chat_id, now)
        return _finish(200, closed_body(policy), "closed")

    if deps.rag_enabled:
        rag_response = _rag_branch(deps, question, history, chat_id, idem, now, policy)
        if rag_response is not None:
            return rag_response

    kind, lines = classify(question, deps.corpus)
    if kind in {"off_topic", "not_in_cv"}:
        deps.ledger.write_unanswered(question, now)
        deps.ledger.bump(chat_id, now)
        return _finish(200, abstain_body(policy, kind), "abstain")

    completed = deps.ledger.read_turns(chat_id, now)
    live_answers = int(policy["live_answers"])
    backend = deps.haiku
    live = False
    if deps.theseus is not None and completed < live_answers:
        backend = deps.theseus
        live = True
    system = system_prompt(lines)
    slot = "default" if live else "helper"
    worst = worst_case_micro(question, history, system, slot)
    month = f"SPEND#{now.strftime('%Y-%m')}"
    try:
        hold = deps.ledger.reserve(month, idem, worst)
    except LedgerError:
        return _finish(200, closed_body(policy), "closed")
    if not hold.ok:
        deps.ledger.mark_closed(chat_id, now)
        return _finish(200, closed_body(policy), "closed")
    if hold.settled and hold.response:
        return _finish(200, hold.response, "answer")
    if hold.replay:
        return _finish(200, retry_body(policy), "retry", "retry")

    model_id = deps.live_model_id if live else deps.helper_model_id
    try:
        result = _converse(backend, model_id, system, question, history)
        used_live = live
    except ModelDenied:
        if not live:
            deps.ledger.mark_closed(chat_id, now)
            return _finish(200, closed_body(policy), "closed")
        try:
            result = _converse(deps.haiku, deps.helper_model_id, system, question, history)
        except ModelDenied:
            deps.ledger.mark_closed(chat_id, now)
            return _finish(200, closed_body(policy), "closed")
        except ModelTimeout:
            return _finish(200, retry_body(policy), "retry", "retry")
        used_live = False
    except ModelTimeout:
        if not live:
            return _finish(200, retry_body(policy), "retry", "retry")
        try:
            result = _converse(deps.haiku, deps.helper_model_id, system, question, history)
        except ModelDenied:
            deps.ledger.mark_closed(chat_id, now)
            return _finish(200, closed_body(policy), "closed")
        except ModelTimeout:
            return _finish(200, retry_body(policy), "retry", "retry")
        used_live = False

    text = (result.text or "").strip()
    if not publishable(text, deps.corpus):
        text = compose(lines)
    if not publishable(text, deps.corpus):
        deps.ledger.bump(chat_id, now)
        body = abstain_body(policy, "not_in_cv")
        deps.ledger.settle(month, idem, 0, body)
        return _finish(200, body, "abstain")

    label = policy["label_rag"] if used_live else policy["label_cv"]
    show_switch = used_live and completed == live_answers - 1
    body = answer_body(policy, text, label, show_switch)
    usage = getattr(result, "usage", None) or {"input_tokens": 0, "output_tokens": 0}
    actual = actual_micro(usage, "default" if used_live else "helper")
    deps.ledger.settle(month, idem, actual, body)
    deps.ledger.bump(chat_id, now)
    return _finish(200, body, "answer", "cv")


class BedrockModel:
    """Converse adapter. Tests inject a stub and never construct this."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def converse(self, **kwargs: Any) -> Any:
        import boto3

        client = boto3.client("bedrock-runtime")
        messages = [
            {"role": item["role"], "content": [{"text": item["text"]}]}
            for item in kwargs["history"]
        ]
        messages.append({"role": "user", "content": [{"text": kwargs["question"]}]})
        reply = client.converse(
            modelId=kwargs["model_id"],
            system=[{"text": kwargs["system"]}],
            messages=messages,
            inferenceConfig={"maxTokens": kwargs["max_tokens"]},
        )
        text = reply["output"]["message"]["content"][0]["text"]
        usage = reply.get("usage") or {}

        @dataclass
        class Reply:
            text: str
            usage: dict

        return Reply(text, {"input_tokens": usage.get("inputTokens", 0), "output_tokens": usage.get("outputTokens", 0)})


def deps_from_env(env: dict | None = None) -> Deps:
    from datetime import timezone

    source = env if env is not None else __import__("os").environ
    policy = load_policy(source.get("CHAT_POLICY_PATH"))
    bucket = source.get("CORPUS_BUCKET")
    if bucket:
        import boto3

        from chat_helper.corpus import cached_lines, s3_loader

        client = boto3.client("s3")
        corpus = cached_lines(s3_loader(client, bucket, source.get("CORPUS_KEY", "chat-corpus.json")))
    else:
        corpus = load_lines()
    theseus = None
    if source.get("THESEUS_HANDOFF") == "1":
        theseus = BedrockModel(live_model_id())
    rag_enabled = source.get("CHAT_RAG_ENABLED") == "1"
    return Deps(
        policy=policy,
        corpus=corpus,
        ledger=ledger_from_env(dict(source), policy),
        haiku=BedrockModel(helper_model_id()),
        theseus=theseus,
        bedrock_enabled=source.get("BEDROCK_ENABLED") == "1",
        now=lambda: datetime.now(timezone.utc),
        helper_model_id=helper_model_id(),
        live_model_id=live_model_id(),
        rag_enabled=rag_enabled,
        rag_invoke=_rag_invoker(dict(source)) if rag_enabled else None,
    )


def lambda_handler(event: dict, context: Any) -> dict:
    return handle(event, deps_from_env())
