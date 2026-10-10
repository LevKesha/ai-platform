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
    MAX_OUTPUT_TOKENS,
    actual_micro,
    bound_history,
    bound_question,
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


def log_event(name: str) -> None:
    if name not in LOG_EVENTS:
        name = "denied"
    sys.stdout.write(json.dumps({"event": name}) + "\n")


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
    return backend.converse(
        model_id=model_id,
        system=system,
        question=question,
        history=history,
        max_tokens=MAX_OUTPUT_TOKENS,
    )


def _finish(status: int, body: dict, event_name: str) -> dict:
    log_event(event_name)
    return _http(status, body)


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
    history = bound_history(payload.get("history") if isinstance(payload.get("history"), list) else [])
    chat_id = str(payload.get("chatId") or "anon")[:64]
    idem = str(payload.get("idempotencyKey") or "")[:80]
    if not question or not idem:
        return _finish(400, {"kind": "rejected"}, "denied")

    policy = deps.policy
    now = deps.now()
    limit = int(policy["turn_limit"])
    if deps.ledger.chat_closed(chat_id, now) or deps.ledger.read_turns(chat_id, now) >= limit:
        deps.ledger.mark_closed(chat_id, now)
        return _finish(200, closed_body(policy), "closed")
    if not deps.bedrock_enabled:
        deps.ledger.mark_closed(chat_id, now)
        return _finish(200, closed_body(policy), "closed")

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
    model_kind = backend.kind
    worst = worst_case_micro(question, history, system, model_kind)
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
            return _finish(200, {"kind": "retry", "input_enabled": True}, "retry")
        used_live = False
    except ModelTimeout:
        if not live:
            return _finish(200, {"kind": "retry", "input_enabled": True}, "retry")
        try:
            result = _converse(deps.haiku, deps.helper_model_id, system, question, history)
        except ModelDenied:
            deps.ledger.mark_closed(chat_id, now)
            return _finish(200, closed_body(policy), "closed")
        except ModelTimeout:
            return _finish(200, {"kind": "retry", "input_enabled": True}, "retry")
        used_live = False

    text = (result.text or "").strip()
    if not publishable(text, deps.corpus):
        text = compose(lines)
    if not publishable(text, deps.corpus):
        deps.ledger.bump(chat_id, now)
        body = abstain_body(policy, "not_in_cv")
        deps.ledger.settle(month, idem, 0, body)
        return _finish(200, body, "abstain")

    label = policy["label_live"] if used_live else policy["label_cv"]
    show_switch = used_live and completed == live_answers - 1
    body = answer_body(policy, text, label, show_switch)
    usage = getattr(result, "usage", None) or {"input_tokens": 0, "output_tokens": 0}
    actual = min(actual_micro(usage, "sonnet" if used_live else "haiku"), worst)
    deps.ledger.settle(month, idem, actual, body)
    deps.ledger.bump(chat_id, now)
    return _finish(200, body, "answer")


class BedrockModel:
    """Converse adapter. Tests inject a stub and never construct this."""

    def __init__(self, kind: str, model_id: str) -> None:
        self.kind = kind
        self.model_id = model_id

    def converse(self, **kwargs: Any) -> Any:
        import boto3

        client = boto3.client("bedrock-runtime")
        reply = client.converse(
            modelId=kwargs["model_id"],
            system=[{"text": kwargs["system"]}],
            messages=[{"role": "user", "content": [{"text": kwargs["question"]}]}],
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
        theseus = BedrockModel("sonnet", live_model_id())
    return Deps(
        policy=policy,
        corpus=corpus,
        ledger=ledger_from_env(dict(source), policy),
        haiku=BedrockModel("haiku", helper_model_id()),
        theseus=theseus,
        bedrock_enabled=source.get("BEDROCK_ENABLED") == "1",
        now=lambda: datetime.now(timezone.utc),
        helper_model_id=helper_model_id(),
        live_model_id=live_model_id(),
    )


def lambda_handler(event: dict, context: Any) -> dict:
    return handle(event, deps_from_env())
