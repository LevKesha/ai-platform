"""RAG Lambda package and the frozen handler's flag-0 gate."""
from __future__ import annotations

import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages"))

from chat_helper.handler import deps_from_env, handle  # noqa: E402
from chat_helper.policy import load_policy  # noqa: E402
from chat_helper.rag import (  # noqa: E402
    SDK_CLIENTS,
    RagDeps,
    fixed_url,
    handle_rag,
    rag_system,
)
from chat_helper.tests.test_chat import Stub, _deps, _event, _json  # noqa: E402

CERT = "AWS Certified Solutions Architect – Professional (SAP-C02), 2026."


def _question() -> str:
    return load_policy()["suggested"][1]


class Reply:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = {"input_tokens": 3, "output_tokens": 2}


def _rag_deps(status: str = "live", post=None, complete=None, secret=None) -> RagDeps:
    calls = {"post": 0, "secret": 0, "complete": 0}

    def read_status() -> str:
        if isinstance(status, Exception):
            raise status
        return status

    def read_secret() -> str:
        calls["secret"] += 1
        if secret is not None:
            return secret()
        return "token-value"

    def send(url: str, body: dict, token: str) -> dict:
        calls["post"] += 1
        if post is not None:
            return post(url, body, token)
        return {"chunks": [CERT]}

    def run(model: str, system: str, question: str, history: list):
        calls["complete"] += 1
        if complete is not None:
            return complete(model, system, question, history)
        return Reply(CERT)

    deps = RagDeps(
        status=read_status,
        secret=read_secret,
        post=send,
        complete=run,
        model_id="helper-slot",
        url="https://rag.internal/retrieve",
    )
    deps.calls = calls  # type: ignore[attr-defined]
    return deps


def test_flag_off_does_not_invoke_rag() -> None:
    def boom(question: str, history: list) -> dict:
        raise AssertionError("rag invoked")

    deps = _deps(Stub("haiku"))
    deps.rag_enabled = False
    deps.rag_invoke = boom
    payload = _json(handle(_event(_question()), deps))
    assert payload["kind"] == "answer"
    assert payload["label"] == load_policy()["label_cv"]


def test_flag_defaults_to_zero() -> None:
    assert deps_from_env({}).rag_enabled is False
    assert deps_from_env({"CHAT_RAG_ENABLED": "0"}).rag_enabled is False
    enabled = deps_from_env({"CHAT_RAG_ENABLED": "1"})
    assert enabled.rag_enabled is True
    assert enabled.rag_invoke is not None


def test_rag_retry_never_falls_through_to_cv() -> None:
    haiku = Stub("haiku")
    deps = _deps(haiku)
    deps.rag_enabled = True
    deps.rag_invoke = lambda question, history: {"kind": "retry"}
    payload = _json(handle(_event(_question(), idem="rag-retry"), deps))
    assert payload["kind"] == "retry"
    assert payload["text"] == load_policy()["retry"]
    assert "Lev has this on his CV" not in payload["text"]
    assert haiku.calls == []


def test_rag_asleep_uses_the_cv_label() -> None:
    haiku = Stub("haiku")
    deps = _deps(haiku)
    deps.rag_enabled = True
    deps.rag_invoke = lambda question, history: {"kind": "cv"}
    payload = _json(handle(_event(_question(), idem="rag-cv"), deps))
    assert payload["label"] == load_policy()["label_cv"]
    assert len(haiku.calls) == 1


def test_grounded_draft_uses_label_rag_and_skips_haiku() -> None:
    haiku = Stub("haiku")
    deps = _deps(haiku)
    deps.rag_enabled = True
    deps.rag_invoke = lambda question, history: {
        "kind": "draft",
        "text": CERT,
        "chunks": [CERT],
        "usage": {"input_tokens": 4, "output_tokens": 4},
    }
    payload = _json(handle(_event(_question(), idem="rag-ok"), deps))
    assert payload["kind"] == "answer"
    assert payload["label"] == load_policy()["label_rag"]
    assert haiku.calls == []


def test_ungrounded_draft_retries_and_does_not_compose() -> None:
    haiku = Stub("haiku")
    deps = _deps(haiku)
    deps.rag_enabled = True
    deps.rag_invoke = lambda question, history: {
        "kind": "draft",
        "text": "Lev has a PhD from MIT in 1999.",
        "chunks": [CERT],
    }
    payload = _json(handle(_event(_question(), idem="rag-bad"), deps))
    assert payload["kind"] == "retry"
    assert "Lev has this on his CV" not in json.dumps(payload)
    assert haiku.calls == []


def test_status_failure_is_retry_and_logs_no_content() -> None:
    question = _question()
    deps = _rag_deps(status=RuntimeError("status down at rag.internal"))
    buf = io.StringIO()
    with redirect_stdout(buf):
        body = handle_rag({"question": question, "url": "https://evil.example/retrieve", "model": "other"}, deps)
    assert body == {"kind": "retry"}
    assert deps.calls["post"] == 0
    log = buf.getvalue()
    assert question not in log
    assert "evil.example" not in log
    assert "rag.internal" not in log
    assert "token-value" not in log
    assert json.loads(log)["path"] == "retry"


def test_asleep_returns_cv_without_a_fetch() -> None:
    deps = _rag_deps(status="waking")
    buf = io.StringIO()
    with redirect_stdout(buf):
        body = handle_rag({"question": _question()}, deps)
    assert body == {"kind": "cv"}
    assert deps.calls["post"] == 0
    assert deps.calls["secret"] == 0
    assert "waking" not in buf.getvalue()


def test_live_uses_the_fixed_url_and_model() -> None:
    seen = {}

    def post(url, body, token):
        seen["url"] = url
        seen["token"] = token
        seen["body"] = body
        return {"chunks": [CERT]}

    def complete(model, system, question, history):
        seen["model"] = model
        seen["system"] = system
        return Reply(CERT)

    deps = _rag_deps(post=post, complete=complete)
    buf = io.StringIO()
    with redirect_stdout(buf):
        body = handle_rag(
            {
                "question": _question(),
                "url": "https://evil.example/steal",
                "model": "other-model",
            },
            deps,
        )
    assert body["kind"] == "draft"
    assert body["text"] == CERT
    assert seen["url"] == "https://rag.internal/retrieve"
    assert seen["model"] == "helper-slot"
    assert "----" in seen["system"]
    assert seen["token"] == "token-value"
    assert "evil.example" not in buf.getvalue()
    assert CERT not in buf.getvalue()
    assert "token-value" not in buf.getvalue()


def test_fixed_url_rejects_user_shaped_values() -> None:
    assert fixed_url({"RAG_RETRIEVE_URL": "https://rag.internal/retrieve"}) == "https://rag.internal/retrieve"
    for raw in (
        "http://rag.internal/retrieve",
        "https://user:pw@rag.internal/retrieve",
        "https://rag.internal/other",
        "https://rag.internal/retrieve?x=1",
        "",
    ):
        with pytest.raises(Exception):
            fixed_url({"RAG_RETRIEVE_URL": raw})


def test_retrieve_accepts_rag_service_chunk_objects() -> None:
    """Items match rag-service /retrieve: {"source": str, "content": str}."""
    from chat_helper.rag import RagError, _chunks

    content = "Oracle Cloud Infrastructure Architect – Associate (1Z0-1072-26), Oracle Academy, 2026."
    payload = {
        "chunks": [
            {"source": "olympus/proof.html", "content": content},
            {"source": "olympus/index.html", "content": None},
            {"source": "olympus/index.html", "content": ["not", "a", "string"]},
            CERT,
        ]
    }
    assert _chunks(payload) == [content, CERT]
    with pytest.raises(RagError):
        _chunks({"chunks": [{"source": "a", "content": 1}]})

    seen = {}

    def post(url, body, token):
        return payload

    def complete(model, system, question, history):
        seen["system"] = system
        return Reply("grounded")

    body = handle_rag({"question": _question()}, _rag_deps(post=post, complete=complete))
    assert body["kind"] == "draft"
    assert content in seen["system"]
    assert CERT in seen["system"]


def test_retrieve_url_requires_https_and_verified_tls(monkeypatch) -> None:
    import ssl

    from chat_helper.rag import RagError, _tls_context, post_json

    with pytest.raises(RagError):
        fixed_url({"RAG_RETRIEVE_URL": "http://rag.internal/retrieve"})
    ctx = _tls_context()
    assert ctx.verify_mode == ssl.CERT_REQUIRED
    assert ctx.check_hostname is True
    source = (ROOT / "packages" / "chat_helper" / "rag.py").read_text(encoding="utf-8")
    assert "CERT_NONE" not in source
    assert "verify=False" not in source
    with pytest.raises(RagError):
        post_json("http://rag.internal/retrieve", {"question": "hi"}, "token")

    def unverified() -> ssl.SSLContext:
        bad = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        bad.check_hostname = False
        bad.verify_mode = ssl.CERT_NONE
        return bad

    monkeypatch.setattr(ssl, "create_default_context", unverified)
    with pytest.raises(RagError):
        _tls_context()


def test_sdk_clients_are_only_bedrock_secrets_and_dynamo() -> None:
    source = (ROOT / "packages" / "chat_helper" / "rag.py").read_text(encoding="utf-8")
    assert "requests" not in source
    assert "httpx" not in source
    assert "aiohttp" not in source
    assert set(SDK_CLIENTS) == {"bedrock-runtime", "secretsmanager", "dynamodb"}
    assert "eu.anthropic" not in source
    assert "boto3.client(\"s3\")" not in source
    assert "boto3.client(\"lambda\")" not in source
    assert "boto3.client(\"sts\")" not in source


def test_system_text_is_not_halved() -> None:
    chunks = [CERT, "Networking and Systems Administration, ICT Corps Training Base, IDF, 2012."]
    system = rag_system(chunks)
    assert system.count("----") == 1
    assert CERT in system
    assert "untrusted data" in system


def test_requirements_are_hash_locked_without_an_http_library() -> None:
    text = (ROOT / "packages" / "chat_helper" / "requirements.txt").read_text(encoding="utf-8")
    assert "--hash=sha256:" in text
    names = [line.split("==", 1)[0].strip() for line in text.splitlines() if "==" in line and not line.startswith("#")]
    assert "boto3" in names
    assert "requests" not in names
    assert "httpx" not in names
    assert re.search(r"^boto3==", text, re.M)


def test_repo_does_not_turn_the_rag_flag_on() -> None:
    needle = "CHAT_RAG_ENABLED" + "=1"
    quoted = 'CHAT_RAG_ENABLED": "' + "1" + '"'
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".yml", ".yaml", ".json", ".js", ".tf"}:
            continue
        if ".git" in path.parts or path.name == "test_rag.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert needle not in text
        assert quoted not in text
