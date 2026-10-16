"""Console dashboards, soft pages, and the OCI cert replace."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROOF_ORDER = (
    "AWS Certified Solutions Architect – Professional (SAP-C02), 2026.",
    "Oracle Cloud Infrastructure Architect – Associate (1Z0-1072-26), Oracle Academy, 2026.",
    "DevSecOps, Bar-Ilan University, 2025.",
    "VMware VCP 6.5, HPE Academy, 2019.",
    "Aruba ACSA, Aruba Academy, 2019.",
    "IT Infrastructure Management (1021), MAMRAM, IDF, 2012.",
    "Networking and Systems Administration, ICT Corps Training Base, IDF, 2012.",
)
OLYMPUS_ASLEEP = (
    "Asleep right now. Request a look at olympus.levkesha.com. "
    "It wakes at the time you pick and is live in about 7 min."
)


def test_proof_certs_are_in_the_demo_order() -> None:
    text = (ROOT / "olympus" / "proof.html").read_text(encoding="utf-8")
    indexes = [text.index(line) for line in PROOF_ORDER]
    assert indexes == sorted(indexes)
    assert "OCI Foundations" not in text


def test_corpus_replaces_only_the_oci_cert() -> None:
    lines = json.loads((ROOT / "olympus" / "chat-corpus.json").read_text(encoding="utf-8"))["lines"]
    certs = [line["text"] for line in lines if line["section"] == "certifications"]
    assert "Oracle Cloud Infrastructure Architect – Associate (1Z0-1072-26), Oracle Academy, 2026." in certs
    blob = "\n".join(certs)
    assert "OCI Foundations" not in blob
    assert "DevSecOps" not in blob
    assert "VCP 6.5" not in blob
    assert "ACSA" not in blob
    assert "SAP-C02" in blob
    assert "1021" in blob
    assert "Networking and Systems Administration" in blob


def _inline_function(html: str) -> bytes:
    """Inline viewer function that returns the page. 10 KB is the platform cap."""
    escaped = json.dumps(html)[1:-1]
    source = (
        "function handler(event) {\n"
        f'  var html = "{escaped}";\n'
        "  return {\n"
        "    statusCode: 503,\n"
        '    statusDescription: "Service Unavailable",\n'
        "    headers: {\n"
        '      "content-type": { value: "text/html; charset=utf-8" },\n'
        '      "cache-control": { value: "no-store" }\n'
        "    },\n"
        "    body: html\n"
        "  };\n"
        "}\n"
    )
    return source.encode("utf-8")


def test_soft_pages_match_the_copy_and_have_no_controls() -> None:
    olympus = (ROOT / "olympus" / "asleep.html").read_text(encoding="utf-8")
    n8n = (ROOT / "olympus" / "asleep-n8n.html").read_text(encoding="utf-8")
    assert OLYMPUS_ASLEEP in olympus
    assert "Asleep right now." in n8n
    assert OLYMPUS_ASLEEP not in n8n
    for text in (olympus, n8n):
        lowered = text.lower()
        assert "<a " not in lowered
        assert "<button" not in lowered
        assert "<link" not in lowered
        assert "#0a0a0a" in text
    assert "<svg" in olympus
    assert "<svg" not in n8n.lower()
    assert "Lev approves" not in n8n
    assert "about 7" not in n8n
    paragraphs = [part.split("<", 1)[0] for part in n8n.split("<p>")[1:]]
    assert paragraphs == ["Asleep right now."]
    for name in ("asleep.html", "asleep-n8n.html"):
        size = len(_inline_function((ROOT / "olympus" / name).read_text(encoding="utf-8")))
        assert size < 10 * 1024, name
    spend = (ROOT / "olympus" / "diagrams" / "spend-loop.svg").read_text(encoding="utf-8")
    assert "wake-only" in spend
    assert "while live" not in spend
    assert "compress" in spend
    assert "n8n workflows" in spend
    assert "M232,292 L376,244" not in spend
    assert "admin · Lev only" in spend
    data = (ROOT / "olympus" / "console-data.js").read_text(encoding="utf-8")
    assert "Wake-only." in data
    assert "admin · Lev only" in data
    subline = "Every CV×Jobs AI request goes through one place and gets counted."
    assert subline in data
    corpus = (ROOT / "olympus" / "chat-corpus.json").read_text(encoding="utf-8")
    assert subline in corpus
    assert "Every AI request goes through one place" not in data
    assert "Every AI request goes through one place" not in corpus


def test_new_copy_has_no_doc_count() -> None:
    needles = ("11 doc", "11 page", "11 source", "12 source", "12 doc", "doc-count", "documents:")
    for rel in (
        "olympus/asleep.html",
        "olympus/asleep-n8n.html",
        "olympus/console.js",
        "olympus/chat.js",
        "olympus/chat-policy.json",
        "packages/chat_helper/rag.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8").lower()
        for needle in needles:
            assert needle not in text, rel
