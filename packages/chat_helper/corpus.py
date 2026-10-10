"""Frozen corpus built from the CV and the Olympus portal text."""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9][a-z0-9+\-]{2,}")
STOP = frozenset(
    """
    a an the and or of to for on in with from by at as is are was were be been being
    this that these those it its into over than then also not but about your you
    what which when where who how does do did has have had can will just only
    lev his him her their he she they
    """.split()
)
OFF_TOPIC_CUES = (
    "weather",
    "joke",
    "poem",
    "recipe",
    "sport",
    "movie",
    "song",
    "bitcoin",
    "stock price",
    "girlfriend",
    "politics",
    "religion",
    "ignore previous",
    "ignore your",
    "ignore all",
    "system prompt",
    "jailbreak",
    "developer message",
)
SECTION_CUES = {
    "certifications": ("certif", "degree", "education", "university", "studied", "study"),
    "skills": (
        "skill",
        "aws",
        "azure",
        "terraform",
        "linux",
        "windows",
        "vmware",
        "docker",
        "kubernetes",
        "eks",
    ),
    "timeline": ("career", "employ", "solaredge", "worked", "job history"),
    "platform": ("platform", "olympus", "this site", "n8n", "bedrock", "github"),
}
DROP_LINE = ("asleep", "waking")
HTML_PAGES = (
    "olympus/index.html",
    "olympus/proof.html",
    "olympus/architecture.html",
    "olympus/architecture-view.html",
    "olympus/evidence.html",
    "olympus/cursor/index.html",
    "olympus/console.html",
    "olympus/404.html",
    "olympus/gated.html",
)
JS_PAGES = ("olympus/public-data.js", "olympus/console-data.js")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "platform-config.yaml").is_file():
            return parent
    raise FileNotFoundError("platform-config.yaml not found")


def corpus_path() -> Path:
    return repo_root() / "olympus" / "chat-corpus.json"


class _MainText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip: list[str] = []
        self.in_main = 0
        self.block: str | None = None
        self.buf: list[str] = []
        self.heading = "overview"
        self.rows: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = []
        for key, value in attrs:
            if key == "class" and value:
                classes.extend(value.split())
        hidden = any(key == "aria-hidden" and value == "true" for key, value in attrs)
        # The hero region-note is a commerce CTA inside <main>, not a CV line.
        if tag in {"header", "nav", "script", "style"} or hidden or "region-note" in classes:
            self.skip.append(tag)
        if tag == "main":
            self.in_main += 1
        if self.skip or not self.in_main:
            return
        if tag in {"h1", "h2", "h3"}:
            self._flush(tag)
            self.block = "heading"
        elif tag in {"p", "li", "figcaption"}:
            self._flush(tag)
            self.block = "line"

    def handle_endtag(self, tag: str) -> None:
        if self.block and not self.skip and tag in {"h1", "h2", "h3", "p", "li", "figcaption"}:
            self._flush(tag)
        if self.skip and self.skip[-1] == tag:
            self.skip.pop()
        if tag == "main" and self.in_main:
            self.in_main -= 1

    def handle_data(self, data: str) -> None:
        if self.block and not self.skip and self.in_main:
            self.buf.append(data)

    def _flush(self, tag: str) -> None:
        text = " ".join("".join(self.buf).split())
        self.buf = []
        if self.block == "heading" and text:
            self.heading = text
        elif self.block == "line" and len(text) >= 40:
            self.rows.append((self.heading, text))
        self.block = None


def _section(heading: str) -> str:
    folded = heading.lower()
    if "certif" in folded or "education" in folded:
        return "certifications"
    if "skill" in folded:
        return "skills"
    if "timeline" in folded:
        return "timeline"
    if "evidence" in folded or "inventor" in folded or "skip" in folded:
        return "evidence"
    if "architect" in folded:
        return "architecture"
    return "overview"


_BIND_PORT = re.compile(r":\d{2,5}\b")


def _keep(text: str) -> bool:
    folded = text.lower()
    if any(word in folded for word in DROP_LINE):
        return False
    # Edge bind paths (agent-api :8000 and the same class of port) are routes.
    if _BIND_PORT.search(text):
        return False
    return True


def _js_strings(text: str) -> list[str]:
    """Read double-quoted literals from their opening quote.

    A search that may start at any quote treats the closing quote of a short
    literal as a new string. public-data.js line 28 (`iacBranch`) is that case:
    the characters up to the next quote are JS source, not a sentence.
    """
    rows: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            newline = text.find("\n", i + 2)
            i = n if newline < 0 else newline + 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch in {'"', "'", "`"}:
            quote = ch
            i += 1
            buf: list[str] = []
            while i < n:
                current = text[i]
                if current == "\\":
                    if i + 1 < n:
                        buf.append(text[i + 1])
                    i += 2
                    continue
                if current == quote:
                    i += 1
                    break
                buf.append(current)
                i += 1
            if quote != '"':
                continue
            line = " ".join("".join(buf).split())
            if len(line) < 40 or line.startswith("http") or not _keep(line):
                continue
            rows.append(line)
            continue
        i += 1
    return rows


def build_lines(root: Path | None = None) -> list[dict[str, str]]:
    root = root or repo_root()
    lines: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(source: str, section: str, text: str) -> None:
        if text in seen or not _keep(text):
            return
        seen.add(text)
        lines.append(
            {
                "id": f"{source.replace('/', '-')}-{len(lines) + 1}",
                "source": source,
                "section": section,
                "text": text,
            }
        )

    for rel in HTML_PAGES:
        parser = _MainText()
        parser.feed((root / rel).read_text(encoding="utf-8"))
        parser.close()
        for heading, text in parser.rows:
            add(rel, _section(heading), text)
    for rel in JS_PAGES:
        for text in _js_strings((root / rel).read_text(encoding="utf-8")):
            add(rel, "platform", text)
    return lines


def write_corpus(root: Path | None = None) -> Path:
    root = root or repo_root()
    path = root / "olympus" / "chat-corpus.json"
    payload = {
        "version": 1,
        "built_from": list(HTML_PAGES + JS_PAGES),
        "lines": build_lines(root),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_lines(path: Path | None = None) -> list[dict[str, str]]:
    data = json.loads((path or corpus_path()).read_text(encoding="utf-8"))
    return list(data["lines"])


def content_tokens(text: str) -> set[str]:
    return {token for token in TOKEN.findall(text.lower()) if token not in STOP}


def fact_tokens(text: str) -> set[str]:
    numbers = {item.replace(",", "") for item in re.findall(r"\d[\d,]*", text)}
    return content_tokens(text) | numbers


def grounded(answer: str, lines: list[dict[str, str]]) -> bool:
    blob = "\n".join(line["text"] for line in lines)
    return fact_tokens(answer) <= fact_tokens(blob)


def _overlap(question: str, line: str) -> int:
    return len(content_tokens(question) & content_tokens(line))


def classify(question: str, lines: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    folded = question.lower()
    if any(cue in folded for cue in OFF_TOPIC_CUES):
        return "off_topic", []
    sections = {
        name
        for name, cues in SECTION_CUES.items()
        if any(cue in folded for cue in cues)
    }
    scored: list[tuple[int, dict[str, str]]] = []
    for line in lines:
        score = _overlap(question, line["text"])
        if line["section"] in sections:
            score += 2
        if score:
            scored.append((score, line))
    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    picked = [line for _score, line in scored[:3]]
    if picked:
        return "answer", picked
    if sections or "lev" in folded or re.search(r"\b(he|his|him)\b", folded):
        return "not_in_cv", []
    return "off_topic", []


def sentences(text: str) -> list[str]:
    """Split on sentence punctuation that is followed by whitespace.

    A period inside an initialism such as B.A is not a sentence boundary.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def _sentence(line: str) -> str:
    text = " ".join(line.split())
    if text[-1:] not in ".!?":
        text += "."
    return text


def compose(lines: list[dict[str, str]]) -> str:
    """One prefix sentence plus at most two corpus sentences."""
    chosen: list[str] = []
    for line in lines:
        for part in sentences(line["text"]):
            if len(chosen) == 2:
                break
            chosen.append(_sentence(part))
        if len(chosen) == 2:
            break
    body = " ".join(chosen)
    if not body:
        return "Lev has this on his CV."
    return f"Lev has this on his CV. {body}"


_CACHE: dict[str, list] = {}


def reset_cache() -> None:
    _CACHE.clear()


def cached_lines(loader) -> list:
    if "lines" not in _CACHE:
        _CACHE["lines"] = loader()
    return _CACHE["lines"]


def s3_loader(client, bucket: str, key: str):
    def load() -> list:
        raw = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(raw)["lines"]

    return load


def system_prompt(lines: list[dict[str, str]]) -> str:
    quoted = "\n".join(f"- {line['text']}" for line in lines)
    return (
        "Answer only from the corpus lines below. Third person. "
        "Two or three plain sentences. Every fact must appear in these lines. "
        "If they do not cover the question, do not guess.\n"
        f"{quoted}"
    )

