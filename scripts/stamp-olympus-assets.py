#!/usr/bin/env python3
"""Stamp local Olympus asset URLs with ?v=<commit sha> before upload.

Source files stay unversioned. Deploy runs this on the checkout, then uploads
with the Cache-Control map from cache_metadata(). External URLs are unchanged.

  python3 scripts/stamp-olympus-assets.py olympus --version "$SHORT_SHA"
  python3 scripts/stamp-olympus-assets.py olympus --print-cache-manifest
  python3 scripts/stamp-olympus-assets.py --cache-metadata cursor/index.html
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

HTML_CACHE = "no-cache"
JSON_CACHE = "no-cache"
OTHER_CACHE = "no-cache"
ASSET_CACHE = "public, max-age=31536000, immutable"

JSON_FETCHES = frozenset({"chat-policy.json", "chat-corpus.json"})
VERSION_RE = re.compile(r"[0-9a-f]{7,40}")

ATTR_RE = re.compile(
    r"""(?P<prefix>\b(?:href|src)\s*=\s*)(?P<q>["'])(?P<url>[^"']*)(?P=q)""",
    re.IGNORECASE,
)
CSS_URL_RE = re.compile(
    r"""url\(\s*(?:(?P<q>["'])(?P<quoted>[^"']*)(?P=q)|(?P<bare>[^"')\s]+))\s*\)""",
    re.IGNORECASE,
)
QUOTED_RE = re.compile(r"""(?P<q>["'])(?P<url>[^"'\\\s]*)(?P=q)""")


def cache_metadata(rel: str) -> tuple[str, str]:
    """Return (Cache-Control, Content-Type) for one olympus-relative path.

    HTML is revalidated every load so the new ?v= is visible.
    styles.css, every .js, and png/svg are stamped, so they can be immutable.
    Other CSS (auth/hosted-ui.css) is not stamped, so it stays no-cache.
    JSON is no-cache even when the fetch URL is stamped: chat.js resolves
    new URL("chat-policy.json", script.src) by dropping the script query, and
    a cache that ignores query strings must still not keep a stale corpus.
    """
    path = rel.replace("\\", "/").lstrip("/")
    if path == "README.md":
        raise ValueError("README.md is not uploaded")
    name = path.rsplit("/", 1)[-1]
    if path.endswith(".html"):
        return HTML_CACHE, "text/html; charset=utf-8"
    if path.endswith(".js"):
        return ASSET_CACHE, "text/javascript; charset=utf-8"
    if name == "styles.css":
        return ASSET_CACHE, "text/css; charset=utf-8"
    if path.endswith(".css"):
        return OTHER_CACHE, "text/css; charset=utf-8"
    if path.endswith(".json"):
        return JSON_CACHE, "application/json; charset=utf-8"
    if path.endswith(".png"):
        return ASSET_CACHE, "image/png"
    if path.endswith(".svg"):
        return ASSET_CACHE, "image/svg+xml"
    if path.endswith(".drawio"):
        return OTHER_CACHE, "application/xml"
    if path.endswith(".sh"):
        return OTHER_CACHE, "text/x-shellscript; charset=utf-8"
    raise ValueError(f"no cache rule for {path}")


def iter_cache_manifest(root: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == "README.md":
            continue
        cache, ctype = cache_metadata(rel)
        rows.append((rel, cache, ctype))
    return rows


def _raw(url: str) -> tuple[str, bool]:
    amp = "&amp;" in url
    return url.strip().replace("&amp;", "&"), amp


def _restore_amp(url: str, amp: bool) -> str:
    if amp:
        return url.replace("&", "&amp;")
    return url


def is_external(url: str) -> bool:
    text, _amp = _raw(url)
    if text.startswith("//"):
        return True
    parts = urlsplit(text)
    return bool(parts.scheme or parts.netloc)


def _path(url: str) -> str:
    text, _amp = _raw(url)
    return urlsplit(text).path


def _basename(url: str) -> str:
    path = _path(url)
    return path.rsplit("/", 1)[-1]


def is_html_asset(url: str) -> bool:
    if not url or is_external(url):
        return False
    path = _path(url)
    if not path or path.endswith("/"):
        return False
    if _basename(url) == "styles.css":
        return True
    return path.endswith((".js", ".png", ".svg"))


def is_css_asset(url: str) -> bool:
    if not url or is_external(url):
        return False
    path = _path(url)
    return path.endswith((".png", ".svg"))


def is_json_fetch(url: str) -> bool:
    if not url or is_external(url):
        return False
    return _basename(url) in JSON_FETCHES


def query_version(url: str) -> str | None:
    text, _amp = _raw(url)
    for key, value in parse_qsl(urlsplit(text).query, keep_blank_values=True):
        if key == "v":
            return value
    return None


def with_version(url: str, version: str) -> str:
    text, amp = _raw(url)
    parts = urlsplit(text)
    pairs = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True) if key != "v"]
    pairs.append(("v", version))
    rebuilt = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(pairs), parts.fragment))
    return _restore_amp(rebuilt, amp)


def stamp_html(text: str, version: str) -> str:
    def repl(match: re.Match[str]) -> str:
        url = match.group("url")
        if not is_html_asset(url):
            return match.group(0)
        return f"{match.group('prefix')}{match.group('q')}{with_version(url, version)}{match.group('q')}"

    return ATTR_RE.sub(repl, text)


def stamp_css(text: str, version: str) -> str:
    def repl(match: re.Match[str]) -> str:
        url = match.group("quoted") if match.group("q") else match.group("bare")
        if not is_css_asset(url):
            return match.group(0)
        quote = match.group("q") or ""
        return f"url({quote}{with_version(url, version)}{quote})"

    return CSS_URL_RE.sub(repl, text)


def stamp_js(text: str, version: str) -> str:
    def repl(match: re.Match[str]) -> str:
        url = match.group("url")
        if not is_json_fetch(url):
            return match.group(0)
        return f"{match.group('q')}{with_version(url, version)}{match.group('q')}"

    return QUOTED_RE.sub(repl, text)


def _stamp_file(path: Path, version: str) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".html":
        rewriter = stamp_html
    elif suffix == ".css":
        rewriter = stamp_css
    elif suffix == ".js":
        rewriter = stamp_js
    else:
        return False
    original = path.read_text(encoding="utf-8")
    updated = rewriter(original, version)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def stamp_tree(root: Path, version: str) -> int:
    changed = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if _stamp_file(path, version):
            changed += 1
    return changed


def _refs_in_text(kind: str, text: str) -> list[str]:
    found: list[str] = []
    if kind == "html":
        for match in ATTR_RE.finditer(text):
            url = match.group("url")
            if is_html_asset(url):
                found.append(url)
    elif kind == "css":
        for match in CSS_URL_RE.finditer(text):
            url = match.group("quoted") if match.group("q") else match.group("bare")
            if is_css_asset(url):
                found.append(url)
    elif kind == "js":
        for match in QUOTED_RE.finditer(text):
            url = match.group("url")
            if is_json_fetch(url):
                found.append(url)
    return found


def _walk_refs(root: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        kind = {".html": "html", ".css": "css", ".js": "js"}.get(suffix)
        if kind is None:
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for url in _refs_in_text(kind, text):
            rows.append((rel, url))
    return rows


def unstamped_asset_refs(root: Path) -> list[str]:
    """Local stamp targets that have no non-empty ?v=."""
    missing: list[str] = []
    for rel, url in _walk_refs(root):
        version = query_version(url)
        if not version:
            missing.append(f"{rel}: {url}")
    return missing


def hand_versioned_refs(root: Path) -> list[str]:
    """Stamp targets that already carry ?v= in source. Deploy owns that query."""
    handed: list[str] = []
    for rel, url in _walk_refs(root):
        if query_version(url) is not None:
            handed.append(f"{rel}: {url}")
    return handed


def _version(value: str) -> str:
    version = value.strip().lower()
    if VERSION_RE.fullmatch(version) is None:
        raise SystemExit("version must be a 7-40 character hex commit sha")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stamp Olympus asset URLs with a commit sha.")
    parser.add_argument("root", nargs="?", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--print-cache-manifest", action="store_true")
    parser.add_argument("--cache-metadata", metavar="REL")
    args = parser.parse_args(argv)

    if args.cache_metadata:
        cache, ctype = cache_metadata(args.cache_metadata)
        sys.stdout.write(f"{cache}\t{ctype}\n")
        return 0

    if args.root is None or not args.root.is_dir():
        print("root must be the olympus directory", file=sys.stderr)
        return 2

    if args.print_cache_manifest:
        for rel, cache, ctype in iter_cache_manifest(args.root):
            sys.stdout.write(f"{rel}\t{cache}\t{ctype}\n")
        return 0

    if not args.version:
        print("--version is required to stamp", file=sys.stderr)
        return 2
    version = _version(args.version)
    changed = stamp_tree(args.root, version)
    print(f"stamped {changed} file(s) with v={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
