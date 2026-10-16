"""Deploy-time Olympus asset stamps and Cache-Control map."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERSION = "abc1234"


def load():
    path = ROOT / "scripts" / "stamp-olympus-assets.py"
    spec = importlib.util.spec_from_file_location("stamp_olympus_assets", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixture_html_stamps_local_assets_and_keeps_external_urls(tmp_path: Path) -> None:
    mod = load()
    page = tmp_path / "cursor" / "index.html"
    page.parent.mkdir()
    page.write_text(
        "\n".join(
            [
                '<link rel="stylesheet" href="../styles.css">',
                "<link rel='stylesheet' href='styles.css?v=20261004-center'>",
                '<link rel="stylesheet" href="styles.css?theme=dark">',
                '<link rel="stylesheet" href="styles.css?v=old&amp;theme=dark#top">',
                '<link rel="stylesheet" href="https://cdn.example.com/styles.css">',
                '<script src="//cdn.example.com/app.js"></script>',
                '<script src="../chat.js"></script>',
                '<script src="/console.js"></script>',
                '<a href="index.html">Overview</a>',
                '<a href="console.html?state=form&amp;where=IL">Form</a>',
                '<img src="../diagrams/hero.png" alt="">',
                '<img src="https://cdn.example.com/hero.png" alt="">',
                '<img src="mark.svg#icon" alt="">',
            ]
        ),
        encoding="utf-8",
    )
    assert mod.stamp_tree(tmp_path, VERSION) == 1
    text = page.read_text(encoding="utf-8")
    assert 'href="../styles.css?v=abc1234"' in text
    assert "href='styles.css?v=abc1234'" in text
    assert 'href="styles.css?theme=dark&v=abc1234"' in text
    assert 'href="styles.css?theme=dark&amp;v=abc1234#top"' in text
    assert 'href="https://cdn.example.com/styles.css"' in text
    assert 'src="//cdn.example.com/app.js"' in text
    assert 'src="../chat.js?v=abc1234"' in text
    assert 'src="/console.js?v=abc1234"' in text
    assert 'href="index.html"' in text
    assert 'href="console.html?state=form&amp;where=IL"' in text
    assert 'src="../diagrams/hero.png?v=abc1234"' in text
    assert 'src="https://cdn.example.com/hero.png"' in text
    assert 'src="mark.svg?v=abc1234#icon"' in text
    assert mod.asset_refs_not_at_version(tmp_path, VERSION) == []
    assert mod.stamp_tree(tmp_path, VERSION) == 0


def test_css_url_and_json_fetches_are_stamped(tmp_path: Path) -> None:
    mod = load()
    css = tmp_path / "styles.css"
    css.write_text(
        "\n".join(
            [
                'a { background: url("../diagrams/a.png"); }',
                "b { background: url('mark.svg'); }",
                "c { background: url(https://cdn.example.com/a.png); }",
                'd { marker-end: url("#hub-arrow"); }',
                "e { mask: url(#hub-arrow); }",
            ]
        ),
        encoding="utf-8",
    )
    script = tmp_path / "chat.js"
    script.write_text(
        "\n".join(
            [
                'const policyUrl = new URL("chat-policy.json", script.src);',
                "const corpusUrl = new URL('chat-corpus.json', script.src);",
                'const keep = "not a fetch";',
                'const remote = "https://cdn.example.com/chat-policy.json";',
            ]
        ),
        encoding="utf-8",
    )
    mod.stamp_tree(tmp_path, VERSION)
    css_text = css.read_text(encoding="utf-8")
    js_text = script.read_text(encoding="utf-8")
    assert 'url("../diagrams/a.png?v=abc1234")' in css_text
    assert "url('mark.svg?v=abc1234')" in css_text
    assert "url(https://cdn.example.com/a.png)" in css_text
    assert 'url("#hub-arrow")' in css_text
    assert "url(#hub-arrow)" in css_text
    assert 'new URL("chat-policy.json?v=abc1234", script.src)' in js_text
    assert "new URL('chat-corpus.json?v=abc1234', script.src)" in js_text
    assert '"not a fetch"' in js_text
    assert '"https://cdn.example.com/chat-policy.json"' in js_text
    assert mod.asset_refs_not_at_version(tmp_path, VERSION) == []


def test_js_template_replaces_frozen_asset_token() -> None:
    mod = load()
    text = "\n".join(
        [
            "const src = `diagrams/x.svg?v=old`;",
            "const fig = '<img src=\"diagrams/cv-jobs-flow.svg?v=20261004-center\" alt=\"\">';",
            "const cls = event.target.closest(\".js-private\");",
            "const remote = `https://cdn.example.com/app.js`;",
            "const nested = `pre ${'chat.js'} post`;",
        ]
    )
    stamped = mod.stamp_js(text, VERSION)
    assert "const src = `diagrams/x.svg?v=abc1234`;" in stamped
    assert 'src="diagrams/cv-jobs-flow.svg?v=abc1234"' in stamped
    assert 'closest(".js-private")' in stamped
    assert "`https://cdn.example.com/app.js`" in stamped
    assert "`pre ${'chat.js?v=abc1234'} post`" in stamped


def test_guard_rejects_a_frozen_token_until_stamped(tmp_path: Path) -> None:
    mod = load()
    script = tmp_path / "console.js"
    script.write_text("const src = `diagrams/x.svg?v=old`;\n", encoding="utf-8")
    assert mod.asset_refs_not_at_version(tmp_path, VERSION) == ["console.js: diagrams/x.svg?v=old"]
    mod.stamp_tree(tmp_path, VERSION)
    assert mod.asset_refs_not_at_version(tmp_path, VERSION) == []


def test_existing_json_query_keeps_other_params() -> None:
    mod = load()
    text = 'fetch("chat-corpus.json?draft=1")'
    assert load().stamp_js(text, VERSION) == 'fetch("chat-corpus.json?draft=1&v=abc1234")'
    assert mod.stamp_js('fetch("chat-policy.json?v=old")', VERSION) == 'fetch("chat-policy.json?v=abc1234")'


def test_bad_version_is_rejected() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "stamp-olympus-assets.py"), "olympus", "--version", "hand-edited"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


def test_source_assets_have_no_hand_version() -> None:
    assert load().hand_versioned_refs(ROOT / "olympus") == []


def test_stamped_olympus_copy_has_no_bare_local_assets(tmp_path: Path) -> None:
    mod = load()
    dest = tmp_path / "olympus"
    shutil.copytree(ROOT / "olympus", dest)
    mod.stamp_tree(dest, VERSION)
    assert mod.asset_refs_not_at_version(dest, VERSION) == []
    cursor = (dest / "cursor" / "index.html").read_text(encoding="utf-8")
    assert 'href="../styles.css?v=abc1234"' in cursor
    if mod.chat_enabled(dest):
        assert 'src="../chat.js?v=abc1234"' in cursor
    else:
        assert "chat.js" not in cursor
    assert "olympus-agent-company-cursor-v5.png?v=abc1234" in cursor
    assert "https://github.com/LevKesha/ai-platform/pull/32" in cursor
    chat = (dest / "chat.js").read_text(encoding="utf-8")
    assert 'new URL("chat-policy.json?v=abc1234", script.src)' in chat
    assert 'new URL("chat-corpus.json?v=abc1234", script.src)' in chat
    proof = (dest / "proof.html").read_text(encoding="utf-8")
    assert "styles.css?v=abc1234" in proof
    assert "20261004-center" not in proof
    console = (dest / "console.html").read_text(encoding="utf-8")
    assert "url(#hub-arrow)" in console


def test_cache_metadata_for_each_uploaded_olympus_file() -> None:
    mod = load()
    seen: set[str] = set()
    for path in (ROOT / "olympus").rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT / "olympus").as_posix()
        if rel == "README.md":
            with pytest.raises(ValueError, match="README.md"):
                mod.cache_metadata(rel)
            continue
        cache, ctype = mod.cache_metadata(rel)
        seen.add(rel)
        if Path(rel).name in {"asleep.html", "asleep-n8n.html"}:
            assert cache == "no-store", rel
        elif rel.endswith((".html", ".json")) or rel == "auth/hosted-ui.css":
            assert cache == "no-cache", rel
        if rel.endswith((".js", ".png", ".svg")) or rel == "styles.css":
            assert cache == mod.ASSET_CACHE, rel
        assert ctype
    assert "chat-policy.json" in seen
    assert "chat-corpus.json" in seen
    assert "styles.css" in seen
    assert "cursor/index.html" in seen
    with pytest.raises(ValueError, match="no cache rule"):
        mod.cache_metadata("notes.txt")


def test_manifest_matches_cache_metadata() -> None:
    mod = load()
    rows = mod.iter_cache_manifest(ROOT / "olympus")
    rels = [rel for rel, _cache, _ctype in rows]
    assert "README.md" not in rels
    assert rels == sorted(rels)
    for rel, cache, ctype in rows:
        assert (cache, ctype) == mod.cache_metadata(rel)


def test_deploy_workflow_stamps_uploads_headers_and_invalidates() -> None:
    text = (ROOT / ".github/workflows/deploy-olympus.yml").read_text(encoding="utf-8")
    assert 'python3 scripts/stamp-olympus-assets.py olympus --version "${GITHUB_SHA:0:7}"' in text
    assert "python3 scripts/stamp-olympus-assets.py olympus --print-cache-manifest" in text
    assert "--cache-metadata cursor/index.html" in text
    assert 'create-invalidation --distribution-id "${DIST_ID}" --paths "/*"' in text
    assert "max-age=300" not in text
