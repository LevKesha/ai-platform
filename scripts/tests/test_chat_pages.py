"""Chat wiring follows chat-policy.json pages and nothing else."""
from __future__ import annotations

import importlib.util
import json
import shutil
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


def test_source_html_does_not_hand_wire_chat() -> None:
    mod = load()
    olympus = ROOT / "olympus"
    pages = mod.chat_page_list(olympus)
    assert pages is not None
    assert "evidence.html" in pages
    assert mod.chat_script_srcs(olympus) == {}
    for rel in pages:
        text = (olympus / rel).read_text(encoding="utf-8")
        href = f"{'../' * rel.count('/')}styles.css"
        assert f'href="{href}"' in text
    console = (olympus / "console.html").read_text(encoding="utf-8")
    assert "IntersectionObserver" in console
    assert 'querySelector(".chat-bubble")' in console
    css = (olympus / "styles.css").read_text(encoding="utf-8")
    assert "body.page-console:has(.console-shell) .chat-bubble" in css
    assert ".page-evidence:has(.chat-bubble)" in css
    assert "if (!b) return;" in console


def _copy_with_flag(tmp_path: Path, enabled: bool) -> Path:
    dest = tmp_path / "olympus"
    shutil.copytree(ROOT / "olympus", dest)
    policy_path = dest / "chat-policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    policy["enabled"] = enabled
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return dest


def _expect_listed(mod, dest: Path) -> None:
    pages = mod.chat_page_list(dest)
    srcs = mod.chat_script_srcs(dest)
    assert pages is not None
    assert set(srcs) == set(pages)
    for rel, found in srcs.items():
        expect = f"{'../' * rel.count('/')}chat.js?v={VERSION}"
        assert found == [expect]
    assert '<script src="chat.js?v=abc1234"></script>' in (dest / "evidence.html").read_text(encoding="utf-8")


def test_stamp_follows_the_flag_in_chat_policy(tmp_path: Path) -> None:
    """CI guard: the committed flag decides wiring. It is not hardcoded here."""
    mod = load()
    policy = json.loads((ROOT / "olympus" / "chat-policy.json").read_text(encoding="utf-8"))
    enabled = policy["enabled"]
    assert isinstance(enabled, bool)
    dest = tmp_path / "olympus"
    shutil.copytree(ROOT / "olympus", dest)
    assert mod.stamp_tree(dest, VERSION) > 0
    assert mod.stamp_tree(dest, VERSION) == 0
    assert mod.chat_enabled(dest) is enabled
    if enabled:
        _expect_listed(mod, dest)
        console = (dest / "console.html").read_text(encoding="utf-8")
        assert console.index("IntersectionObserver") < console.index("chat.js?v=abc1234")
    else:
        assert mod.chat_script_srcs(dest) == {}
        assert "chat.js" not in (dest / "evidence.html").read_text(encoding="utf-8")
        assert "chat.js" not in (dest / "console.html").read_text(encoding="utf-8")
    assert mod.asset_refs_not_at_version(dest, VERSION) == []
    assert mod.hand_versioned_refs(ROOT / "olympus") == []


def test_flag_false_stamps_no_chat_script(tmp_path: Path) -> None:
    mod = load()
    dest = _copy_with_flag(tmp_path, False)
    mod.stamp_tree(dest, VERSION)
    assert mod.chat_enabled(dest) is False
    assert mod.chat_script_srcs(dest) == {}


def test_flag_true_stamps_only_listed_pages(tmp_path: Path) -> None:
    mod = load()
    dest = _copy_with_flag(tmp_path, True)
    mod.stamp_tree(dest, VERSION)
    assert mod.chat_enabled(dest) is True
    _expect_listed(mod, dest)


def test_a_page_outside_the_list_is_not_a_chat_page(tmp_path: Path) -> None:
    mod = load()
    root = tmp_path / "site"
    root.mkdir()
    (root / "chat-policy.json").write_text(
        json.dumps({"enabled": True, "pages": ["keep.html"]}),
        encoding="utf-8",
    )
    (root / "keep.html").write_text("<html><body><p>keep</p></body></html>", encoding="utf-8")
    (root / "skip.html").write_text(
        '<html><body><script src="chat.js"></script></body></html>',
        encoding="utf-8",
    )
    mod.stamp_tree(root, VERSION)
    srcs = mod.chat_script_srcs(root)
    assert set(srcs) != set(mod.chat_page_list(root))
    assert srcs["skip.html"] == ["chat.js?v=abc1234"]
    assert srcs["keep.html"] == ["chat.js?v=abc1234"]


def test_policy_without_pages_fails(tmp_path: Path) -> None:
    mod = load()
    root = tmp_path / "site"
    root.mkdir()
    (root / "chat-policy.json").write_text(json.dumps({"enabled": True}), encoding="utf-8")
    (root / "index.html").write_text("<html><body></body></html>", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing pages"):
        mod.stamp_tree(root, VERSION)


def test_listed_page_without_body_fails(tmp_path: Path) -> None:
    mod = load()
    root = tmp_path / "site"
    root.mkdir()
    (root / "chat-policy.json").write_text(
        json.dumps({"enabled": True, "pages": ["index.html"]}),
        encoding="utf-8",
    )
    (root / "index.html").write_text("<html><p>no body</p></html>", encoding="utf-8")
    with pytest.raises(SystemExit, match="no </body>"):
        mod.stamp_tree(root, VERSION)


def test_policy_without_enabled_fails(tmp_path: Path) -> None:
    mod = load()
    root = tmp_path / "site"
    root.mkdir()
    (root / "chat-policy.json").write_text(json.dumps({"pages": ["index.html"]}), encoding="utf-8")
    (root / "index.html").write_text("<html><body></body></html>", encoding="utf-8")
    with pytest.raises(SystemExit, match="enabled must be a boolean"):
        mod.stamp_tree(root, VERSION)


def test_corpus_extractor_does_not_read_chat_policy() -> None:
    text = (ROOT / "packages" / "chat_helper" / "corpus.py").read_text(encoding="utf-8")
    assert "chat-policy.json" not in text
