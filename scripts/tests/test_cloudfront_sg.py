"""Empty or missing CloudFront SG ids fail closed. No live AWS."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str, filename: str | None = None):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (filename or f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_and_empty_sg_id_fail() -> None:
    sg = _load("cloudfront_sg")
    key = "n8n_cloudfront_sg_id"
    for data in ({}, {key: None}, {key: ""}, {key: "  "}):
        try:
            sg.require_sg_id(data, key)
        except ValueError as exc:
            assert "missing or empty" in str(exc)
        else:
            raise AssertionError(data)


def test_render_refuses_empty_sg_id() -> None:
    render = _load("render_from_infra", "render-from-infra.py")
    try:
        render.n8n_sg_annotations({})
    except SystemExit as exc:
        assert "missing or empty" in str(exc)
    else:
        raise AssertionError("empty id must not render")


def test_render_pins_snapshot_id() -> None:
    render = _load("render_from_infra", "render-from-infra.py")
    sg_id = "sg-" + "abc123"
    text = render.n8n_sg_annotations({"n8n_cloudfront_sg_id": sg_id})
    assert f"alb.ingress.kubernetes.io/security-groups: {sg_id}" in text
    assert 'manage-backend-security-group-rules: "true"' in text
