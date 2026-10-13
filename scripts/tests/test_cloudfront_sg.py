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


def test_check_does_not_write_on_drift(tmp_path: Path, monkeypatch) -> None:
    render = _load("render_from_infra", "render-from-infra.py")
    path = tmp_path / "ingress.yaml"
    path.write_text("old\n", encoding="utf-8")
    monkeypatch.setattr(render, "ROOT", tmp_path)
    try:
        render.store_rendered(path, "new\n", True)
    except SystemExit as exc:
        assert str(exc) == "drift: ingress.yaml"
    else:
        raise AssertionError("drift must fail")
    assert path.read_text(encoding="utf-8") == "old\n"


def test_deploy_workflow_has_no_infra_literals() -> None:
    scanner = _load("check_workflow_literals", "check-workflow-literals.py")
    text = (ROOT / ".github" / "workflows" / "deploy-n8n.yml").read_text(encoding="utf-8")
    owners = scanner.owner_values(ROOT / "platform-infra.dev.json")
    hits = scanner.scan_text(text, owners)
    assert hits == [], hits
    assert "n8n/k8s/secrets.yaml" not in text
    assert "render-from-infra.py --check" in text
    assert "scripts/smoke-direct-alb.py" in text
    assert "n8n/k8s/base.yaml" in text
    assert "n8n/k8s/ingress.yaml" in text
    assert "branches: [main]" in text
    assert text.count("github.ref == 'refs/heads/main'") == 2
    assert "kubectl apply -n n8n -f n8n/k8s/base.yaml" in text
    assert "kubectl apply -n n8n -f n8n/k8s/ingress.yaml" in text
    assert '"$EKS_CLUSTER_NAME"' in text
    assert '"$AWS_REGION"' in text
    assert "${{ env.EKS_CLUSTER_NAME }}" not in text
    nodes = text.split("- name: Count Ready nodes", 1)[1].split("- name: Dry-run apply", 1)[0]
    assert "set -euo pipefail" in nodes
    smoke = (ROOT / ".github" / "workflows" / "n8n-direct-alb-smoke.yml").read_text(encoding="utf-8")
    assert "branches: [main]" in smoke


def test_render_check_matches_committed_files() -> None:
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "render-from-infra.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK: render --check" in proc.stdout
    assert "drift:" not in proc.stdout
