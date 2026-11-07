"""Sibling clones are skipped locally and required when CI asks for them."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check-platform-config.py"
WORKFLOW = ROOT / ".github" / "workflows" / "ai-platform-ci.yml"


def _checker():
    spec = importlib.util.spec_from_file_location("check_platform_config", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run(siblings_dir: Path, require: bool) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PLATFORM_CONFIG_SIBLINGS_DIR"] = str(siblings_dir)
    env["PLATFORM_CONFIG_SKIP_GITHUB"] = "1"
    env.pop("PLATFORM_CONFIG_REQUIRE_SIBLINGS", None)
    if require:
        env["PLATFORM_CONFIG_REQUIRE_SIBLINGS"] = "1"
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sibling_list_is_every_github_repo_except_this_one() -> None:
    checker = _checker()
    repos = checker.sibling_repos()
    assert repos == [name for name in checker.PLATFORM_REPOS if name != "ai-platform"]
    assert "athena-labs" in repos
    assert "cicd" in repos
    assert "infrastructure" in repos
    assert "mcp-server" in repos
    assert "rag-service" in repos
    assert "ai-platform" not in repos


def test_github_repo_set_must_match() -> None:
    checker = _checker()
    expected = set(checker.PLATFORM_REPOS)
    assert checker.repo_set_errors(expected, expected) == []
    extra = checker.repo_set_errors(expected, expected | {"microservices"})
    assert any("microservices" in item for item in extra)
    missing = checker.repo_set_errors(expected, expected - {"athena-labs"})
    assert any("athena-labs" in item for item in missing)
    covered = {(repo, rel) for repo, rel, _kind in checker.CHECKS}
    assert ("rag-service", "agent-spec.yaml") in covered
    assert ("rag-service", "k8s/helm/values.yaml") in covered
    assert ("rag-service", ".env.example") in covered


def test_missing_sibling_skips_by_default(tmp_path: Path) -> None:
    proc = _run(tmp_path, require=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "skip siblings (clone not present):" in proc.stdout
    assert "required sibling repo not present" not in proc.stdout


def test_missing_sibling_fails_when_required(tmp_path: Path) -> None:
    checker = _checker()
    proc = _run(tmp_path, require=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    for repo in checker.sibling_repos():
        assert f"required sibling repo not present: {tmp_path / repo}" in proc.stdout
    assert "skip siblings" not in proc.stdout


def test_present_sibling_dirs_are_not_reported_missing(tmp_path: Path) -> None:
    checker = _checker()
    for repo in checker.sibling_repos():
        (tmp_path / repo).mkdir()
    proc = _run(tmp_path, require=True)
    assert "required sibling repo not present" not in proc.stdout
    assert "skip siblings" not in proc.stdout
    # Empty clones still fail on their missing files, as before.
    assert proc.returncode == 1
    assert "missing:" in proc.stdout


def test_ci_checks_out_exactly_the_siblings_the_script_reads() -> None:
    checker = _checker()
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = doc["jobs"]["platform-config-siblings"]
    assert job["permissions"] == {"contents": "read"}
    checkouts = [step for step in job["steps"] if str(step.get("uses", "")).startswith("actions/checkout@")]
    assert checkouts, "job must check out repos"
    for step in checkouts:
        assert step["with"]["persist-credentials"] is False
    siblings = [step for step in checkouts if "repository" in step["with"]]
    owner_repo = {step["with"]["repository"] for step in siblings}
    assert owner_repo == {f"${{{{ github.repository_owner }}}}/{repo}" for repo in checker.sibling_repos()}
    for step in siblings:
        assert step["with"]["token"] == "${{ secrets.SIBLING_REPOS_READ_TOKEN }}"
        assert step["with"]["path"].startswith("siblings/")
    run_steps = [step for step in job["steps"] if "run" in step]
    check_step = next(step for step in run_steps if "check-platform-config.py" in step["run"])
    assert check_step["env"]["PLATFORM_CONFIG_REQUIRE_SIBLINGS"] == "1"
    assert check_step["env"]["PLATFORM_CONFIG_SIBLINGS_DIR"] == "${{ github.workspace }}/siblings"
    blob = yaml.dump(job)
    assert "set -x" not in blob
    # The raw secret value flows only into checkout `token:` inputs.
    # Run steps may test presence (`!= ''`) but never receive the value.
    for step in run_steps:
        for value in (step.get("env") or {}).values():
            if "SIBLING_REPOS_READ_TOKEN" in str(value):
                assert "!= ''" in str(value)
        assert "secrets.SIBLING_REPOS_READ_TOKEN" not in step["run"]
