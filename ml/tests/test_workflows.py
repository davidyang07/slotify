"""The committed GitHub Actions workflows parse, and declare the jobs CI needs.

A workflow file that does not parse fails in the quietest possible way. GitHub
creates a run named after the file *path* rather than the workflow, with zero
jobs, and marks it failed. Every job the file was supposed to contain stops
running, and nothing in the repository notices -- which is worse than a red
build, because a red build gets read.

An inline ``run:`` command is the usual way in: an unquoted YAML scalar cannot
contain ``": "``, so a `node -e` one-liner that formats a message with a colon
turns the rest of the file into a parse error. These tests are cheap and they
run in the same suite as everything else, so the breakage is caught before the
push rather than after it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from slotify_rank.config.settings import find_repo_root

WORKFLOW_DIR = find_repo_root() / ".github" / "workflows"


def _workflows() -> list[Path]:
    if not WORKFLOW_DIR.is_dir():
        return []
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", _workflows(), ids=lambda p: p.name)
def test_every_workflow_parses(path: Path):
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        pytest.fail(
            f"{path.name} is not valid YAML, so GitHub would run none of its "
            f"jobs and report a nameless failed run:\n{error}"
        )
    assert isinstance(document, dict), f"{path.name} is not a mapping"
    assert document.get("jobs"), f"{path.name} declares no jobs"


def test_the_workflows_this_repository_relies_on_are_present():
    """Named explicitly, so deleting one is a decision rather than an accident."""
    names = {path.name for path in _workflows()}
    assert {"ci.yml", "model-smoke.yml"} <= names


def test_ci_declares_a_job_for_every_layer():
    ci = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8"))
    jobs = set(ci["jobs"])
    # The product, the ranking package, the corpus registry and the frozen
    # baseline's cross-language parity. Dropping any one of them would leave a
    # whole layer unverified while CI still went green.
    assert {"frontend", "backend", "scripts", "ml", "corpus-registry", "parity"} <= jobs


def test_ci_runs_the_same_verification_a_developer_runs():
    """CI must not drift into its own private list of steps.

    `npm run verify:ml` is the one command that knows what to check; if CI
    reimplemented it, the two would disagree the first time a check was added.
    """
    text = (WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/verify-ml.mjs" in text
    assert "scripts/check-hygiene.mjs" in text
