# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SHA = "a" * 40
REPO = "CTCycle/DILIGENT-Clinical-Copilot"
GATES = (
    "release-preflight",
    "release-persistence-preflight",
    "release-live-provider-e2e",
    "release-security-preflight",
)


def _responses() -> dict:
    base = f"repos/{REPO}"
    return {
        f"{base}/branches/main": {"commit": {"sha": SHA}},
        f"{base}/branches/develop": {"commit": {"sha": SHA}},
        f"{base}/git/matching-refs/tags/v3.4.0": [],
        f"{base}/releases?per_page=100": None,
        f"{base}/environments/desktop-release": {
            "can_admins_bypass": False,
            "protection_rules": [
                {"type": "required_reviewers", "reviewers": [{"id": 1}]}
            ],
        },
        f"{base}/actions/runs/123": {
            "path": ".github/workflows/release.yml",
            "event": "workflow_dispatch",
            "head_branch": "main",
            "head_sha": SHA,
            "status": "completed",
            "conclusion": "success",
            "run_attempt": 1,
            "html_url": "https://example.invalid/run/123",
        },
        f"{base}/actions/runs/123/jobs?filter=latest&per_page=100": {
            "jobs": [
                {"name": name, "conclusion": "success"} for name in (*GATES, "prepare")
            ],
        },
    }


def _invoke(
    tmp_path: Path, responses: dict, action: str = "Candidate"
) -> subprocess.CompletedProcess:
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("The release guard executes on Windows/PowerShell hosts.")
    fixtures = tmp_path / "responses.json"
    fixtures.write_text(
        json.dumps(
            {
                key: json.dumps(value) if value is not None else ""
                for key, value in responses.items()
            }
        ),
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "GITHUB_REPOSITORY": REPO,
        "GITHUB_SHA": SHA,
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_OUTPUT": str(tmp_path / "output"),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "OPERATOR_EVIDENCE": "Reviewed QA evidence",
        "WINDOWS_CODESIGN_CERT_THUMBPRINT": "B" * 40,
        "RELEASE_TEST_RESPONSES": str(fixtures),
        "RELEASE_TEST_DIRECTORY": str(tmp_path),
        "RELEASE_TEST_ACTION": action,
    }
    # No network or mutation is allowed: an unexpected endpoint/method fails.
    command = r"""
        $responses = Get-Content $env:RELEASE_TEST_RESPONSES -Raw | ConvertFrom-Json -AsHashtable
        function gh {
            if ($args -contains '--method') { throw 'Unexpected mutation' }
            $endpoint = @($args | Where-Object { $_ -like 'repos/*' })[0]
            if (-not $responses.ContainsKey($endpoint)) { throw "Unexpected endpoint: $endpoint" }
            $global:LASTEXITCODE = 0
            $responses[$endpoint]
        }
        & ./app/desktop/build/release_candidate.ps1 -Action $env:RELEASE_TEST_ACTION -Version 3.4.0 -CandidateRunId 123 -ArtifactDirectory $env:RELEASE_TEST_DIRECTORY
    """
    return subprocess.run(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    "failure",
    [
        "stale_source",
        "existing_tag",
        "wrong_run",
        "skipped_provider",
        "unprotected_environment",
    ],
)
def test_release_rejects_incomplete_or_unrelated_evidence(
    tmp_path: Path, failure: str
) -> None:
    responses = _responses()
    base = f"repos/{REPO}"
    expected = ""
    if failure == "stale_source":
        responses[f"{base}/branches/develop"]["commit"]["sha"] = "c" * 40
        expected = "not synchronized"
    elif failure == "existing_tag":
        responses[f"{base}/git/matching-refs/tags/v3.4.0"] = [
            {"ref": "refs/tags/v3.4.0"}
        ]
        expected = "already exists"
    elif failure == "wrong_run":
        responses[f"{base}/actions/runs/123"]["head_sha"] = "c" * 40
        expected = "exact main SHA"
    elif failure == "skipped_provider":
        responses[f"{base}/actions/runs/123/jobs?filter=latest&per_page=100"]["jobs"][
            2
        ]["conclusion"] = "skipped"
        expected = "did not pass"
    else:
        responses[f"{base}/environments/desktop-release"]["protection_rules"] = []
        expected = "require reviewer approval"
    result = _invoke(tmp_path, responses)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not (tmp_path / "output").exists()


def test_release_rejects_modified_candidate_before_publication(tmp_path: Path) -> None:
    qa = tmp_path / "qa"
    qa.mkdir()
    smoke = qa / "smoke.json"
    smoke.write_text('{"success": true}', encoding="utf-8")
    executable = "DILIGENT-v3.4.0-windows-x64-portable.exe"
    (tmp_path / executable).write_bytes(b"tampered candidate")
    receipt = {
        "version": "3.4.0",
        "source_sha": SHA,
        "run_id": "123",
        "run_attempt": 1,
        "signer": "B" * 40,
        "gates": {gate: "success" for gate in GATES},
        "smoke_sha256": hashlib.sha256(smoke.read_bytes()).hexdigest(),
        "files": {executable: "0" * 64},
    }
    (tmp_path / "candidate.json").write_text(json.dumps(receipt), encoding="utf-8")
    result = _invoke(tmp_path, _responses(), "Publish")
    assert result.returncode != 0
    assert "Candidate digest mismatch" in result.stderr
    assert "Unexpected mutation" not in result.stderr


def test_complete_candidate_reaches_approval_without_publication(
    tmp_path: Path,
) -> None:
    result = _invoke(tmp_path, _responses())
    assert result.returncode == 0, result.stderr
    assert "attempt=1" in (tmp_path / "output").read_text(encoding="utf-8-sig")
    assert "Approval attests" in (tmp_path / "summary").read_text(encoding="utf-8-sig")
