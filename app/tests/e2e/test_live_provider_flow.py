"""Hosted live-provider E2E coverage for the release gate.

The test is opt-in because it sends synthetic clinical text to the configured
provider and requires a real repository secret. The workflow gives it a fresh
SQLite database, so its temporary key, catalog rows, and session are isolated
from developer data.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from playwright.sync_api import APIRequestContext, Page
from repositories.context import RepositoryContext
from repositories.drug_catalog_repository import DrugCatalogRepository
from repositories.knowledge_repository import KnowledgeRepository

LIVE_PROVIDER_ENABLED = os.getenv("DILIGENT_LIVE_PROVIDER_E2E", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
PROVIDER = "opencode_go"
ACCESS_KEY_PROVIDER = "opencode"
MODEL = "deepseek-v4-flash"
REVISION_INSTRUCTION = (
    "Read the session context once, then append exactly this sentence to the "
    "revised report: Synthetic release-gate revision: human review is required. "
    "Preserve all original clinical findings and do not invent any new clinical facts."
)
REVISION_APPEND_SENTENCE = "Synthetic release-gate revision: human review is required."

pytestmark = pytest.mark.skipif(
    not LIVE_PROVIDER_ENABLED,
    reason="Set DILIGENT_LIVE_PROVIDER_E2E=1 to run the live provider gate.",
)


###############################################################################
def _seed_release_gate_catalogs() -> None:
    """Provide the minimum public catalog evidence required by clinical preflight."""

    context = RepositoryContext.create()
    DrugCatalogRepository(context).upsert_drugs_catalog_records(
        [
            {
                "rxcui": "723",
                "raw_name": "amoxicillin",
                "term_type": "IN",
                "name": "amoxicillin",
                "brand_names": "Amoxil",
                "synonyms": "amoxicillin",
            }
        ]
    )
    KnowledgeRepository(context).save_livertox_records(
        pd.DataFrame(
            [
                {
                    "drug_name": "amoxicillin",
                    "nbk_id": "NBK548517",
                    "synonyms": "amoxicillin",
                    "excerpt": "Synthetic release-gate catalog evidence for amoxicillin.",
                    "include_in_livertox": True,
                    "source_url": "https://www.ncbi.nlm.nih.gov/books/NBK548517/",
                }
            ]
        )
    )


###############################################################################
def _clinical_input() -> str:
    return (
        "## Anamnesis\n"
        "Synthetic release-gate subject reports persistent fatigue, nausea, and "
        "mild right upper quadrant discomfort. There is no reported viral illness, "
        "alcohol exposure, or chronic liver disease in this fictional record. "
        "The subject is an adult and has no documented prior reaction to this drug.\n\n"
        "## Therapy\n"
        "Amoxicillin 500 mg twice daily was started on 2026-09-01 and stopped on "
        "2026-09-08 after symptoms developed. No other new medicine is reported.\n\n"
        "## Laboratory history\n"
        "On 2026-09-08 ALT was 210 U/L with ULN 40 U/L, AST was 170 U/L, ALP was "
        "130 U/L with ULN 120 U/L, and total bilirubin was 2.1 mg/dL."
    )


###############################################################################
def _wait_for_job(
    api_context: APIRequestContext,
    job_id: str,
    *,
    status_path: str = "/api/clinical/jobs",
    label: str = "Live provider clinical",
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = api_context.get(f"{status_path}/{job_id}?_={time.monotonic_ns()}")
        assert response.status == 200, response.text()
        last_payload = response.json()
        if last_payload.get("status") in {"completed", "failed", "cancelled"}:
            return last_payload
        time.sleep(2)
    pytest.fail(
        f"{label} job did not reach a terminal state within "
        f"{timeout_seconds}s: status={last_payload.get('status')}"
    )


###############################################################################
def _assert_secret_absent(value: Any, provider_key: str, *, scope: str) -> None:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    assert provider_key not in serialized, f"Provider credential leaked in {scope}."


###############################################################################
def _assert_live_provider_logs_are_sanitized(provider_key: str) -> None:
    log_dir_value = os.getenv("DILIGENT_LIVE_PROVIDER_LOG_DIR", "").strip()
    if not log_dir_value:
        return
    log_dir = Path(log_dir_value)
    log_paths = sorted(log_dir.glob("*.log"))
    assert log_paths, f"No live-provider diagnostics were found in {log_dir}."
    for log_path in log_paths:
        try:
            content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            pytest.fail(f"Could not inspect live-provider log {log_path.name}: {exc}")
        assert provider_key not in content, (
            f"Provider credential leaked in live-provider log {log_path.name}."
        )


###############################################################################
def _read_revision_attempt(
    api_context: APIRequestContext,
    *,
    job_id: str,
    pipeline_run_id: str | None = None,
    revision_version_id: int | None = None,
    source_session_id: int,
    provider_key: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    terminal_job = _wait_for_job(
        api_context,
        job_id,
        status_path="/api/inspection/sessions/revision/jobs",
        label="Live provider revision",
        timeout_seconds=timeout_seconds,
    )
    _assert_secret_absent(terminal_job, provider_key, scope="revision job response")
    assert terminal_job["job_id"] == job_id
    assert terminal_job["job_type"] == "session_revision"
    assert terminal_job["status"] in {"completed", "failed", "cancelled"}
    result = terminal_job.get("result")
    assert isinstance(result, dict)
    resolved_pipeline_run_id = pipeline_run_id or result.get("pipeline_run_id")
    resolved_revision_version_id = revision_version_id or result.get(
        "revision_version_id"
    )
    assert isinstance(resolved_pipeline_run_id, str) and resolved_pipeline_run_id
    assert isinstance(resolved_revision_version_id, int)

    run_response = api_context.get(
        f"/api/inspection/sessions/revision/pipeline-runs/{resolved_pipeline_run_id}"
    )
    assert run_response.status == 200, run_response.text()
    run = run_response.json()
    _assert_secret_absent(run, provider_key, scope="revision run response")
    assert run["pipeline_run_id"] == resolved_pipeline_run_id
    assert run["session_id"] == source_session_id
    assert run["target_revision_version_id"] == resolved_revision_version_id
    assert run["configuration"]["pipeline_run_id"] == resolved_pipeline_run_id
    assert run["configuration"]["source_session_id"] == source_session_id
    assert run["configuration"]["model_provider"] == PROVIDER
    assert run["configuration"]["model_name"] == MODEL
    assert run["configuration"]["provider_max_retries"] == 2
    assert run["configuration"].get("metadata", {}).get("live_provider_e2e") is True

    steps_response = api_context.get(
        f"/api/inspection/sessions/revision/pipeline-runs/"
        f"{resolved_pipeline_run_id}/steps"
    )
    assert steps_response.status == 200, steps_response.text()
    steps_payload = steps_response.json()
    _assert_secret_absent(steps_payload, provider_key, scope="revision step response")
    steps = steps_payload.get("items")
    assert isinstance(steps, list)
    assert steps
    for step in steps:
        assert step["pipeline_run_id"] == resolved_pipeline_run_id
        assert step["model_provider"] == PROVIDER
        assert step["model_name"] == MODEL
        assert step["status"] in {"completed", "failed", "cancelled"}
        assert step["latency_ms"] is not None

    version_response = api_context.get(
        f"/api/inspection/sessions/{source_session_id}/versions/"
        f"{resolved_revision_version_id}"
    )
    assert version_response.status == 200, version_response.text()
    version_detail = version_response.json()
    _assert_secret_absent(
        version_detail, provider_key, scope="revision version response"
    )
    version = version_detail["version"]
    assert version["revision_version_id"] == resolved_revision_version_id
    assert version["source_version_id"] == run["source_version_id"]
    assert version["pipeline_run_id"] == resolved_pipeline_run_id
    assert version["model_configuration"]["model_provider"] == PROVIDER
    assert version["model_configuration"]["model_name"] == MODEL

    artifacts_response = api_context.get(
        f"/api/inspection/sessions/{source_session_id}/versions/"
        f"{resolved_revision_version_id}/artifacts"
    )
    assert artifacts_response.status == 200, artifacts_response.text()
    artifacts_payload = artifacts_response.json()
    _assert_secret_absent(artifacts_payload, provider_key, scope="revision artifacts")
    artifacts = artifacts_payload.get("items")
    assert isinstance(artifacts, list)
    return {
        "job": terminal_job,
        "run": run,
        "steps": steps,
        "version": version,
        "artifacts": artifacts,
        "pipeline_run_id": resolved_pipeline_run_id,
        "revision_version_id": resolved_revision_version_id,
    }


###############################################################################
def _wait_for_browser_job_submission(
    page: Page,
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def capture(response: Any) -> None:
        if response.request.method == "POST" and response.url.rstrip("/").endswith(
            "/api/clinical/jobs"
        ):
            captured["response"] = response

    page.on("response", capture)
    try:
        page.get_by_role("button", name="Run DILI analysis").click()
        deadline = time.monotonic() + timeout_seconds
        dialog = page.get_by_role("dialog")
        while "response" not in captured and time.monotonic() < deadline:
            if dialog.is_visible():
                continue_button = dialog.get_by_role(
                    "button", name="Continue with limitations"
                )
                if not continue_button.is_visible():
                    pytest.fail(
                        "Live provider browser preflight blocked the synthetic case: "
                        + dialog.inner_text()
                    )
                continue_button.click()
            page.wait_for_timeout(250)
        if "response" not in captured:
            pytest.fail("Browser did not submit the live provider clinical job.")
        response = captured["response"]
        assert response.status == 202, response.text()
        return response.json()
    finally:
        page.remove_listener("response", capture)


###############################################################################
def test_live_opencode_go_provider_and_browser_clinical_flow(
    page: Page,
    base_url: str,
    api_context: APIRequestContext,
) -> None:
    provider_key = os.getenv("OPENCODE_GO_API_KEY", "").strip()
    if not provider_key:
        pytest.fail(
            "DILIGENT_LIVE_PROVIDER_E2E=1 requires the OPENCODE_GO_API_KEY secret."
        )

    _seed_release_gate_catalogs()
    created_key_id: int | None = None
    job_id: str | None = None
    session_id: int | None = None
    revision_job_id: str | None = None
    revision_pipeline_run_id: str | None = None

    try:
        create_key = api_context.post(
            "/api/access-keys",
            data={"provider": ACCESS_KEY_PROVIDER, "access_key": provider_key},
        )
        assert create_key.status == 201, create_key.text()
        created_key = create_key.json()
        _assert_secret_absent(created_key, provider_key, scope="access-key response")
        created_key_id = created_key.get("id")
        assert isinstance(created_key_id, int)
        assert "access_key" not in created_key

        activate_key = api_context.put(
            f"/api/access-keys/{created_key_id}/activate?provider={ACCESS_KEY_PROVIDER}"
        )
        assert activate_key.status == 200, activate_key.text()
        _assert_secret_absent(
            activate_key.json(), provider_key, scope="access-key activation response"
        )

        config = api_context.put(
            "/api/model-config",
            data={
                "use_cloud_services": True,
                "llm_provider": PROVIDER,
                "cloud_model": MODEL,
                "text_extraction_model": MODEL,
                "clinical_model": MODEL,
                "revision_model": MODEL,
                "timeline_model": MODEL,
                "reasoning_level": "off",
            },
        )
        assert config.status == 200, config.text()
        config_payload = config.json()
        _assert_secret_absent(
            config_payload, provider_key, scope="model config response"
        )
        assert config_payload["llm_provider"] == PROVIDER
        assert config_payload["cloud_model"] == MODEL
        assert config_payload["clinical_model"] == MODEL
        assert config_payload["revision_model"] == MODEL

        connectivity = api_context.post(
            "/api/model-config/connectivity-check",
            data={"provider": PROVIDER, "model": MODEL},
        )
        assert connectivity.status == 200, connectivity.text()
        connectivity_payload = connectivity.json()
        _assert_secret_absent(
            connectivity_payload, provider_key, scope="connectivity response"
        )
        assert connectivity_payload == {
            "provider": PROVIDER,
            "model": MODEL,
            "ok": True,
            "response_preview": connectivity_payload.get("response_preview"),
            "error": None,
        }
        assert connectivity_payload["response_preview"]

        page.goto(base_url)
        page.get_by_label("Clinical Input").fill(_clinical_input())
        page.get_by_label("Patient Name").fill("Synthetic Release Gate Subject")
        page.get_by_label("Visit Date").fill("2026-09-08")
        start_payload = _wait_for_browser_job_submission(page)
        job_id = start_payload.get("job_id")
        assert isinstance(job_id, str) and job_id
        assert start_payload.get("job_type") == "clinical"

        final_payload = _wait_for_job(api_context, job_id)
        _assert_secret_absent(
            final_payload, provider_key, scope="clinical job response"
        )
        assert final_payload.get("status") == "completed", final_payload.get("error")
        result = final_payload.get("result")
        assert isinstance(result, dict)
        assert isinstance(result.get("session_id"), int)
        session_id = result["session_id"]
        assert result.get("report") or result.get("final_report")

        page.wait_for_function(
            """() => {
                const report = document.querySelector('#rendered-report-content');
                return Boolean(report && report.textContent?.trim().length);
            }""",
            timeout=120000,
        )
        report_text = page.locator("#rendered-report-content").inner_text()
        assert report_text.strip()

        session_response = api_context.get(f"/api/inspection/sessions/{session_id}")
        assert session_response.status == 200, session_response.text()
        source_session_snapshot = session_response.json()
        assert source_session_snapshot.get("session_id") == session_id
        _assert_secret_absent(
            source_session_snapshot, provider_key, scope="source session response"
        )
        source_versions_response = api_context.get(
            f"/api/inspection/sessions/{session_id}/versions"
        )
        assert source_versions_response.status == 200, source_versions_response.text()
        source_versions_payload = source_versions_response.json()
        _assert_secret_absent(
            source_versions_payload, provider_key, scope="source version response"
        )
        source_versions = source_versions_payload.get("items")
        assert isinstance(source_versions, list) and source_versions
        source_version = source_versions[-1]
        source_version_id = source_version["version_id"]
        root_session_id = source_version["root_session_id"]

        revision_start = api_context.post(
            f"/api/inspection/sessions/{session_id}/revision/jobs",
            data={
                "revision_instruction": REVISION_INSTRUCTION,
                "metadata": {"live_provider_e2e": True, "dry_run": False},
                "max_tasks": 1,
                "max_tool_iterations": 2,
                "allowed_tools": ["read_session_context"],
                "revision_goal": "full_report_revision",
                "dry_run": False,
            },
        )
        assert revision_start.status == 202, revision_start.text()
        revision_start_payload = revision_start.json()
        _assert_secret_absent(
            revision_start_payload, provider_key, scope="revision start response"
        )
        revision_job_id = revision_start_payload.get("job_id")
        assert isinstance(revision_job_id, str) and revision_job_id

        attempts = [
            _read_revision_attempt(
                api_context,
                job_id=revision_job_id,
                source_session_id=session_id,
                provider_key=provider_key,
                timeout_seconds=1800,
            )
        ]
        first_attempt = attempts[0]
        revision_pipeline_run_id = first_attempt["pipeline_run_id"]
        if first_attempt["job"]["status"] == "failed":
            first_error = first_attempt["run"].get("error")
            assert isinstance(first_error, dict)
            _assert_secret_absent(first_error, provider_key, scope="revision failure")
            assert first_error.get("provider") == PROVIDER
            assert first_error.get("model") == MODEL
            assert first_error.get("retryable") is True

            retry_response = api_context.post(
                f"/api/inspection/sessions/revision/pipeline-runs/"
                f"{revision_pipeline_run_id}/retry"
            )
            assert retry_response.status == 202, retry_response.text()
            retry_payload = retry_response.json()
            _assert_secret_absent(
                retry_payload, provider_key, scope="revision retry response"
            )
            retry_job_id = retry_payload.get("job_id")
            assert isinstance(retry_job_id, str) and retry_job_id != revision_job_id
            revision_job_id = retry_job_id
            attempts.append(
                _read_revision_attempt(
                    api_context,
                    job_id=retry_job_id,
                    source_session_id=session_id,
                    provider_key=provider_key,
                    timeout_seconds=1800,
                )
            )
            retry_attempt = attempts[-1]
            retry_pipeline_run_id = retry_attempt["pipeline_run_id"]
            assert retry_pipeline_run_id != revision_pipeline_run_id
            revision_pipeline_run_id = retry_pipeline_run_id
            original_run_reload = api_context.get(
                f"/api/inspection/sessions/revision/pipeline-runs/"
                f"{first_attempt['pipeline_run_id']}"
            )
            assert original_run_reload.status == 200, original_run_reload.text()
            original_run = original_run_reload.json()
            _assert_secret_absent(
                original_run, provider_key, scope="reloaded failed revision run"
            )
            assert original_run["status"] == "failed"
            assert original_run["pipeline_run_id"] == first_attempt["pipeline_run_id"]
        elif first_attempt["job"]["status"] != "completed":
            pytest.fail(
                "Live revision reached an unexpected terminal state: "
                f"{first_attempt['job']['status']}"
            )

        final_attempt = attempts[-1]
        final_job = final_attempt["job"]
        final_run = final_attempt["run"]
        final_version = final_attempt["version"]
        final_artifacts = final_attempt["artifacts"]
        assert final_job["status"] == "completed", final_run.get("error")
        assert final_run["status"] == "completed"
        assert final_job["result"].get("revision_status") != "dry_run"
        assert final_run["revision_mode"] == "agentic_revision"
        assert final_run["revision_kind"] == "llm_assisted_revision"
        assert final_run["root_session_id"] == root_session_id
        assert final_run["source_version_id"] == source_version_id
        assert final_run["completed_at"] is not None

        final_steps = final_attempt["steps"]
        assert all(step["status"] == "completed" for step in final_steps)
        assert {step["step_name"] for step in final_steps} >= {
            "revision_agent_planner",
            "revision_agent_editor",
            "revision_agent_qa",
        }
        task_steps = [
            step
            for step in final_steps
            if step["step_name"].startswith("revision_agent_task_")
        ]
        assert task_steps
        assert final_job["result"].get("tool_call_count", 0) >= 1

        artifact_by_key = {
            artifact["artifact_key"]: artifact for artifact in final_artifacts
        }
        assert {
            "revision_agent_context",
            "revision_agent_plan",
            "revision_agent_tool_trace",
        } <= artifact_by_key.keys()
        draft_artifacts = [
            artifact
            for artifact in final_artifacts
            if artifact["artifact_key"]
            in {"revision_agent_draft_report", "revision_agent_draft_report_repair"}
        ]
        qa_artifacts = [
            artifact
            for artifact in final_artifacts
            if artifact["artifact_key"]
            in {"revision_agent_qa", "revision_agent_qa_repair"}
        ]
        assert draft_artifacts
        assert qa_artifacts
        tool_trace = artifact_by_key["revision_agent_tool_trace"]["payload"]
        assert isinstance(tool_trace, dict)
        observations = tool_trace.get("observations")
        assert isinstance(observations, list)
        assert any(
            observation.get("tool") == "read_session_context"
            for observation in observations
            if isinstance(observation, dict)
        )
        draft_payload = draft_artifacts[-1]["payload"]
        assert isinstance(draft_payload, dict)
        assert draft_payload["patches"]
        assert REVISION_APPEND_SENTENCE in draft_payload["revised_report_text"]
        qa_artifact = qa_artifacts[-1]
        assert qa_artifact["status"] in {"passed", "qa_failed"}
        qa_payload = qa_artifact["payload"]
        assert isinstance(qa_payload, dict)
        assert isinstance(qa_payload.get("blocking_issues"), list)
        assert isinstance(qa_payload.get("warnings"), list)
        assert qa_artifact["status"] == (
            "qa_failed" if qa_payload["blocking_issues"] else "passed"
        )
        assert final_version["version_status"] in {
            "llm_qa_passed",
            "qa_failed",
            "requires_human_review",
        }
        assert final_version["llm_qa_status"] in {
            "passed",
            "failed",
            "requires_human_review",
        }
        assert final_version["completed_at"] is not None

        source_after_response = api_context.get(
            f"/api/inspection/sessions/{session_id}"
        )
        assert source_after_response.status == 200, source_after_response.text()
        source_after = source_after_response.json()
        _assert_secret_absent(
            source_after, provider_key, scope="reloaded source session"
        )
        assert source_after == source_session_snapshot

        if final_version["version_status"] == "llm_qa_passed":
            child_session_id = final_version.get("session_id")
            assert isinstance(child_session_id, int) and child_session_id != session_id
            child_response = api_context.get(
                f"/api/inspection/sessions/{child_session_id}"
            )
            assert child_response.status == 200, child_response.text()
            child = child_response.json()
            _assert_secret_absent(child, provider_key, scope="accepted child session")
            child_payload = child.get("result_payload")
            assert isinstance(child_payload, dict)
            child_revision = child_payload.get("revision")
            assert isinstance(child_revision, dict)
            assert child_revision.get("pipeline_run_id") == final_run["pipeline_run_id"]
            child_reload = api_context.get(
                f"/api/inspection/sessions/{child_session_id}"
            )
            assert child_reload.status == 200, child_reload.text()
            assert child_reload.json() == child
            child_versions_response = api_context.get(
                f"/api/inspection/sessions/{child_session_id}/versions"
            )
            assert child_versions_response.status == 200, child_versions_response.text()
            child_versions = child_versions_response.json().get("items")
            assert any(
                item["version_id"] == final_version["version_id"]
                and item["source_version_id"] == source_version_id
                and item["root_session_id"] == root_session_id
                for item in child_versions
            )
        else:
            assert final_version.get("session_id") is None

        source_versions_after_response = api_context.get(
            f"/api/inspection/sessions/{session_id}/versions"
        )
        assert source_versions_after_response.status == 200, (
            source_versions_after_response.text()
        )
        source_versions_after = source_versions_after_response.json().get("items")
        assert any(
            item["version_id"] == final_version["version_id"]
            and item["source_version_id"] == source_version_id
            and item["pipeline_run_id"] == final_run["pipeline_run_id"]
            for item in source_versions_after
        )
        _assert_live_provider_logs_are_sanitized(provider_key)
    finally:
        if job_id is not None and session_id is None:
            api_context.delete(f"/api/clinical/jobs/{job_id}")
        if revision_job_id is not None:
            revision_status = api_context.get(
                f"/api/inspection/sessions/revision/jobs/{revision_job_id}"
            )
            if revision_status.status == 200 and revision_status.json().get(
                "status"
            ) not in {
                "completed",
                "failed",
                "cancelled",
            }:
                api_context.delete(
                    f"/api/inspection/sessions/revision/jobs/{revision_job_id}"
                )
        if session_id is not None:
            api_context.delete(f"/api/inspection/sessions/{session_id}")
        if created_key_id is not None:
            api_context.delete(
                f"/api/access-keys/{created_key_id}?provider={ACCESS_KEY_PROVIDER}"
            )
