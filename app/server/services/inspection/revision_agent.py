from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Callable

from configurations.startup import get_server_settings
from common.utils.clinical_safety import (
    RECHALLENGE_RECOMMENDATION_MESSAGE,
    contains_rechallenge_recommendation,
)
from domain.inspection import (
    RevisionAgentPlan,
    RevisionAgentQaResult,
    RevisionAgentToolCall,
    RevisionDraftResult,
    RevisionReportPatch,
    RevisionIssueScanResult,
    SessionRevisionRequest,
)
from services.inspection.revision_clinical_safety import audit_revised_dili_report
from services.inspection.revision_context import build_revision_context
from services.inspection.revision_patches import validate_draft_report
from common.prompts.revision_agent import (
    EDITOR_PROMPT_VERSION,
    PLANNER_PROMPT_VERSION,
    QA_PROMPT_VERSION,
    REPAIR_PROMPT_VERSION,
    REVISION_AGENT_SYSTEM_PROMPT,
    build_revision_issue_scan_user_prompt,
    editor_prompt,
    planner_prompt,
    qa_prompt,
    repair_editor_prompt,
    tool_prompt,
    TOOL_PROMPT_VERSION,
)
from services.inspection.revision_tools import RevisionToolRegistry
from services.llm.provider_factory import select_llm_provider
from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.generation_policy import GenerationPurpose
from services.llm.model_capabilities import resolve_model_capabilities
from services.llm.tool_loop import ToolLoopExecutor, ToolLoopResult
from domain.llm.transports import ChatMessage
from repositories.clinical_session_repository import ClinicalSessionRepository
from repositories.knowledge_repository import KnowledgeRepository
from repositories.session_revision_repository import SessionRevisionRepository

REVISION_AGENT_PROMPT_VERSION = "revision-agent-issue-scan-v1"
REVISION_AGENT_SCHEMA_NAME = "revision_issue_scan_result"
REVISION_AGENT_SCHEMA_VERSION = "1"
REVISION_AGENT_STEP_NAME = "revision_agent_issue_scan"
REVISION_PROVIDER_MAX_RETRIES = 1
MAX_QA_REPAIR_ATTEMPTS = 1


MAX_TEXT_CHARS = 30000
MAX_REPORT_CHARS = 20000
MAX_JSON_CHARS = 30000

StructuredCall = Callable[..., Any]
StopCheck = Callable[[], bool]
ProgressUpdate = Callable[[dict[str, Any]], None]

###############################################################################
@dataclass(frozen=True)
class RevisionAgentRuntime:
    provider: str
    model: str

###############################################################################
class RevisionAgentCancelled(RuntimeError):
    """Raised when a revision job is cancelled at a cooperative checkpoint."""

###############################################################################
def _clip_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n[TRUNCATED: {len(text) - limit} characters omitted]"

###############################################################################
def _safe_json(value: Any, limit: int = MAX_JSON_CHARS) -> str:
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        serialized = json.dumps(str(value), ensure_ascii=False)
    return _clip_text(serialized, limit)

###############################################################################
def _requested_append_sentence(instruction: str | None) -> str | None:
    text = str(instruction or "").strip()
    match = re.match(
        r"^Append exactly this sentence to the revised report:\s*"
        r"(?P<sentence>.+?)(?=\s+(?:Preserve|Keep|Do not|Ensure|Maintain)\b|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    sentence = match.group("sentence").strip()
    return sentence if sentence and sentence[-1] in ".!?" else None

###############################################################################
def _revision_provider_timeout(provider: str) -> float:
    runtime = get_server_settings().runtime
    cap = (
        runtime.local_llm_timeout_cap
        if provider.strip().lower() == "ollama"
        else runtime.cloud_llm_timeout_cap
    )
    return max(
        float(runtime.minimum_llm_timeout),
        min(float(runtime.default_llm_timeout), float(cap)),
    )

###############################################################################
def resolve_revision_agent_runtime() -> RevisionAgentRuntime:
    provider, model = LLMRuntimeConfig.resolve_provider_and_model("revision")
    return RevisionAgentRuntime(
        provider=provider,
        model=model,
    )

###############################################################################
def build_revision_agent_user_prompt(
    *,
    session: dict[str, Any],
    request: SessionRevisionRequest,
) -> str:
    sections = (
        session.get("sections") if isinstance(session.get("sections"), dict) else {}
    )
    result_payload = (
        session.get("result_payload")
        if isinstance(session.get("result_payload"), dict)
        else {}
    )
    packet = {
        "session_identity": {
            "session_id": session.get("session_id"),
            "patient_name": session.get("patient_name"),
            "visit_date": session.get("visit_date"),
            "session_timestamp": session.get("session_timestamp"),
            "version": session.get("version"),
            "status": session.get("status"),
        },
        "model_context": {
            "text_extraction_model": session.get("text_extraction_model"),
            "clinical_model": session.get("clinical_model"),
        },
        "original_clinical_input": {
            "source_clinical_text": _clip_text(
                session.get("source_clinical_text") or session.get("session_text"),
                MAX_TEXT_CHARS,
            ),
            "sections": sections,
        },
        "generated_output_under_review": {
            "report": _clip_text(
                session.get("official_report_text") or session.get("report"),
                MAX_REPORT_CHARS,
            ),
            "result_payload_json": _safe_json(result_payload, MAX_JSON_CHARS),
        },
        "user_revision_context": {
            "selected_text": _clip_text(request.selected_text, MAX_TEXT_CHARS),
            "revision_instruction": _clip_text(
                request.revision_instruction,
                4000,
            ),
            "metadata": request.metadata,
        },
        "tool_manifest_status": {
            "available": False,
            "instruction": "No tools are available in this implementation slice. Return proposed tool intents only.",
        },
    }
    return build_revision_issue_scan_user_prompt(
        packet_json=_safe_json(
            packet,
            MAX_TEXT_CHARS + MAX_REPORT_CHARS + MAX_JSON_CHARS,
        )
    )

###############################################################################
class RevisionAgentRunner:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        clinical_session_repository: ClinicalSessionRepository,
        session_revision_repository: SessionRevisionRepository,
        knowledge_repository: KnowledgeRepository,
        structured_call: StructuredCall | None = None,
    ) -> None:
        self.clinical_session_repository = clinical_session_repository
        self.session_revision_repository = session_revision_repository
        self.knowledge_repository = knowledge_repository
        self.structured_call = structured_call

    # -------------------------------------------------------------------------
    @staticmethod
    def _publish_phase(
        progress_update: ProgressUpdate | None,
        phase: str,
        *,
        repair_attempt: int = 0,
    ) -> None:
        if progress_update is None:
            return
        try:
            progress_update(
                {
                    "revision_phase": phase,
                    "repair_attempt": repair_attempt,
                }
            )
        except Exception:  # noqa: BLE001
            # Progress reporting must never turn a completed revision into a
            # failed revision when a job-status backend is unavailable.
            return

    # -------------------------------------------------------------------------
    def _call_revision_stage(
        self,
        *,
        runtime: RevisionAgentRuntime,
        pipeline_run_id: str,
        step_name: str,
        step_index: int,
        step_count: int,
        input_summary: dict[str, Any],
        prompt: str,
        schema: type[Any],
        purpose: GenerationPurpose,
        prompt_version: str,
        stop_check: StopCheck | None,
        output_summary: Callable[[Any], dict[str, Any]],
    ) -> Any:
        step = self.session_revision_repository.start_revision_step(
            pipeline_run_id=pipeline_run_id,
            step_name=step_name,
            step_index=step_index,
            step_count=max(1, step_count),
            input_summary=input_summary,
            schema_name=schema.__name__,
            schema_version="1",
            prompt_version=prompt_version,
            parser_version="structured-llm-v1",
            model_provider=runtime.provider,
            model_name=runtime.model,
        )
        attempt_number = int(step["attempt_number"])
        started = perf_counter()
        try:
            result = self._call_schema(
                runtime,
                prompt,
                schema,
                purpose=purpose,
            )
            self._raise_if_stopped(stop_check)
            latency_ms = int((perf_counter() - started) * 1000)
            self.session_revision_repository.complete_revision_step(
                pipeline_run_id=pipeline_run_id,
                step_name=step_name,
                attempt_number=attempt_number,
                status="completed",
                output_summary=output_summary(result),
                output_payload=result.model_dump(mode="json"),
                latency_ms=latency_ms,
                retry_count=max(0, attempt_number - 1),
            )
            return result
        except RevisionAgentCancelled as exc:
            latency_ms = int((perf_counter() - started) * 1000)
            self.session_revision_repository.fail_revision_step(
                pipeline_run_id=pipeline_run_id,
                step_name=step_name,
                attempt_number=attempt_number,
                status="cancelled",
                error={"message": str(exc)},
                latency_ms=latency_ms,
            )
            raise
        except Exception as exc:
            latency_ms = int((perf_counter() - started) * 1000)
            self.session_revision_repository.fail_revision_step(
                pipeline_run_id=pipeline_run_id,
                step_name=step_name,
                attempt_number=attempt_number,
                error={"message": str(exc)},
                latency_ms=latency_ms,
            )
            raise

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_draft(
        *,
        source_report: str,
        draft: RevisionDraftResult,
        request: SessionRevisionRequest,
    ) -> tuple[RevisionDraftResult, str]:
        try:
            applied_report = validate_draft_report(source_report, draft.patches)
        except ValueError as exc:
            draft = draft.model_copy(
                update={
                    "patches": [],
                    "revised_report_text": source_report,
                    "unresolved_issues": [
                        *draft.unresolved_issues,
                        f"The proposed patch could not be validated against the canonical report: {exc}",
                    ],
                    "human_review_requirements": [
                        *draft.human_review_requirements,
                        "Repair or manually review the invalid patch before accepting this revision.",
                    ],
                }
            )
            applied_report = source_report
        requested_append = _requested_append_sentence(request.revision_instruction)
        if requested_append and not applied_report.rstrip().endswith(requested_append):
            draft = draft.model_copy(
                update={
                    "patches": [
                        *draft.patches,
                        RevisionReportPatch(
                            start=len(source_report),
                            end=len(source_report),
                            replacement=f"\n\n{requested_append}",
                            expected_text="",
                            evidence_references=["user_revision_instruction"],
                        ),
                    ],
                    "changed_sections": [
                        *draft.changed_sections,
                        "user_requested_append",
                    ],
                }
            )
            try:
                applied_report = validate_draft_report(source_report, draft.patches)
            except ValueError as exc:  # pragma: no cover - deterministic append
                draft = draft.model_copy(
                    update={
                        "patches": [],
                        "revised_report_text": source_report,
                        "unresolved_issues": [
                            *draft.unresolved_issues,
                            f"The requested append could not be validated: {exc}",
                        ],
                    }
                )
                applied_report = source_report
        if not draft.revised_report_text:
            draft = draft.model_copy(update={"revised_report_text": applied_report})
        elif applied_report != draft.revised_report_text:
            draft = draft.model_copy(
                update={
                    "revised_report_text": applied_report,
                    "unresolved_issues": [
                        *draft.unresolved_issues,
                        "Model-provided revised report text differed from the deterministic patch output; the deterministic result is authoritative.",
                    ],
                    "human_review_requirements": [
                        *draft.human_review_requirements,
                        "Verify the deterministic patch result during clinical review.",
                    ],
                }
            )
        return draft, applied_report

    # -------------------------------------------------------------------------
    @staticmethod
    def _merge_quality_gates(
        *,
        qa: RevisionAgentQaResult,
        session: dict[str, Any],
        applied_report: str,
        source_report: str,
    ) -> RevisionAgentQaResult:
        blocking_issues = list(qa.blocking_issues)
        if applied_report == source_report:
            blocking_issues.append(
                "Revision produced no validated report edits; a new session cannot be created until a verified patch is produced."
            )
        clinical_safety_issues = audit_revised_dili_report(
            session=session,
            report_text=applied_report,
        )
        if contains_rechallenge_recommendation(applied_report) and (
            RECHALLENGE_RECOMMENDATION_MESSAGE not in clinical_safety_issues
        ):
            clinical_safety_issues.append(RECHALLENGE_RECOMMENDATION_MESSAGE)
        blocking_issues.extend(clinical_safety_issues)
        if not blocking_issues:
            return qa
        return qa.model_copy(
            update={
                "blocking_issues": list(dict.fromkeys(blocking_issues)),
                "manual_review_required": True,
            }
        )

    # -------------------------------------------------------------------------
    def run_issue_scan(
        self,
        *,
        job_id: str,
        pipeline_run_id: str,
        revision_version_id: int,
        source_version_id: int | None,
        session: dict[str, Any],
        request: SessionRevisionRequest,
        model_configuration: dict[str, Any],
    ) -> dict[str, Any]:
        del job_id
        runtime = resolve_revision_agent_runtime()
        user_prompt = build_revision_agent_user_prompt(
            session=session,
            request=request,
        )
        step = self.session_revision_repository.start_revision_step(
            pipeline_run_id=pipeline_run_id,
            step_name=REVISION_AGENT_STEP_NAME,
            step_index=1,
            step_count=1,
            input_summary={
                "session_id": session.get("session_id"),
                "selected_text_provided": bool(
                    str(request.selected_text or "").strip()
                ),
                "revision_instruction_provided": bool(
                    str(request.revision_instruction or "").strip()
                ),
                "report_provided": bool(
                    str(
                        session.get("official_report_text")
                        or session.get("report")
                        or ""
                    ).strip()
                ),
            },
            input_payload={
                "system_prompt": REVISION_AGENT_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
            },
            schema_name=REVISION_AGENT_SCHEMA_NAME,
            schema_version=REVISION_AGENT_SCHEMA_VERSION,
            prompt_version=REVISION_AGENT_PROMPT_VERSION,
            parser_version="structured-llm-v1",
            model_provider=runtime.provider,
            model_name=runtime.model,
        )
        attempt_number = int(step["attempt_number"])
        started = perf_counter()
        try:
            result = self._run_structured_scan(
                runtime=runtime,
                user_prompt=user_prompt,
            )
            payload = result.model_dump(mode="json")
            latency_ms = int((perf_counter() - started) * 1000)
            self.session_revision_repository.complete_revision_step(
                pipeline_run_id=pipeline_run_id,
                step_name=REVISION_AGENT_STEP_NAME,
                attempt_number=attempt_number,
                status="completed",
                output_summary={
                    "issue_count": len(result.issues),
                    "tool_intent_count": len(result.tool_intents)
                    + sum(len(issue.tool_intents) for issue in result.issues),
                },
                output_payload=payload,
                latency_ms=latency_ms,
            )
            artifact = self.session_revision_repository.persist_revision_artifact(
                pipeline_run_id=pipeline_run_id,
                revision_version_id=revision_version_id,
                artifact_key="revision_agent_issue_scan",
                status="requires_human_review",
                payload={
                    **payload,
                    "metadata": {
                        "prompt_version": REVISION_AGENT_PROMPT_VERSION,
                        "schema_name": REVISION_AGENT_SCHEMA_NAME,
                        "schema_version": REVISION_AGENT_SCHEMA_VERSION,
                        "model_provider": runtime.provider,
                        "model_name": runtime.model,
                        "source_version_id": source_version_id,
                    },
                },
            )
            completed_at = datetime.now(UTC)
            self.session_revision_repository.create_or_update_revision_run(
                pipeline_run_id=pipeline_run_id,
                session_id=int(session["session_id"]),
                root_session_id=int(model_configuration["root_session_id"]),
                source_version_id=int(model_configuration["source_version_id"]),
                target_revision_version_id=revision_version_id,
                revision_mode="agent_issue_scan",
                revision_kind="llm_assisted_revision",
                configuration=model_configuration,
                reviewer_note=request.revision_instruction,
                status="completed",
                completed_at=completed_at,
                latency_ms=latency_ms,
            )
            return {
                "pipeline_run_id": pipeline_run_id,
                "revision_version_id": revision_version_id,
                "step_name": REVISION_AGENT_STEP_NAME,
                "issue_scan": payload,
                "artifacts": artifact,
            }
        except Exception:
            latency_ms = int((perf_counter() - started) * 1000)
            error = {"message": "Revision agent issue scan failed."}
            self.session_revision_repository.fail_revision_step(
                pipeline_run_id=pipeline_run_id,
                step_name=REVISION_AGENT_STEP_NAME,
                attempt_number=attempt_number,
                error=error,
                latency_ms=latency_ms,
            )
            self.session_revision_repository.fail_revision_run(
                pipeline_run_id=pipeline_run_id,
                error=error,
            )
            raise

    # -------------------------------------------------------------------------
    def run_agentic(
        self,
        *,
        job_id: str,
        pipeline_run_id: str,
        revision_version_id: int,
        source_version_id: int | None,
        session: dict[str, Any],
        request: SessionRevisionRequest,
        model_configuration: dict[str, Any],
        stop_check: StopCheck | None = None,
        progress_update: ProgressUpdate | None = None,
    ) -> dict[str, Any]:
        runtime = resolve_revision_agent_runtime()
        self._raise_if_stopped(stop_check)
        self._publish_phase(progress_update, "planning")
        lineage = self.session_revision_repository.list_session_versions(
            int(session["session_id"])
        )
        context_effective = LLMRuntimeConfig.resolve_effective_inference_config(
            purpose=GenerationPurpose.REVISION_PLANNING,
            provider=runtime.provider,
            model=runtime.model,
        )
        context = build_revision_context(
            session=session,
            manual_edits=session.get("manual_edit_history") or [],
            lineage=lineage,
            selected_text=request.selected_text,
            instruction=request.revision_instruction,
            input_budget=context_effective.input_budget,
        )
        registry = RevisionToolRegistry(
            clinical_session_repository=self.clinical_session_repository,
            session_revision_repository=self.session_revision_repository,
            knowledge_repository=self.knowledge_repository,
            session=session,
            context=context,
        )
        manifest = registry.manifest(request.allowed_tools)
        native_tool_definitions = registry.tool_definitions(request.allowed_tools)
        capabilities = resolve_model_capabilities(
            provider=runtime.provider,
            model=runtime.model,
            descriptor=LLMRuntimeConfig.get_model_descriptor(
                runtime.provider, runtime.model
            ),
        )
        use_native_tools = (
            self.structured_call is None
            and bool(native_tool_definitions)
            and capabilities.supports_tools is True
            and capabilities.tool_call_mode == "native"
        )
        self.session_revision_repository.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_context",
            payload=context,
        )
        stage_count = max(3, request.max_tasks + 3)
        plan = self._call_revision_stage(
            runtime=runtime,
            pipeline_run_id=pipeline_run_id,
            step_name="revision_agent_planner",
            step_index=1,
            step_count=stage_count,
            input_summary={
                "manifest_count": len(manifest),
                "context_keys": sorted(context.keys()),
            },
            prompt=planner_prompt(context, manifest),
            schema=RevisionAgentPlan,
            purpose=GenerationPurpose.REVISION_PLANNING,
            prompt_version=PLANNER_PROMPT_VERSION,
            stop_check=stop_check,
            output_summary=lambda value: {
                "task_count": len(value.tasks),
                "expected_final_output_type": value.expected_final_output_type,
            },
        )
        plan.tasks = plan.tasks[: request.max_tasks]
        self.session_revision_repository.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_plan",
            payload=plan.model_dump(mode="json"),
        )
        observations: list[dict[str, Any]] = []
        tool_calls = 0
        self._publish_phase(progress_update, "tool_selection")
        for task_index, task in enumerate(plan.tasks, start=1):
            self._raise_if_stopped(stop_check)
            remaining_tool_calls = request.max_tool_iterations - tool_calls
            if remaining_tool_calls <= 0:
                break
            step = self.session_revision_repository.start_revision_step(
                pipeline_run_id=pipeline_run_id,
                step_name=f"revision_agent_task_{task_index}",
                step_index=task_index + 1,
                step_count=stage_count,
                input_summary={"task_id": task.task_id},
                input_payload=task.model_dump(mode="json"),
                schema_name="revision_agent_tool_call",
                schema_version="1",
                prompt_version=TOOL_PROMPT_VERSION,
                parser_version="structured-llm-v1",
                model_provider=runtime.provider,
                model_name=runtime.model,
            )
            attempt = int(step["attempt_number"])
            task_started = perf_counter()
            task_observations: list[dict[str, Any]] = []
            try:
                self._raise_if_stopped(stop_check)
                if use_native_tools:
                    native_result = self._run_native_tool_task(
                        runtime=runtime,
                        task=task,
                        manifest=manifest,
                        tools=native_tool_definitions,
                        registry=registry,
                        max_tool_calls=remaining_tool_calls,
                        stop_check=stop_check,
                    )
                    task_observations.extend(native_result.observations)
                    observations.extend(native_result.observations)
                    tool_calls += native_result.tool_call_count
                    if native_result.stopped:
                        raise RevisionAgentCancelled("Revision job cancelled.")
                else:
                    for _ in range(remaining_tool_calls):
                        self._raise_if_stopped(stop_check)
                        decision = self._call_schema(
                            runtime,
                            tool_prompt(
                                task.model_dump(mode="json"), task_observations, manifest
                            ),
                            RevisionAgentToolCall,
                            purpose=GenerationPurpose.REVISION_TOOL_SELECTION,
                        )
                        if decision.task_complete:
                            break
                        try:
                            observation = registry.execute(
                                decision.tool_name,
                                decision.arguments,
                                request.allowed_tools,
                            )
                        except ValueError as exc:
                            observation = {
                                "error": str(exc),
                                "invalid_tool_input": True,
                            }
                        task_observations.append(
                            {"tool": decision.tool_name, "observation": observation}
                        )
                        observations.append(task_observations[-1])
                        tool_calls += 1
                self.session_revision_repository.complete_revision_step(
                    pipeline_run_id=pipeline_run_id,
                    step_name=f"revision_agent_task_{task_index}",
                    attempt_number=attempt,
                    status="completed",
                    output_summary={"tool_call_count": len(task_observations)},
                    output_payload={"observations": task_observations},
                    latency_ms=int((perf_counter() - task_started) * 1000),
                    retry_count=max(0, attempt - 1),
                )
            except RevisionAgentCancelled as exc:
                self.session_revision_repository.fail_revision_step(
                    pipeline_run_id=pipeline_run_id,
                    step_name=f"revision_agent_task_{task_index}",
                    attempt_number=attempt,
                    status="cancelled",
                    error={"message": str(exc)},
                    latency_ms=int((perf_counter() - task_started) * 1000),
                )
                raise
            except Exception as exc:
                self.session_revision_repository.fail_revision_step(
                    pipeline_run_id=pipeline_run_id,
                    step_name=f"revision_agent_task_{task_index}",
                    attempt_number=attempt,
                    error={"message": str(exc)},
                    latency_ms=int((perf_counter() - task_started) * 1000),
                )
                raise
            if tool_calls >= request.max_tool_iterations:
                break
        self._raise_if_stopped(stop_check)
        self.session_revision_repository.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_tool_trace",
            payload={"observations": observations},
        )
        self._raise_if_stopped(stop_check)
        self._publish_phase(progress_update, "editing")
        draft = self._call_revision_stage(
            runtime=runtime,
            pipeline_run_id=pipeline_run_id,
            step_name="revision_agent_editor",
            step_index=len(plan.tasks) + 2,
            step_count=stage_count,
            input_summary={
                "observation_count": len(observations),
                "repair_attempt": 0,
            },
            prompt=editor_prompt(context, observations),
            schema=RevisionDraftResult,
            purpose=GenerationPurpose.REVISION_EDITING,
            prompt_version=EDITOR_PROMPT_VERSION,
            stop_check=stop_check,
            output_summary=lambda value: {
                "patch_count": len(value.patches),
                "changed_section_count": len(value.changed_sections),
            },
        )
        source_report = str(
            session.get("official_report_text") or session.get("report") or ""
        )
        draft, applied_report = self._normalize_draft(
            source_report=source_report,
            draft=draft,
            request=request,
        )
        self.session_revision_repository.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_draft_report",
            payload=draft.model_dump(mode="json"),
        )
        self._publish_phase(progress_update, "quality_review")
        qa = self._call_revision_stage(
            runtime=runtime,
            pipeline_run_id=pipeline_run_id,
            step_name="revision_agent_qa",
            step_index=len(plan.tasks) + 3,
            step_count=stage_count,
            input_summary={
                "patch_count": len(draft.patches),
                "changed_section_count": len(draft.changed_sections),
                "repair_attempt": 0,
            },
            prompt=qa_prompt(context, draft.model_dump(mode="json")),
            schema=RevisionAgentQaResult,
            purpose=GenerationPurpose.REVISION_QA,
            prompt_version=QA_PROMPT_VERSION,
            stop_check=stop_check,
            output_summary=lambda value: {
                "blocking_issue_count": len(value.blocking_issues),
                "warning_count": len(value.warnings),
            },
        )
        qa = self._merge_quality_gates(
            qa=qa,
            session=session,
            applied_report=applied_report,
            source_report=source_report,
        )
        self.session_revision_repository.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_qa",
            payload=qa.model_dump(mode="json"),
            status="qa_failed" if qa.blocking_issues else "passed",
        )
        repair_attempt = 0
        if qa.blocking_issues and MAX_QA_REPAIR_ATTEMPTS > 0:
            repair_attempt = 1
            self._publish_phase(
                progress_update,
                "repairing",
                repair_attempt=repair_attempt,
            )
            repair_draft = self._call_revision_stage(
                runtime=runtime,
                pipeline_run_id=pipeline_run_id,
                step_name="revision_agent_editor",
                step_index=len(plan.tasks) + 2,
                step_count=stage_count,
                input_summary={
                    "observation_count": len(observations),
                    "repair_attempt": repair_attempt,
                    "blocking_issue_count": len(qa.blocking_issues),
                },
                prompt=repair_editor_prompt(
                    context,
                    observations,
                    draft.model_dump(mode="json"),
                    qa.blocking_issues,
                ),
                schema=RevisionDraftResult,
                purpose=GenerationPurpose.REVISION_EDITING,
                prompt_version=REPAIR_PROMPT_VERSION,
                stop_check=stop_check,
                output_summary=lambda value: {
                    "patch_count": len(value.patches),
                    "changed_section_count": len(value.changed_sections),
                    "repair_attempt": repair_attempt,
                },
            )
            repair_draft, applied_report = self._normalize_draft(
                source_report=source_report,
                draft=repair_draft,
                request=request,
            )
            draft = repair_draft
            self.session_revision_repository.persist_revision_artifact(
                pipeline_run_id=pipeline_run_id,
                revision_version_id=revision_version_id,
                artifact_key="revision_agent_draft_report_repair",
                payload={
                    **draft.model_dump(mode="json"),
                    "repair_attempt": repair_attempt,
                    "prior_blocking_issues": qa.blocking_issues,
                },
                status="pending_qa",
            )
            self._publish_phase(
                progress_update,
                "quality_review",
                repair_attempt=repair_attempt,
            )
            qa = self._call_revision_stage(
                runtime=runtime,
                pipeline_run_id=pipeline_run_id,
                step_name="revision_agent_qa",
                step_index=len(plan.tasks) + 3,
                step_count=stage_count,
                input_summary={
                    "patch_count": len(draft.patches),
                    "changed_section_count": len(draft.changed_sections),
                    "repair_attempt": repair_attempt,
                },
                prompt=qa_prompt(context, draft.model_dump(mode="json")),
                schema=RevisionAgentQaResult,
                purpose=GenerationPurpose.REVISION_QA,
                prompt_version=QA_PROMPT_VERSION,
                stop_check=stop_check,
                output_summary=lambda value: {
                    "blocking_issue_count": len(value.blocking_issues),
                    "warning_count": len(value.warnings),
                    "repair_attempt": repair_attempt,
                },
            )
            qa = self._merge_quality_gates(
                qa=qa,
                session=session,
                applied_report=applied_report,
                source_report=source_report,
            )
            self.session_revision_repository.persist_revision_artifact(
                pipeline_run_id=pipeline_run_id,
                revision_version_id=revision_version_id,
                artifact_key="revision_agent_qa_repair",
                payload={
                    **qa.model_dump(mode="json"),
                    "repair_attempt": repair_attempt,
                },
                status="qa_failed" if qa.blocking_issues else "passed",
            )
        self._publish_phase(
            progress_update,
            "finalizing",
            repair_attempt=repair_attempt,
        )
        model_configuration = {
            **model_configuration,
            "repair_attempt_count": repair_attempt,
            "provider_max_retries": REVISION_PROVIDER_MAX_RETRIES,
        }
        revised_session_id: int | None = None
        has_validated_edit = applied_report != source_report and bool(draft.patches)
        version_status = (
            "requires_human_review"
            if not has_validated_edit
            else "qa_failed"
            if qa.blocking_issues
            else "llm_qa_passed"
        )
        llm_qa_status = (
            "requires_human_review"
            if not has_validated_edit
            else "failed"
            if qa.blocking_issues
            else "passed"
        )
        if not request.dry_run and not qa.blocking_issues and has_validated_edit:
            root_session_id = int(model_configuration["root_session_id"])
            session_sections = session.get("sections")
            sections: dict[str, Any] = (
                session_sections if isinstance(session_sections, dict) else {}
            )
            revised_session_id = self.clinical_session_repository.save_clinical_session(
                {
                    "patient_name": session.get("patient_name"),
                    "session_timestamp": datetime.now(UTC),
                    "version": self.session_revision_repository.get_next_session_version(
                        root_session_id
                    ),
                    "root_session_id": root_session_id,
                    "session_kind": "agentic_revision",
                    "session_status": "successful",
                    "anamnesis": sections.get("anamnesis"),
                    "drugs": sections.get("drugs") or sections.get("therapy"),
                    "laboratory_analysis": sections.get("laboratory_analysis"),
                    "final_report": applied_report,
                    "session_result_payload": {
                        **(session.get("result_payload") or {}),
                        "report": applied_report,
                        "revision": {
                            "pipeline_run_id": pipeline_run_id,
                            "qa": qa.model_dump(mode="json"),
                            "source_session_id": session["session_id"],
                        },
                    },
                    "metadata": {
                        **(session.get("metadata") or {}),
                        "revision_source_session_id": session["session_id"],
                    },
                }
            )
            if revised_session_id is None:
                raise RuntimeError("Revision draft could not be persisted.")
        if not request.dry_run:
            self.session_revision_repository.finalize_revision_version(
                pipeline_run_id=pipeline_run_id,
                persisted_session_id=revised_session_id,
                model_configuration=model_configuration,
                version_status=version_status,
                llm_qa_status=llm_qa_status,
                clinical_review_status="not_reviewed",
            )
        self.session_revision_repository.create_or_update_revision_run(
            pipeline_run_id=pipeline_run_id,
            session_id=int(session["session_id"]),
            root_session_id=int(model_configuration["root_session_id"]),
            source_version_id=int(model_configuration["source_version_id"]),
            target_revision_version_id=revision_version_id,
            revision_mode="agentic_revision",
            revision_kind="llm_assisted_revision",
            configuration=model_configuration,
            reviewer_note=request.revision_instruction,
            status="completed",
            completed_at=datetime.now(UTC),
        )
        self._publish_phase(
            progress_update,
            "completed",
            repair_attempt=repair_attempt,
        )
        return {
            "pipeline_run_id": pipeline_run_id,
            "revision_version_id": revision_version_id,
            "revised_session_id": revised_session_id,
            "revision_status": "dry_run" if request.dry_run else version_status,
            "task_count": len(plan.tasks),
            "tool_call_count": tool_calls,
            "blocking_issue_count": len(qa.blocking_issues),
            "manual_review_required": True,
            "revision_phase": "completed",
            "repair_attempt": repair_attempt,
        }

    # -------------------------------------------------------------------------
    def _run_native_tool_task(
        self,
        *,
        runtime: RevisionAgentRuntime,
        task: Any,
        manifest: list[str],
        tools: list[Any],
        registry: RevisionToolRegistry,
        max_tool_calls: int,
        stop_check: StopCheck | None,
    ) -> ToolLoopResult:
        async def run() -> ToolLoopResult:
            client = select_llm_provider(
                provider=runtime.provider,
                default_model=runtime.model,
                timeout_s=_revision_provider_timeout(runtime.provider),
                max_retries=REVISION_PROVIDER_MAX_RETRIES,
            )
            try:
                executor = ToolLoopExecutor(
                    chat=client.chat_result,  # type: ignore[attr-defined]
                    execute=lambda name, arguments: registry.execute(
                        name, arguments, manifest
                    ),
                    max_iterations=max(1, max_tool_calls),
                    max_tool_calls=max(1, max_tool_calls),
                    stop_check=stop_check,
                )
                try:
                    return await executor.run(
                        model=runtime.model,
                        messages=[
                            ChatMessage(
                                role="system",
                                content=REVISION_AGENT_SYSTEM_PROMPT,
                            ),
                            ChatMessage(
                                role="user",
                                content=tool_prompt(
                                    task.model_dump(mode="json"), [], manifest
                                ),
                            ),
                        ],
                        tools=tools,
                        purpose=GenerationPurpose.REVISION_TOOL_SELECTION,
                    )
                except Exception as exc:  # noqa: BLE001
                    if getattr(exc, "error_code", None) == "cancelled" or (
                        stop_check is not None and stop_check()
                    ):
                        raise RevisionAgentCancelled("Revision job cancelled.") from exc
                    raise
            finally:
                close = getattr(client, "close", None)
                if close is not None:
                    await close()

        return asyncio.run(run())

    # -------------------------------------------------------------------------
    @staticmethod
    def _raise_if_stopped(stop_check: StopCheck | None) -> None:
        if stop_check is not None and stop_check():
            raise RevisionAgentCancelled("Revision job cancelled.")

    # -------------------------------------------------------------------------
    def _call_schema(
        self,
        runtime: RevisionAgentRuntime,
        user_prompt: str,
        schema: type[Any],
        *,
        purpose: GenerationPurpose,
    ) -> Any:
        if self.structured_call is not None:
            return schema.model_validate(
                self.structured_call(
                    model=runtime.model,
                    system_prompt=REVISION_AGENT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema=schema,
                    purpose=purpose,
                )
            )
        client = select_llm_provider(
            provider=runtime.provider,
            default_model=runtime.model,
            timeout_s=_revision_provider_timeout(runtime.provider),
            max_retries=REVISION_PROVIDER_MAX_RETRIES,
        )
        async def call() -> Any:
            try:
                return await client.llm_structured_call(
                    model=runtime.model,
                    system_prompt=REVISION_AGENT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema=schema,
                    purpose=purpose,
                    use_json_mode=True,
                    max_repair_attempts=3,
                )
            finally:
                close = getattr(client, "close", None)
                if close is not None:
                    await close()

        return asyncio.run(call())

    # -------------------------------------------------------------------------
    def _run_structured_scan(
        self,
        *,
        runtime: RevisionAgentRuntime,
        user_prompt: str,
    ) -> RevisionIssueScanResult:
        if self.structured_call is not None:
            value = self.structured_call(
                model=runtime.model,
                system_prompt=REVISION_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=RevisionIssueScanResult,
                purpose=GenerationPurpose.REVISION_SCAN,
            )
            if isinstance(value, RevisionIssueScanResult):
                return value
            return RevisionIssueScanResult.model_validate(value)

        client = select_llm_provider(
            provider=runtime.provider,
            default_model=runtime.model,
            timeout_s=_revision_provider_timeout(runtime.provider),
            max_retries=REVISION_PROVIDER_MAX_RETRIES,
        )
        async def call() -> RevisionIssueScanResult:
            try:
                return await client.llm_structured_call(
                    model=runtime.model,
                    system_prompt=REVISION_AGENT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema=RevisionIssueScanResult,
                    purpose=GenerationPurpose.REVISION_SCAN,
                    use_json_mode=True,
                    max_repair_attempts=3,
                )
            finally:
                close = getattr(client, "close", None)
                if close is not None:
                    await close()

        return asyncio.run(call())
