from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Callable

from domain.inspection import (
    RevisionAgentPlan,
    RevisionAgentQaResult,
    RevisionAgentToolCall,
    RevisionDraftResult,
    RevisionIssueScanResult,
    SessionRevisionRequest,
)
from services.inspection.revision_context import build_revision_context
from services.inspection.revision_patches import validate_draft_report
from common.prompts.revision_agent import (
    REVISION_AGENT_SYSTEM_PROMPT,
    editor_prompt,
    planner_prompt,
    qa_prompt,
    tool_prompt,
)
from services.inspection.revision_tools import RevisionToolRegistry
from services.llm.provider_factory import select_llm_provider
from services.llm.runtime_config import LLMRuntimeConfig
from services.llm.generation_policy import GenerationPurpose

REVISION_AGENT_PROMPT_VERSION = "revision-agent-issue-scan-v1"
REVISION_AGENT_SCHEMA_NAME = "revision_issue_scan_result"
REVISION_AGENT_SCHEMA_VERSION = "1"
REVISION_AGENT_STEP_NAME = "revision_agent_issue_scan"


MAX_TEXT_CHARS = 30000
MAX_REPORT_CHARS = 20000
MAX_JSON_CHARS = 30000

StructuredCall = Callable[..., Any]


###############################################################################
@dataclass(frozen=True)
class RevisionAgentRuntime:
    provider: str
    model: str


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
def resolve_revision_agent_runtime(
    model_overrides: dict[str, Any] | None,
) -> RevisionAgentRuntime:
    overrides = dict(model_overrides or {})
    with LLMRuntimeConfig.override_for_run(overrides):
        provider, model = LLMRuntimeConfig.resolve_provider_and_model("clinical")
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
    return (
        "Inspect the following revision packet and return the structured issue scan.\n"
        "The user_revision_context may steer focus but is not clinical evidence.\n\n"
        f"{_safe_json(packet, MAX_TEXT_CHARS + MAX_REPORT_CHARS + MAX_JSON_CHARS)}"
    )


###############################################################################
class RevisionAgentRunner:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        serializer: Any,
        structured_call: StructuredCall | None = None,
    ) -> None:
        self.serializer = serializer
        self.structured_call = structured_call

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
        runtime = resolve_revision_agent_runtime(request.model_overrides)
        user_prompt = build_revision_agent_user_prompt(
            session=session,
            request=request,
        )
        step = self.serializer.start_revision_step(
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
            self.serializer.complete_revision_step(
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
            artifact = self.serializer.persist_revision_agent_issue_scan(
                pipeline_run_id=pipeline_run_id,
                revision_version_id=revision_version_id,
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
            self.serializer.create_or_update_revision_run(
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
            self.serializer.fail_revision_step(
                pipeline_run_id=pipeline_run_id,
                step_name=REVISION_AGENT_STEP_NAME,
                attempt_number=attempt_number,
                error=error,
                latency_ms=latency_ms,
            )
            self.serializer.fail_revision_run(
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
    ) -> dict[str, Any]:
        del job_id
        runtime = resolve_revision_agent_runtime(request.model_overrides)
        lineage = self.serializer.list_session_versions(int(session["session_id"]))
        context = build_revision_context(
            session=session,
            manual_edits=session.get("manual_edit_history") or [],
            lineage=lineage,
            selected_text=request.selected_text,
            instruction=request.revision_instruction,
        )
        registry = RevisionToolRegistry(
            serializer=self.serializer, session=session, context=context
        )
        manifest = registry.manifest(request.allowed_tools)
        self.serializer.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_context",
            payload=context,
        )
        plan = self._call_schema(
            runtime, planner_prompt(context, manifest), RevisionAgentPlan
        )
        plan.tasks = plan.tasks[: request.max_tasks]
        self.serializer.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_plan",
            payload=plan.model_dump(mode="json"),
        )
        observations: list[dict[str, Any]] = []
        tool_calls = 0
        for task_index, task in enumerate(plan.tasks, start=1):
            step = self.serializer.start_revision_step(
                pipeline_run_id=pipeline_run_id,
                step_name=f"revision_agent_task_{task_index}",
                step_index=task_index + 1,
                step_count=len(plan.tasks) + 3,
                input_summary={"task_id": task.task_id},
                input_payload=task.model_dump(mode="json"),
                schema_name="revision_agent_tool_call",
                schema_version="1",
                prompt_version="revision-agent-tool-controller-v1",
                parser_version="structured-llm-v1",
                model_provider=runtime.provider,
                model_name=runtime.model,
            )
            attempt = int(step["attempt_number"])
            task_observations: list[dict[str, Any]] = []
            try:
                for _ in range(request.max_tool_iterations - tool_calls):
                    decision = self._call_schema(
                        runtime,
                        tool_prompt(
                            task.model_dump(mode="json"), task_observations, manifest
                        ),
                        RevisionAgentToolCall,
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
                self.serializer.complete_revision_step(
                    pipeline_run_id=pipeline_run_id,
                    step_name=f"revision_agent_task_{task_index}",
                    attempt_number=attempt,
                    status="completed",
                    output_summary={"tool_call_count": len(task_observations)},
                    output_payload={"observations": task_observations},
                )
            except Exception as exc:
                self.serializer.fail_revision_step(
                    pipeline_run_id=pipeline_run_id,
                    step_name=f"revision_agent_task_{task_index}",
                    attempt_number=attempt,
                    error={"message": str(exc)},
                )
                raise
            if tool_calls >= request.max_tool_iterations:
                break
        self.serializer.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_tool_trace",
            payload={"observations": observations},
        )
        draft = self._call_schema(
            runtime, editor_prompt(context, observations), RevisionDraftResult
        )
        source_report = str(
            session.get("official_report_text") or session.get("report") or ""
        )
        applied_report = validate_draft_report(source_report, draft.patches)
        if applied_report != draft.revised_report_text:
            raise ValueError(
                "Revision draft text must equal deterministic patch output."
            )
        self.serializer.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_draft_report",
            payload=draft.model_dump(mode="json"),
        )
        qa = self._call_schema(
            runtime,
            qa_prompt(context, draft.model_dump(mode="json")),
            RevisionAgentQaResult,
        )
        self.serializer.persist_revision_artifact(
            pipeline_run_id=pipeline_run_id,
            revision_version_id=revision_version_id,
            artifact_key="revision_agent_qa",
            payload=qa.model_dump(mode="json"),
            status="qa_failed" if qa.blocking_issues else "passed",
        )
        revised_session_id: int | None = None
        version_status = "qa_failed" if qa.blocking_issues else "llm_qa_passed"
        if not request.dry_run:
            root_session_id = int(model_configuration["root_session_id"])
            revised_session_id = self.serializer.save_clinical_session(
                {
                    "patient_name": session.get("patient_name"),
                    "session_timestamp": datetime.now(UTC),
                    "version": self.serializer.get_next_session_version(
                        root_session_id
                    ),
                    "root_session_id": root_session_id,
                    "session_kind": "agentic_revision",
                    "session_status": "successful",
                    "anamnesis": (session.get("sections") or {}).get("anamnesis"),
                    "drugs": (session.get("sections") or {}).get("therapy"),
                    "laboratory_analysis": (session.get("sections") or {}).get(
                        "laboratory_analysis"
                    ),
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
            self.serializer.finalize_revision_version(
                pipeline_run_id=pipeline_run_id,
                persisted_session_id=revised_session_id,
                model_configuration=model_configuration,
                version_status=version_status,
                llm_qa_status="failed" if qa.blocking_issues else "passed",
                clinical_review_status="not_reviewed",
            )
        self.serializer.create_or_update_revision_run(
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
        return {
            "pipeline_run_id": pipeline_run_id,
            "revision_version_id": revision_version_id,
            "revised_session_id": revised_session_id,
            "revision_status": "dry_run" if request.dry_run else version_status,
            "task_count": len(plan.tasks),
            "tool_call_count": tool_calls,
            "blocking_issue_count": len(qa.blocking_issues),
            "manual_review_required": True,
        }

    # -------------------------------------------------------------------------
    def _call_schema(
        self, runtime: RevisionAgentRuntime, user_prompt: str, schema: type[Any]
    ) -> Any:
        if self.structured_call is not None:
            return schema.model_validate(
                self.structured_call(
                    model=runtime.model,
                    system_prompt=REVISION_AGENT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    schema=schema,
                    purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
                )
            )
        client = select_llm_provider(
            provider=runtime.provider, default_model=runtime.model
        )
        return asyncio.run(
            client.llm_structured_call(
                model=runtime.model,
                system_prompt=REVISION_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=schema,
                purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
                use_json_mode=True,
                max_repair_attempts=1,
            )
        )

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
                purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
            )
            if isinstance(value, RevisionIssueScanResult):
                return value
            return RevisionIssueScanResult.model_validate(value)

        client = select_llm_provider(
            provider=runtime.provider,
            default_model=runtime.model,
        )
        return asyncio.run(
            client.llm_structured_call(
                model=runtime.model,
                system_prompt=REVISION_AGENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=RevisionIssueScanResult,
                purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
                use_json_mode=True,
                max_repair_attempts=1,
            )
        )
