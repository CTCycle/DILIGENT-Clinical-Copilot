from __future__ import annotations

from datetime import UTC, datetime

from repositories.schemas.models import Base
from repositories.serialization.data import DataSerializer
from sqlalchemy import create_engine

###############################################################################
def build_serializer() -> DataSerializer:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    return DataSerializer(engine=engine)

###############################################################################
def test_revision_step_retry_supersedes_previous_attempt_and_increments_retry_count() -> None:
    serializer = build_serializer()
    pipeline_run_id = "checkpoint-run-001"
    started_at = datetime.now(UTC)

    first = serializer.start_revision_step(
        pipeline_run_id=pipeline_run_id,
        step_name="generate_revision",
        step_index=5,
        step_count=16,
        input_summary={"phase": "first"},
        input_payload={"phase": "first"},
        schema_name="revision_output",
        schema_version="1",
        prompt_version="prompt-a",
        parser_version="parser-a",
        model_provider="ollama",
        model_name="model-a",
        started_at=started_at,
    )
    assert first["attempt_number"] == 1
    assert first["retry_count"] == 0
    assert first["superseded_at"] is None

    serializer.fail_revision_step(
        pipeline_run_id=pipeline_run_id,
        step_name="generate_revision",
        attempt_number=1,
        error={"message": "temporary failure"},
        completed_at=started_at,
    )

    second = serializer.start_revision_step(
        pipeline_run_id=pipeline_run_id,
        step_name="generate_revision",
        step_index=5,
        step_count=16,
        input_summary={"phase": "retry"},
        input_payload={"phase": "retry"},
        schema_name="revision_output",
        schema_version="1",
        prompt_version="prompt-a",
        parser_version="parser-a",
        model_provider="ollama",
        model_name="model-a",
        started_at=started_at,
    )
    assert second["attempt_number"] == 2
    assert second["retry_count"] == 1

    serializer.complete_revision_step(
        pipeline_run_id=pipeline_run_id,
        step_name="generate_revision",
        attempt_number=2,
        status="completed",
        output_summary={"phase": "retry"},
        output_payload={"report": "ok"},
        token_usage={"total_tokens": 42},
        latency_ms=123,
        completed_at=started_at,
    )

    steps = serializer.list_revision_steps(pipeline_run_id)
    assert [step["attempt_number"] for step in steps] == [1, 2]
    assert steps[0]["status"] == "failed"
    assert steps[0]["superseded_at"] is not None
    assert steps[1]["status"] == "completed"
    assert steps[1]["superseded_at"] is None
    assert steps[1]["token_usage"] == {"total_tokens": 42}
    assert steps[1]["latency_ms"] == 123
    assert steps[1]["schema_name"] == "revision_output"
    assert steps[1]["schema_version"] == "1"
    assert steps[1]["prompt_version"] == "prompt-a"
