from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest
from pydantic import BaseModel

from services.llm.cloud import CloudLLMClient
from services.llm.structured import StructuredOutputParser

###############################################################################
class FakeClinicalSchema(BaseModel):
    status: str

###############################################################################
def test_structured_parse_failure_logs_hash_not_raw_output(caplog) -> None:
    client = CloudLLMClient.__new__(CloudLLMClient)

    async def fail_repair(**kwargs: Any) -> str:
        _ = kwargs
        return "Patient Mario Rossi CF RSSMRA fake PHI {not valid json"

    client.chat = fail_repair
    parser = StructuredOutputParser(schema=FakeClinicalSchema)

    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        asyncio.run(
            client.parse_with_repairs(
                parser=parser,
                text="Patient Mario Rossi CF RSSMRA fake PHI {not valid json",
                model="test-model",
                system_prompt="Return JSON.",
                format_instructions="Return a schema-valid JSON object.",
                use_json_mode=True,
                max_repair_attempts=1,
            )
        )

    logs = caplog.text
    assert "Structured parse failed after retries" in logs
    assert "output_hash=" in logs
    assert "Mario Rossi" not in logs
    assert "RSSMRA" not in logs
