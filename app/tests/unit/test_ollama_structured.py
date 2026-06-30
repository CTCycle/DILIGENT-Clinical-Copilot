from __future__ import annotations

import asyncio
import logging

import pytest

from services.llm import ollama_structured


###############################################################################
def test_resolve_text_extraction_models_prefers_live_installed_models() -> None:
    class FakeClient:
        default_model = "gpt-4.1-mini"

        async def get_cached_models(self) -> set[str]:
            return {"qwen3.5:2b", "qwen3.5:9b"}

    models = asyncio.run(
        ollama_structured.resolve_text_extraction_models(FakeClient(), "gpt-4.1-mini")
    )

    assert models == ["qwen3.5:2b", "qwen3.5:9b"]


###############################################################################
def test_looks_like_schema_echo_detects_schema_payload() -> None:
    schema_like = (
        '{"$defs":{"Item":{"properties":{"name":{"type":"string"}}}},'
        '"title":"Payload","type":"object","required":["entries"]}'
    )

    assert ollama_structured.looks_like_schema_echo(schema_like) is True


###############################################################################
def test_parse_with_repairs_uses_compact_repair_messages_for_schema_echo() -> None:
    class FakeParser:
        def __init__(self) -> None:
            self.calls = 0

        def parse(self, text: str) -> dict[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("schema echo")
            return {"ok": text}

    class FakeClient:
        captured_messages: list[dict[str, str]] | None = None

        async def chat(self, **kwargs):
            self.captured_messages = kwargs["messages"]
            return '{"entries":[]}'

        build_repair_messages = staticmethod(ollama_structured.build_repair_messages)
        build_compact_repair_messages = staticmethod(
            ollama_structured.build_compact_repair_messages
        )
        _coerce_llm_text = staticmethod(ollama_structured._coerce_llm_text)

    parser = FakeParser()
    client = FakeClient()

    result = asyncio.run(
        ollama_structured.parse_with_repairs(
            client,
            parser=parser,
            text='{"$defs":{"Item":{"properties":{"name":{"type":"string"}}}},"title":"Payload","type":"object","required":["entries"]}',
            active_model="qwen3.5:2b",
            system_prompt="ignored",
            format_instructions="ignored",
            use_json_mode=True,
            max_repair_attempts=1,
        )
    )

    assert result == {"ok": '{"entries":[]}'}
    assert client.captured_messages is not None
    assert "schema or wrapper instead of data" in client.captured_messages[1]["content"]


###############################################################################
def test_parse_failure_logs_hash_not_raw_ollama_output(caplog) -> None:
    class FakeParser:
        def parse(self, text: str) -> dict[str, str]:
            _ = text
            raise ValueError("bad json")

    class FakeClient:
        async def chat(self, **kwargs):
            _ = kwargs
            return "Patient Mario Rossi CF RSSMRA fake PHI {not valid json"

        build_repair_messages = staticmethod(ollama_structured.build_repair_messages)
        build_compact_repair_messages = staticmethod(
            ollama_structured.build_compact_repair_messages
        )
        _coerce_llm_text = staticmethod(ollama_structured._coerce_llm_text)

    caplog.set_level(logging.ERROR)
    with pytest.raises(RuntimeError):
        asyncio.run(
            ollama_structured.parse_with_repairs(
                FakeClient(),
                parser=FakeParser(),
                text="Patient Mario Rossi CF RSSMRA fake PHI {not valid json",
                active_model="qwen3.5:2b",
                system_prompt="Return JSON.",
                format_instructions="Return JSON.",
                use_json_mode=True,
                max_repair_attempts=1,
            )
        )

    logs = caplog.text
    assert "Structured parse failed after retries" in logs
    assert "output_hash=" in logs
    assert "Mario Rossi" not in logs
    assert "RSSMRA" not in logs
