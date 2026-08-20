from __future__ import annotations

import asyncio

from services.llm import ollama_structured
from services.llm.generation_policy import GenerationPurpose

###############################################################################
def test_resolve_text_extraction_models_prefers_live_installed_models() -> None:

    ###############################################################################
    class FakeClient:
        default_model = "gpt-4.1-mini"

        # -------------------------------------------------------------------------
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

    ###############################################################################
    class FakeParser:

        # -------------------------------------------------------------------------
        def __init__(self) -> None:
            self.calls = 0

        # -------------------------------------------------------------------------
        def parse(self, text: str) -> dict[str, str]:
            self.calls += 1
            if self.calls == 1:
                raise ValueError("schema echo")
            return {"ok": text}

    ###############################################################################
    class FakeClient:
        captured_messages: list[dict[str, str]] | None = None

        # -------------------------------------------------------------------------
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
def test_chat_structured_model_forwards_generation_purpose() -> None:

    ###############################################################################
    class FakeClient:
        captured_purpose: GenerationPurpose | None = None

        # -------------------------------------------------------------------------
        async def chat(self, **kwargs):
            self.captured_purpose = kwargs["purpose"]
            return '{"ok": true}'

    client = FakeClient()

    result = asyncio.run(
        ollama_structured._chat_structured_model(
            client,
            active_model="gpt-oss:20b",
            messages=[],
            use_json_mode=True,
            temperature=0.0,
            purpose=GenerationPurpose.STRUCTURED_EXTRACTION,
        )
    )

    assert result == '{"ok": true}'
    assert client.captured_purpose is GenerationPurpose.STRUCTURED_EXTRACTION
