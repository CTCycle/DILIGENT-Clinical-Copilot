from __future__ import annotations

import asyncio

from services.llm import ollama_chat
from services.llm.generation_policy import GenerationPurpose
from services.llm.runtime_config import LLMRuntimeConfig


###############################################################################
def test_ollama_gpt_oss_preserves_level_reasoning_and_omits_temperature(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "get_ollama_seed",
        classmethod(lambda cls: None),
    )

    temperature, think, options = ollama_chat.prepare_generation_parameters(
        object(),
        model="gpt-oss:20b",
        purpose=GenerationPurpose.CLINICAL_SYNTHESIS,
        temperature=0.7,
        think="high",
        options={"temperature": 0.7},
    )

    assert temperature is None
    assert think == "high"
    assert options is None


###############################################################################
def test_ollama_context_window_is_task_sized_and_intersects_runtime_capacity() -> None:

    ###############################################################################
    class FakeOllama:
        # -------------------------------------------------------------------------
        @staticmethod
        def estimate_tokens(text: str) -> int:
            return max(1, len(text.split()))

        # -------------------------------------------------------------------------
        async def get_model_context_limit(self, model: str) -> int:
            assert model == "qwen3:8b"
            return 8192

        # -------------------------------------------------------------------------
        async def estimate_max_feasible_context(self, model: str) -> int:
            assert model == "qwen3:8b"
            return 4096

    context_window = asyncio.run(
        ollama_chat.calculate_context_window(
            FakeOllama(),
            model="qwen3:8b",
            messages=[{"role": "user", "content": "clinical evidence " * 20}],
            purpose=GenerationPurpose.TIMELINE_EXTRACTION,
            min_ctx=2048,
        )
    )

    assert context_window is not None
    assert 2048 <= context_window <= 4096


###############################################################################
def test_ollama_payload_omits_think_when_transport_does_not_support_reasoning() -> None:

    ###############################################################################
    class FakeOllama:
        # -------------------------------------------------------------------------
        @staticmethod
        def compose_payload(
            payload: dict[str, object],
            *,
            format: str | None,
            options: dict[str, object] | None,
            keep_alive: str | None,
        ) -> dict[str, object]:
            return ollama_chat.compose_payload(
                payload,
                format=format,
                options=options,
                keep_alive=keep_alive,
            )

    payload = ollama_chat.build_chat_payload(
        FakeOllama(),
        model="unknown-model",
        messages=[{"role": "user", "content": "hello"}],
        stream=False,
        format=None,
        temperature=None,
        think=None,
        options=None,
        keep_alive=None,
    )

    assert "think" not in payload
