from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from DILIGENT.server.configurations.runtime_state import LLMRuntimeConfig
from DILIGENT.server.models.providers import OllamaClient


# -----------------------------------------------------------------------------
def test_evaluate_dual_residency_plan_checks_ram_and_vram(monkeypatch) -> None:
    gib = 1_073_741_824
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "get_clinical_model",
        classmethod(lambda cls: "clinical-model"),
    )
    monkeypatch.setattr(
        LLMRuntimeConfig,
        "get_parsing_model",
        classmethod(lambda cls: "text-model"),
    )
    monkeypatch.setattr(
        OllamaClient,
        "get_available_memory_bytes",
        staticmethod(lambda: 20 * gib),
    )
    monkeypatch.setattr(
        OllamaClient,
        "get_available_vram_bytes",
        staticmethod(lambda: 16 * gib),
    )
    monkeypatch.setattr(client, "list_running_models", AsyncMock(return_value={}))
    monkeypatch.setattr(
        client,
        "get_model_footprint_bytes",
        AsyncMock(side_effect=[(5 * gib, 4 * gib), (4 * gib, 3 * gib)]),
    )

    plan = asyncio.run(client.evaluate_dual_residency_plan())
    asyncio.run(client.close())

    assert plan["dual_residency"] is True
    assert plan["required_ram"] == 9 * gib
    assert plan["required_vram"] == 7 * gib


# -----------------------------------------------------------------------------
def test_predict_next_target_model_prefers_frequent_transition() -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    now = time.monotonic()
    client.residency_usage_window_s = 600.0
    client.residency_transition_window_s = 120.0
    client.residency_usage_history.extend(
        [
            (now - 50, "clinical-model"),
            (now - 45, "text-model"),
            (now - 40, "clinical-model"),
            (now - 35, "text-model"),
            (now - 20, "clinical-model"),
        ]
    )

    predicted = client.predict_next_target_model(
        current_model="clinical-model",
        target_models=["clinical-model", "text-model"],
    )
    asyncio.run(client.close())

    assert predicted == "text-model"


# -----------------------------------------------------------------------------
def test_resolve_policy_keep_alive_uses_dual_setting(monkeypatch) -> None:
    client = OllamaClient(base_url="http://127.0.0.1:11434")
    client.residency_dual_keep_alive = "8h"
    monkeypatch.setattr(
        client,
        "get_cached_residency_plan",
        AsyncMock(
            return_value={
                "models": ["clinical-model", "text-model"],
                "dual_residency": True,
            }
        ),
    )

    keep_alive = asyncio.run(
        client.resolve_policy_keep_alive(
            active_model="clinical-model",
            requested_keep_alive=None,
        )
    )
    asyncio.run(client.close())

    assert keep_alive == "8h"
