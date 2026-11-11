# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any

import pytest
from domain.agents import (
    DiliAgentConfig,
    RevisionAgentConfig,
    agent_config_hash,
    parse_agent_config,
)
from pydantic import ValidationError


###############################################################################
def dili_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "dili",
        "models": {
            "extraction": {"provider": "ollama", "model": "qwen3:8b"},
            "clinical": {"provider": "openai", "model": "gpt-4.1"},
            "timeline": {"provider": "ollama", "model": "qwen3:8b"},
        },
    }
    payload.update(overrides)
    return payload


###############################################################################
def revision_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": "revision",
        "models": {"planning": {"provider": "openai", "model": "gpt-4.1"}},
    }
    payload.update(overrides)
    return payload


###############################################################################
def test_agent_config_discriminates_kind_and_rejects_foreign_fields() -> None:
    assert isinstance(parse_agent_config(dili_payload()), DiliAgentConfig)
    assert isinstance(parse_agent_config(revision_payload()), RevisionAgentConfig)
    with pytest.raises(ValidationError):
        parse_agent_config(revision_payload(retrieval={}))
    with pytest.raises(ValidationError):
        parse_agent_config(dili_payload(limits={"max_tasks": 2}))


###############################################################################
def test_agent_config_hash_is_canonical_and_content_sensitive() -> None:
    first = parse_agent_config(dili_payload())
    reordered = parse_agent_config(
        {"models": dili_payload()["models"], "kind": "dili"}
    )
    changed = parse_agent_config(
        dili_payload(instructions={"assessment_focus": "Chronology first."})
    )
    assert agent_config_hash(first) == agent_config_hash(reordered)
    assert agent_config_hash(first) != agent_config_hash(changed)


###############################################################################
def test_retrieval_policy_rejects_inconsistent_counts() -> None:
    with pytest.raises(ValidationError):
        parse_agent_config(
            dili_payload(
                retrieval={
                    "retrieval_candidate_count": 2,
                    "retrieval_selected_count": 5,
                }
            )
        )
