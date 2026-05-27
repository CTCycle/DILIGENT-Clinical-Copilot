from __future__ import annotations

import json
from typing import get_args

from app import app
from domain.keys import ProviderName


def test_openrouter_is_not_supported_provider() -> None:
    assert "openrouter" not in get_args(ProviderName)


def test_provider_descriptions_match_supported_providers() -> None:
    assert set(get_args(ProviderName)) == {"openai", "gemini", "brave"}


def test_access_key_openapi_schema_excludes_openrouter() -> None:
    schema = app.openapi()
    assert "openrouter" not in json.dumps(schema).lower()
