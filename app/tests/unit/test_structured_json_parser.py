from __future__ import annotations

import pytest
import services.llm.structured as structured_module
from services.llm.structured import parse_json_object_strict

###############################################################################
def test_parse_json_object_strict_valid_object() -> None:
    parsed = parse_json_object_strict('{"a": 1, "b": "x"}')
    assert parsed == {"a": 1, "b": "x"}

###############################################################################
def test_parse_json_object_strict_fenced_json() -> None:
    parsed = parse_json_object_strict('```json\n{"a": 1}\n```')
    assert parsed == {"a": 1}

###############################################################################
def test_parse_json_object_strict_rejects_leading_prose() -> None:
    with pytest.raises(
        ValueError, match="invalid_json_object|leading_prose_not_allowed"
    ):
        parse_json_object_strict('hello {"a":1}')

###############################################################################
def test_parse_json_object_strict_rejects_trailing_prose() -> None:
    with pytest.raises(ValueError, match="trailing_prose_not_allowed"):
        parse_json_object_strict('{"a":1} trailing')

###############################################################################
def test_parse_json_object_strict_rejects_multiple_objects() -> None:
    with pytest.raises(ValueError, match="trailing_prose_not_allowed"):
        parse_json_object_strict('{"a":1}{"b":2}')

###############################################################################
def test_legacy_lenient_json_helpers_are_absent() -> None:
    assert not hasattr(structured_module, "extract_first_json_dict_lenient_nonclinical")
    assert not hasattr(structured_module, "parse_json_dict_lenient_nonclinical")
