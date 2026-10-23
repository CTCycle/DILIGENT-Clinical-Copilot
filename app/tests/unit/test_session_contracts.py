# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
from services.runtime.async_batches import run_batched_in_order
import pytest
from domain.clinical.entities import ClinicalSessionRequest
from fastapi import HTTPException
from api.session_validation import validate_clinical_session_request

def test_parallelization_runner_preserves_input_order() -> None:
    async def worker(name: str) -> str:
        await asyncio.sleep(0.001)
        return f"done:{name}"

    inputs = ["therapy", "anamnesis", "disease", "labs"]
    outputs = asyncio.run(
        run_batched_in_order(
            inputs,
            batch_size=2,
            max_concurrency=2,
            worker=worker,
        )
    )
    assert outputs == [f"done:{item}" for item in inputs]

def test_parallelization_runner_batch_size_enforced() -> None:
    concurrent = 0
    peak = 0

    async def worker(item: int) -> int:
        nonlocal concurrent, peak
        concurrent += 1
        peak = max(peak, concurrent)
        await asyncio.sleep(0.002)
        concurrent -= 1
        return item

    result = asyncio.run(
        run_batched_in_order(
            [1, 2, 3, 4, 5],
            batch_size=2,
            max_concurrency=2,
            worker=worker,
        )
    )
    assert result == [1, 2, 3, 4, 5]
    assert peak <= 2

def test_run_batched_in_order_preserves_order() -> None:
    async def worker(item: int) -> int:
        await asyncio.sleep(0.001 * (5 - item))
        return item * 2

    result = asyncio.run(
        run_batched_in_order(
            [1, 2, 3, 4],
            batch_size=2,
            max_concurrency=2,
            worker=worker,
        )
    )
    assert result == [2, 4, 6, 8]

def test_run_batched_in_order_empty_input() -> None:
    async def worker(item: int) -> int:
        return item

    result = asyncio.run(
        run_batched_in_order(
            [],
            batch_size=2,
            max_concurrency=2,
            worker=worker,
        )
    )
    assert result == []

def test_run_batched_in_order_propagates_exceptions() -> None:
    async def worker(item: int) -> int:
        if item == 2:
            raise RuntimeError("boom")
        return item

    try:
        asyncio.run(
            run_batched_in_order(
                [1, 2, 3],
                batch_size=3,
                max_concurrency=2,
                worker=worker,
            )
        )
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert str(exc) == "boom"

def _words(count: int) -> str:
    return " ".join(f"w{i}" for i in range(count))

def _request(**overrides: object) -> ClinicalSessionRequest:
    payload = {
        "visit_date": "2025-01-01",
        "clinical_input": _words(60),
        "selected_model_providers": ["openai"],
    }
    payload.update(overrides)
    return ClinicalSessionRequest.model_validate(payload)

def test_missing_visit_date_fails() -> None:
    with pytest.raises(HTTPException):
        validate_clinical_session_request(_request(visit_date=None))

def test_empty_clinical_input_fails() -> None:
    with pytest.raises(HTTPException):
        validate_clinical_session_request(_request(clinical_input=""))

def test_fifty_nine_words_fails() -> None:
    with pytest.raises(HTTPException):
        validate_clinical_session_request(_request(clinical_input=_words(59)))

def test_sixty_words_pass() -> None:
    validate_clinical_session_request(_request(clinical_input=_words(60)))

def test_empty_provider_list_fails() -> None:
    with pytest.raises(HTTPException):
        validate_clinical_session_request(_request(selected_model_providers=[]))
