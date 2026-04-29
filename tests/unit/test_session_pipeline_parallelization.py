from __future__ import annotations

import asyncio

from DILIGENT.server.services.async_batches import run_batched_in_order


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
