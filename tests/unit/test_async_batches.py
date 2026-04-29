from __future__ import annotations

import asyncio

from DILIGENT.server.services.async_batches import run_batched_in_order


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

