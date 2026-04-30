from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def _run_batched_item(
    semaphore: asyncio.Semaphore,
    worker: Callable[[T], Awaitable[R]],
    index: int,
    item: T,
) -> tuple[int, R]:
    async with semaphore:
        result = await worker(item)
        return index, result


async def run_batched_in_order(
    items: Sequence[T],
    *,
    batch_size: int,
    max_concurrency: int,
    worker: Callable[[T], Awaitable[R]],
) -> list[R]:
    if not items:
        return []
    safe_batch_size = max(int(batch_size), 1)
    safe_max_concurrency = max(int(max_concurrency), 1)
    semaphore = asyncio.Semaphore(safe_max_concurrency)
    ordered_results: list[R] = []

    for batch_start in range(0, len(items), safe_batch_size):
        batch_items = items[batch_start : batch_start + safe_batch_size]
        tasks = [
            asyncio.create_task(_run_batched_item(semaphore, worker, offset, item))
            for offset, item in enumerate(batch_items)
        ]
        try:
            completed = await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        completed.sort(key=lambda item: item[0])
        ordered_results.extend(result for _, result in completed)
    return ordered_results
