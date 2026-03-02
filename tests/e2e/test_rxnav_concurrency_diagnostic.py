from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass

import httpx
import pytest

DRUGS_ENDPOINT = "https://rxnav.nlm.nih.gov/REST/drugs.json"
PROPERTY_ENDPOINT = "https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/property.json"
DRUG_QUERIES = [
    "acetaminophen",
    "ibuprofen",
    "metformin",
    "atorvastatin",
    "amoxicillin",
    "omeprazole",
]
RXCUI_QUERIES = ["161", "857004", "860975", "617320", "308192", "197361"]


###############################################################################
@dataclass(slots=True)
class RequestMetric:
    status_code: int
    elapsed_seconds: float
    error: str | None


# -----------------------------------------------------------------------------
async def perform_request(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    payload: str,
) -> RequestMetric:
    started = time.perf_counter()
    try:
        if endpoint == "drugs":
            response = await client.get(
                DRUGS_ENDPOINT,
                params={"name": payload, "expand": "psn"},
            )
        else:
            response = await client.get(
                PROPERTY_ENDPOINT.format(rxcui=payload),
                params={"propName": "RxNorm Synonym"},
            )
        elapsed = time.perf_counter() - started
        return RequestMetric(
            status_code=response.status_code,
            elapsed_seconds=elapsed,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        return RequestMetric(
            status_code=0,
            elapsed_seconds=elapsed,
            error=str(exc),
        )


# -----------------------------------------------------------------------------
def make_workload(total_requests: int) -> list[tuple[str, str]]:
    payload: list[tuple[str, str]] = []
    while len(payload) < total_requests:
        for query in DRUG_QUERIES:
            payload.append(("drugs", query))
            if len(payload) >= total_requests:
                return payload
        for rxcui in RXCUI_QUERIES:
            payload.append(("property", rxcui))
            if len(payload) >= total_requests:
                return payload
    return payload


# -----------------------------------------------------------------------------
async def run_probe(
    *,
    concurrency: int,
    total_requests: int,
    timeout_seconds: float,
) -> list[RequestMetric]:
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    semaphore = asyncio.Semaphore(concurrency)
    metrics: list[RequestMetric] = []
    workload = make_workload(total_requests)

    async with httpx.AsyncClient(timeout=timeout_seconds, limits=limits) as client:
        async def run_single(endpoint: str, payload: str) -> None:
            async with semaphore:
                metrics.append(
                    await perform_request(
                        client,
                        endpoint=endpoint,
                        payload=payload,
                    )
                )

        await asyncio.gather(
            *(run_single(endpoint, payload) for endpoint, payload in workload)
        )
    return metrics


# -----------------------------------------------------------------------------
def summarize(metrics: list[RequestMetric]) -> dict[str, object]:
    status_counts: dict[int, int] = {}
    error_counts: dict[str, int] = {}
    latencies: list[float] = []
    for metric in metrics:
        latencies.append(metric.elapsed_seconds)
        if metric.error is not None:
            error_counts[metric.error] = error_counts.get(metric.error, 0) + 1
            continue
        status_counts[metric.status_code] = status_counts.get(metric.status_code, 0) + 1
    successes = sum(
        count for status, count in status_counts.items() if 200 <= status < 300
    )
    return {
        "total": len(metrics),
        "successes": successes,
        "http_429": status_counts.get(429, 0),
        "status_counts": status_counts,
        "error_counts": error_counts,
        "avg_ms": statistics.fmean(latencies) * 1000 if latencies else 0.0,
    }


# -----------------------------------------------------------------------------
def enabled() -> bool:
    return os.environ.get("RUN_RXNAV_CONCURRENCY_DIAGNOSTIC", "").strip() == "1"


# -----------------------------------------------------------------------------
@pytest.mark.skipif(
    not enabled(),
    reason="Set RUN_RXNAV_CONCURRENCY_DIAGNOSTIC=1 to run RxNav concurrency diagnostics.",
)
def test_rxnav_concurrency_diagnostic() -> None:
    levels = [1, 4, 8, 12, 16]
    total_requests = 60
    timeout_seconds = 12.0
    summaries: list[tuple[int, dict[str, object]]] = []

    for level in levels:
        metrics = asyncio.run(
            run_probe(
                concurrency=level,
                total_requests=total_requests,
                timeout_seconds=timeout_seconds,
            )
        )
        summary = summarize(metrics)
        summaries.append((level, summary))

    baseline = summaries[0][1]
    baseline_successes = int(baseline["successes"])  # type: ignore[arg-type]
    assert baseline_successes > 0, (
        "RxNav baseline probe had no successful responses. "
        f"Summary: {baseline}"
    )

    for level, summary in summaries:
        errors = summary["error_counts"]
        assert int(summary["http_429"]) == 0, (
            f"RxNav returned HTTP 429 at concurrency={level}. Summary: {summary}"
        )
        assert int(summary["successes"]) > 0, (
            f"RxNav had no successes at concurrency={level}. "
            f"Summary: {summary}. Errors: {errors}"
        )
