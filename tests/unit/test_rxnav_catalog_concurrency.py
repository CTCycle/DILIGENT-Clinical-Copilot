from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from DILIGENT.server.services.updater.rxnav import RxNavDrugCatalogBuilder


###############################################################################
class RxClientStub:
    def __init__(self) -> None:
        self.timeout = 2.0
        self.max_concurrency = 8
        self.term_calls: list[str] = []
        self.synonym_calls: list[str] = []

    # -------------------------------------------------------------------------
    def _build_limits(self) -> httpx.Limits:
        return httpx.Limits(
            max_connections=self.max_concurrency,
            max_keepalive_connections=self.max_concurrency,
        )

    # -------------------------------------------------------------------------
    async def fetch_drug_terms_async(
        self,
        raw_name: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[str]:
        self.term_calls.append(raw_name)
        return [raw_name.title()]

    # -------------------------------------------------------------------------
    async def fetch_rxcui_synonyms_async(
        self,
        rxcui: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> list[str]:
        self.synonym_calls.append(rxcui)
        return [f"Synonym-{rxcui}"]


###############################################################################
class SerializerStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    def upsert_drugs_catalog_records(self, records: Any, **kwargs: Any) -> None:
        self.calls.append(
            {
                "records_type": type(records).__name__,
                "kwargs": kwargs,
            }
        )


# -----------------------------------------------------------------------------
def test_prefetch_concept_queries_fetches_unique_cache_misses() -> None:
    rx_client = RxClientStub()
    builder = RxNavDrugCatalogBuilder(rx_client=rx_client)
    concepts: list[dict[str, Any]] = [
        {"fullName": "Acetaminophen 500 MG Tablet", "rxcui": "161"},
        {"fullName": "Acetaminophen 325 MG Tablet", "rxcui": "161"},
        {"fullName": "Ibuprofen [Advil] 200 MG Tablet", "rxcui": "5640"},
    ]

    builder.prefetch_concept_queries(concepts)

    assert set(rx_client.synonym_calls) == {"161", "5640"}
    assert len(rx_client.term_calls) == len(set(rx_client.term_calls))
    assert "Acetaminophen 500 MG Tablet" in rx_client.term_calls
    assert "Acetaminophen 325 MG Tablet" in rx_client.term_calls
    assert "Ibuprofen [Advil] 200 MG Tablet" in rx_client.term_calls
    assert "acetaminophen" in builder.alias_cache
    assert "ibuprofen" in builder.alias_cache
    assert "161" in builder.rxcui_cache
    assert "5640" in builder.rxcui_cache


# -----------------------------------------------------------------------------
def test_persist_catalog_prefetches_by_batch() -> None:
    builder = RxNavDrugCatalogBuilder(rx_client=RxClientStub())
    builder.BATCH_SIZE = 2
    prefetch_batch_sizes: list[int] = []
    persisted_batch_sizes: list[int] = []

    def prefetch_stub(concepts: list[dict[str, Any]]) -> None:
        prefetch_batch_sizes.append(len(concepts))

    def sanitize_stub(concept: dict[str, Any]) -> dict[str, Any]:
        return {
            "rxcui": str(concept.get("rxcui")),
            "term_type": "SCD",
            "raw_name": str(concept.get("fullName")),
            "name": "stub",
            "brand_names": None,
            "synonyms": [],
        }

    def persist_batch_stub(batch: list[dict[str, Any]]) -> None:
        persisted_batch_sizes.append(len(batch))

    builder.prefetch_concept_queries = prefetch_stub  # type: ignore[method-assign]
    builder.sanitize_concept = sanitize_stub  # type: ignore[method-assign]
    builder.persist_batch = persist_batch_stub  # type: ignore[method-assign]

    payload = [
        {"fullName": "Drug Alpha 10 MG Tablet", "rxcui": "1001"},
        {"fullName": "Drug Beta 20 MG Tablet", "rxcui": "1002"},
        {"fullName": "Drug Gamma 30 MG Tablet", "rxcui": "1003"},
    ]
    chunks = iter([json.dumps(payload).encode("utf-8")])

    result = builder.persist_catalog(chunks)

    assert result["count"] == 3
    assert prefetch_batch_sizes == [2, 1]
    assert persisted_batch_sizes == [2, 1]


# -----------------------------------------------------------------------------
def test_curated_aliases_are_loaded_and_forwarded_to_serializer(
    tmp_path: Path,
) -> None:
    curated_file = tmp_path / "curated_aliases.json"
    curated_file.write_text(
        json.dumps(
            {
                "aliases": [
                    {
                        "canonical_name": "Metformin",
                        "alias_kind": "synonym",
                        "aliases": ["Metformina"],
                    },
                    {
                        "canonical_name": "Trimethoprim Sulfamethoxazole",
                        "alias_kind": "brand",
                        "aliases": ["Bactrim"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    serializer_stub = SerializerStub()
    builder = RxNavDrugCatalogBuilder(
        rx_client=RxClientStub(),
        serializer=serializer_stub,  # type: ignore[arg-type]
        curated_aliases_path=str(curated_file),
    )
    assert builder.curated_aliases_by_canonical["metformin"] == [
        ("Metformina", "synonym")
    ]
    assert builder.curated_aliases_by_canonical["trimethoprim sulfamethoxazole"] == [
        ("Bactrim", "brand")
    ]

    builder.persist_batch(
        [
            {
                "rxcui": "860975",
                "term_type": "SCD",
                "raw_name": "Metformin 500 MG Tablet",
                "name": "Metformin",
                "brand_names": None,
                "synonyms": [],
            }
        ]
    )

    assert len(serializer_stub.calls) == 1
    kwargs = serializer_stub.calls[0]["kwargs"]
    assert "curated_aliases_by_canonical" in kwargs
    assert kwargs["curated_aliases_by_canonical"]["metformin"] == [
        ("Metformina", "synonym")
    ]
