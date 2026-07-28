from __future__ import annotations

import pandas as pd

from repositories.drug_catalog_repository import DrugCatalogRepository

###############################################################################
def test_stream_drugs_catalog_uses_paged_repository_function(monkeypatch) -> None:
    repository = object.__new__(DrugCatalogRepository)
    calls: list[tuple[int, int | None]] = []

    def fake_get_drugs_catalog(
        observed_repository: DrugCatalogRepository,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        assert observed_repository is repository
        calls.append((offset, limit))
        if offset == 0:
            return pd.DataFrame([{"rxcui": "1", "name": "Example"}])
        return pd.DataFrame()

    monkeypatch.setattr(
        DrugCatalogRepository, "get_drugs_catalog", fake_get_drugs_catalog
    )

    pages = list(repository.stream_drugs_catalog(page_size=1000))

    assert len(pages) == 1
    assert pages[0].to_dict(orient="records") == [
        {"rxcui": "1", "name": "Example"}
    ]
    assert calls == [(0, 1000), (1000, 1000), (1000, 1)]
