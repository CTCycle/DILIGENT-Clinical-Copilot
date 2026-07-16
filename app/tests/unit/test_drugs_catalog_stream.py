from __future__ import annotations

import pandas as pd

from repositories.serialization import evidence_data

###############################################################################
def test_stream_drugs_catalog_uses_paged_repository_function(monkeypatch) -> None:
    serializer = object()
    calls: list[tuple[int, int | None]] = []

    def fake_get_drugs_catalog(
        observed_serializer: object,
        *,
        offset: int = 0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        assert observed_serializer is serializer
        calls.append((offset, limit))
        if offset == 0:
            return pd.DataFrame([{"rxcui": "1", "name": "Example"}])
        return pd.DataFrame()

    monkeypatch.setattr(evidence_data, "get_drugs_catalog", fake_get_drugs_catalog)

    pages = list(evidence_data.stream_drugs_catalog(serializer, page_size=1000))

    assert len(pages) == 1
    assert pages[0].to_dict(orient="records") == [
        {"rxcui": "1", "name": "Example"}
    ]
    assert calls == [(0, 1000), (1000, 1000), (1000, 1)]
