from __future__ import annotations

from types import SimpleNamespace

import pytest

from configurations.startup import get_server_settings
from services import startup_validation

###############################################################################
class _FakeModelConfigService:

    # -------------------------------------------------------------------------
    def __init__(self, *, clinical_model: str, text_extraction_model: str) -> None:
        self._snapshot = SimpleNamespace(
            clinical_model=clinical_model,
            text_extraction_model=text_extraction_model,
            revision_model=clinical_model,
            timeline_model=text_extraction_model,
        )

    # -------------------------------------------------------------------------
    def load_current_snapshot(self) -> SimpleNamespace:
        return self._snapshot

###############################################################################
def test_run_startup_validations_requires_reference_catalogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        startup_validation,
        "get_reference_catalog_snapshot",
        lambda: SimpleNamespace(entries_by_scope={}),
    )
    monkeypatch.setattr(
        startup_validation,
        "ModelConfigService",
        lambda: _FakeModelConfigService(
            clinical_model="clinical-model",
            text_extraction_model="parser-model",
        ),
    )

    with pytest.raises(RuntimeError, match="Reference catalogs"):
        startup_validation.run_startup_validations(get_server_settings())


