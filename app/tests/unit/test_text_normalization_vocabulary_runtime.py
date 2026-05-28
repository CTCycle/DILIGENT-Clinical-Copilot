from __future__ import annotations

from types import SimpleNamespace

from repositories.schemas.models import Base
from services.text import vocabulary as vocabulary_module
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def test_runtime_upsert_list_and_deactivate_term() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    vocabulary_module.get_text_normalization_snapshot.cache_clear()
    vocabulary_module.get_default_repository = lambda: SimpleNamespace(  # type: ignore[method-assign]
        engine=engine, session_factory=factory
    )
    payload = vocabulary_module.upsert_text_normalization_term_payload(
        category="brand_combo_preference",
        term="Bactrim",
        replacement="trimethoprim sulfamethoxazole",
        source="runtime",
        is_active=True,
    )
    assert payload["replacement"] == "trimethoprim sulfamethoxazole"
    rows = vocabulary_module.list_text_normalization_term_payloads(
        category="brand_combo_preference"
    )
    assert len(rows) == 1
    assert rows[0]["is_active"] is True
    updated = vocabulary_module.deactivate_text_normalization_term_payload(
        category="brand_combo_preference",
        term="Bactrim",
    )
    assert updated is True
    rows = vocabulary_module.list_text_normalization_term_payloads(
        category="brand_combo_preference"
    )
    assert rows[0]["is_active"] is False


def test_seed_mapping_categories_are_loaded() -> None:
    # Smoke check: default snapshot exposes the new mapping fields.
    snapshot = vocabulary_module.get_text_normalization_snapshot()
    assert isinstance(snapshot.lab_marker_aliases, dict)
    assert isinstance(snapshot.brand_combo_preferences, dict)
    assert isinstance(snapshot.knowledge_source_references, dict)
