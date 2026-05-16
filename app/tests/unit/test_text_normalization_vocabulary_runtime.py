from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from repositories.schemas.models import Base
from repositories.serialization.text_normalization import (
    TextNormalizationVocabularySerializer,
)
from services.text import vocabulary as vocabulary_module


def test_runtime_upsert_list_and_deactivate_term() -> None:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    serializer = TextNormalizationVocabularySerializer(
        engine=engine, session_factory=factory
    )
    session = factory()
    try:
        serializer.upsert_term(
            session,
            category="brand_combo_preference",
            term="Bactrim",
            replacement="trimethoprim sulfamethoxazole",
            source="runtime",
        )
        session.commit()
        rows = serializer.list_terms(session, category="brand_combo_preference")
        assert len(rows) == 1
        assert rows[0].replacement == "trimethoprim sulfamethoxazole"
        updated = serializer.set_term_active(
            session,
            category="brand_combo_preference",
            term="Bactrim",
            is_active=False,
        )
        assert updated is True
        session.commit()
        rows = serializer.list_terms(session, category="brand_combo_preference")
        assert rows[0].is_active is False
    finally:
        session.close()


def test_seed_mapping_categories_are_loaded() -> None:
    # Smoke check: default snapshot exposes the new mapping fields.
    snapshot = vocabulary_module.get_text_normalization_snapshot()
    assert isinstance(snapshot.lab_marker_aliases, dict)
    assert isinstance(snapshot.brand_combo_preferences, dict)
    assert isinstance(snapshot.knowledge_source_references, dict)


def test_seeded_catalog_is_skipped_when_already_present(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "matching_stopwords": ["tablet", "capsule"],
        "query_aliases": [
            {"term": "Bactrim", "replacement": "trimethoprim sulfamethoxazole"}
        ],
    }
    monkeypatch.setattr(
        "repositories.serialization.text_normalization.CatalogLoader.load_catalog",
        lambda _name: payload,
    )

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    serializer = TextNormalizationVocabularySerializer(
        engine=engine, session_factory=factory
    )
    serializer.ensure_seeded()

    statements: list[str] = []

    def collect_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(str(statement))

    event.listen(engine, "before_cursor_execute", collect_statement)
    try:
        serializer.ensure_seeded()
    finally:
        event.remove(engine, "before_cursor_execute", collect_statement)

    inserts_or_updates = [
        statement
        for statement in statements
        if "INSERT INTO text_normalization_terms" in statement
        or "UPDATE text_normalization_terms" in statement
    ]
    assert inserts_or_updates == []


def test_seeded_catalog_reseeds_when_replacement_changes(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "query_aliases": [
            {"term": "Bactrim", "replacement": "trimethoprim sulfamethoxazole"}
        ],
    }
    monkeypatch.setattr(
        "repositories.serialization.text_normalization.CatalogLoader.load_catalog",
        lambda _name: payload,
    )

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    serializer = TextNormalizationVocabularySerializer(
        engine=engine, session_factory=factory
    )
    serializer.ensure_seeded()
    with factory() as session:
        row = serializer.get_term(
            session,
            category="query_alias",
            term="Bactrim",
        )
        assert row is not None
        row.replacement = "outdated"
        session.commit()

    serializer.ensure_seeded()

    with factory() as session:
        row = serializer.get_term(
            session,
            category="query_alias",
            term="Bactrim",
        )
        assert row is not None
        assert row.replacement == "trimethoprim sulfamethoxazole"

