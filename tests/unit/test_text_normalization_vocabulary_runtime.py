from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from DILIGENT.server.repositories.schemas.models import Base
from DILIGENT.server.repositories.serialization.text_normalization import (
    TextNormalizationVocabularySerializer,
)
from DILIGENT.server.services.text import vocabulary as vocabulary_module


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
