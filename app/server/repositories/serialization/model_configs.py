from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from domain.model_configs import ModelConfigSnapshot
from repositories.database.session import resolve_engine, resolve_session_factory
from repositories.schemas.configuration import ApplicationConfiguration
from repositories.serialization.application_configuration import (
    ApplicationConfigurationSerializer,
)

UNSET = object()

###############################################################################
class ModelConfigSerializer:
    """Persist the validated model configuration as one singleton document."""

    RAG_OPERATIONAL_FIELDS = frozenset(
        {
            "chunk_size",
            "chunk_overlap",
            "embedding_batch_size",
            "use_hybrid_search",
            "use_reranking",
            "retrieval_candidate_count",
            "retrieval_selected_count",
            "reranker_model",
            "hybrid_vector_weight",
            "hybrid_text_weight",
            "vector_stream_batch_size",
            "embedding_offline_mode",
        }
    )
    DEFAULT_OLLAMA_REASONING = False
    DEFAULT_OLLAMA_SEED = 42

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        engine: Engine | None = None,
        session_factory: sessionmaker | None = None,
    ) -> None:
        self.engine = resolve_engine(engine)
        self.session_factory = resolve_session_factory(
            engine=self.engine,
            session_factory=session_factory,
        )
        self.application_configuration = ApplicationConfigurationSerializer(
            engine=self.engine,
            session_factory=self.session_factory,
        )

    # -------------------------------------------------------------------------
    def load_snapshot(self) -> ModelConfigSnapshot:
        with self.session_factory() as db_session:
            row = db_session.get(ApplicationConfiguration, 1)
            if row is None:
                return self.empty_snapshot()
            payload = dict(row.payload)
            return self.snapshot_from_payload(payload, updated_at=row.updated_at)

    # -------------------------------------------------------------------------
    def save_snapshot(
        self,
        *,
        base_snapshot: ModelConfigSnapshot | None = None,
        clinical_model: str | None | object = UNSET,
        text_extraction_model: str | None | object = UNSET,
        use_cloud_models: bool | object = UNSET,
        cloud_provider: str | None | object = UNSET,
        cloud_model: str | None | object = UNSET,
        ollama_reasoning: bool | object = UNSET,
        ollama_seed: int | None | object = UNSET,
        rag_settings: dict[str, object] | object = UNSET,
    ) -> ModelConfigSnapshot:
        current = asdict(base_snapshot or self.load_snapshot())
        current.pop("updated_at", None)
        updates = {
            "clinical_model": clinical_model,
            "text_extraction_model": text_extraction_model,
            "use_cloud_models": use_cloud_models,
            "cloud_provider": cloud_provider,
            "cloud_model": cloud_model,
            "ollama_reasoning": ollama_reasoning,
            "ollama_seed": ollama_seed,
            "rag_settings": rag_settings,
        }
        for key, value in updates.items():
            if value is not UNSET:
                current[key] = value
        current["ollama_reasoning"] = bool(current.get("ollama_reasoning", False))
        current["ollama_seed"] = self.normalize_optional_seed(
            current.get("ollama_seed", self.DEFAULT_OLLAMA_SEED)
        )
        current["rag_settings"] = self.normalize_rag_settings(current.get("rag_settings"))
        saved_payload, updated_at = self.application_configuration.save(
            current,
            return_metadata=True,
        )
        return self.snapshot_from_payload(saved_payload, updated_at=updated_at)

    # -------------------------------------------------------------------------
    @classmethod
    def empty_snapshot(cls) -> ModelConfigSnapshot:
        return ModelConfigSnapshot(
            clinical_model=None,
            text_extraction_model=None,
            use_cloud_models=False,
            cloud_provider=None,
            cloud_model=None,
            ollama_reasoning=cls.DEFAULT_OLLAMA_REASONING,
            ollama_seed=cls.DEFAULT_OLLAMA_SEED,
            rag_settings={},
            updated_at=None,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def snapshot_from_payload(
        cls, payload: dict[str, object], *, updated_at: datetime | None
    ) -> ModelConfigSnapshot:
        return ModelConfigSnapshot(
            clinical_model=cls.normalize_optional_text(payload.get("clinical_model")),
            text_extraction_model=cls.normalize_optional_text(
                payload.get("text_extraction_model")
            ),
            use_cloud_models=bool(payload.get("use_cloud_models", False)),
            cloud_provider=cls.normalize_optional_text(payload.get("cloud_provider")),
            cloud_model=cls.normalize_optional_text(payload.get("cloud_model")),
            ollama_reasoning=bool(payload.get("ollama_reasoning", False)),
            ollama_seed=cls.normalize_optional_seed(
                payload.get("ollama_seed", cls.DEFAULT_OLLAMA_SEED)
            ),
            rag_settings=cls.normalize_rag_settings(payload.get("rag_settings")),
            updated_at=updated_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_optional_text(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    # -------------------------------------------------------------------------
    @classmethod
    def normalize_rag_settings(cls, value: object) -> dict[str, object]:
        source = value if isinstance(value, dict) else {}
        return {key: source[key] for key in cls.RAG_OPERATIONAL_FIELDS if key in source}

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_optional_seed(value: object) -> int | None:
        if value is None:
            return None
        try:
            return max(0, int(str(value).strip()))
        except (TypeError, ValueError):
            return None
