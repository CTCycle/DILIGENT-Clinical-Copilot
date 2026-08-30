from __future__ import annotations

from dataclasses import asdict
from datetime import datetime

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from domain.model_configs import ModelConfigSnapshot, ReasoningLevel
from repositories.database.session import resolve_engine, resolve_session_factory
from repositories.schemas.configuration import ApplicationConfiguration
from repositories.serialization.application_configuration import (
    ApplicationConfigurationSerializer,
)

UNSET = object()


###############################################################################
class ModelConfigSerializer:
    """Persist the validated model configuration as one singleton document."""

    REQUIRED_ROLE_FIELDS = (
        "clinical_model",
        "text_extraction_model",
        "revision_model",
        "timeline_model",
    )

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
    DEFAULT_REASONING_LEVEL = ReasoningLevel.OFF
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
                raise ValueError(
                    "Persisted model configuration is missing; database initialization "
                    "must seed the canonical configuration before the service starts."
                )
            if not isinstance(row.payload, dict):
                raise ValueError(
                    "Persisted model configuration payload must be an object."
                )
            payload = dict(row.payload)
            return self.snapshot_from_payload(payload, updated_at=row.updated_at)

    # -------------------------------------------------------------------------
    def seed_if_missing(self, payload: dict[str, object]) -> bool:
        """Seed a complete configuration without replacing existing state."""
        self._require_required_roles(payload)
        normalized_payload = dict(payload)
        normalized_payload["reasoning_level"] = self.reasoning_level_from_payload(
            normalized_payload
        )
        normalized_payload["ollama_seed"] = self.normalize_optional_seed(
            normalized_payload.get("ollama_seed", self.DEFAULT_OLLAMA_SEED)
        )
        normalized_payload["rag_settings"] = self.normalize_rag_settings(
            normalized_payload.get("rag_settings")
        )
        return self.application_configuration.save_if_missing(normalized_payload)

    # -------------------------------------------------------------------------
    def save_snapshot(
        self,
        *,
        base_snapshot: ModelConfigSnapshot | None = None,
        clinical_model: str | None | object = UNSET,
        text_extraction_model: str | None | object = UNSET,
        revision_model: str | None | object = UNSET,
        timeline_model: str | None | object = UNSET,
        use_cloud_models: bool | object = UNSET,
        cloud_provider: str | None | object = UNSET,
        cloud_model: str | None | object = UNSET,
        reasoning_level: ReasoningLevel | object = UNSET,
        ollama_seed: int | None | object = UNSET,
        rag_settings: dict[str, object] | object = UNSET,
    ) -> ModelConfigSnapshot:
        current = asdict(base_snapshot or self.load_snapshot())
        current.pop("updated_at", None)
        updates = {
            "clinical_model": clinical_model,
            "text_extraction_model": text_extraction_model,
            "revision_model": revision_model,
            "timeline_model": timeline_model,
            "use_cloud_models": use_cloud_models,
            "cloud_provider": cloud_provider,
            "cloud_model": cloud_model,
            "reasoning_level": reasoning_level,
            "ollama_seed": ollama_seed,
            "rag_settings": rag_settings,
        }
        for key, value in updates.items():
            if value is not UNSET:
                current[key] = value
        self._require_required_roles(current)
        current["reasoning_level"] = self.reasoning_level_from_payload(current)
        current["ollama_seed"] = self.normalize_optional_seed(
            current.get("ollama_seed", self.DEFAULT_OLLAMA_SEED)
        )
        current["rag_settings"] = self.normalize_rag_settings(
            current.get("rag_settings")
        )
        saved_payload, updated_at = self.application_configuration.save(
            current,
            return_metadata=True,
        )
        return self.snapshot_from_payload(saved_payload, updated_at=updated_at)

    # -------------------------------------------------------------------------
    @classmethod
    def snapshot_from_payload(
        cls, payload: dict[str, object], *, updated_at: datetime | None
    ) -> ModelConfigSnapshot:
        cls._require_required_roles(payload)
        return ModelConfigSnapshot(
            clinical_model=cls.normalize_optional_text(payload.get("clinical_model")),
            text_extraction_model=cls.normalize_optional_text(
                payload.get("text_extraction_model")
            ),
            use_cloud_models=bool(payload.get("use_cloud_models", False)),
            cloud_provider=cls.normalize_optional_text(payload.get("cloud_provider")),
            cloud_model=cls.normalize_optional_text(payload.get("cloud_model")),
            revision_model=cls.normalize_optional_text(payload.get("revision_model")),
            timeline_model=cls.normalize_optional_text(payload.get("timeline_model")),
            reasoning_level=cls.reasoning_level_from_payload(payload),
            ollama_seed=cls.normalize_optional_seed(
                payload.get("ollama_seed", cls.DEFAULT_OLLAMA_SEED)
            ),
            rag_settings=cls.normalize_rag_settings(payload.get("rag_settings")),
            updated_at=updated_at,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _require_required_roles(cls, payload: dict[str, object]) -> None:
        missing = [
            field_name
            for field_name in cls.REQUIRED_ROLE_FIELDS
            if cls.normalize_optional_text(payload.get(field_name)) is None
        ]
        if missing:
            raise ValueError(
                "Persisted model configuration is missing required role assignments: "
                + ", ".join(missing)
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
    def reasoning_level_from_payload(cls, payload: dict[str, object]) -> ReasoningLevel:
        raw_level = payload.get("reasoning_level")
        if isinstance(raw_level, ReasoningLevel):
            return raw_level
        if isinstance(raw_level, str):
            try:
                return ReasoningLevel(raw_level.strip().lower())
            except ValueError:
                pass
        return cls.DEFAULT_REASONING_LEVEL

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
        except TypeError, ValueError:
            return None
