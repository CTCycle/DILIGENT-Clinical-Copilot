from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session, sessionmaker

from domain.model_configs import ModelConfigSnapshot
from repositories.database.session import (
    resolve_engine,
    resolve_session_factory,
)
from repositories.serialization.application_configuration import (
    ApplicationConfigurationSerializer,
)
from repositories.schemas.models import ApplicationConfiguration
from repositories.queries.model_config import (
    ModelConfigRepositoryQueries,
)
from repositories.schemas.models import ModelSelection, RuntimeSetting

ModelRoleType = Literal["clinical", "text_extraction", "cloud"]
UNSET = object()

###############################################################################
class ModelConfigSerializer:
    OLLAMA_TEMPERATURE_KEY = "ollama_temperature"
    CLOUD_TEMPERATURE_KEY = "cloud_temperature"
    OLLAMA_REASONING_KEY = "ollama_reasoning"
    OLLAMA_SEED_KEY = "ollama_seed"
    RAG_SETTING_PREFIX = "rag."
    RAG_SETTING_KEYS = {
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
        "embedding_backend",
        "ollama_embedding_model",
        "hf_embedding_model",
        "cloud_provider",
        "cloud_embedding_model",
        "use_cloud_embeddings",
        "reset_vector_collection",
        "vector_stream_batch_size",
        "embedding_max_workers",
    }
    DEFAULT_OLLAMA_TEMPERATURE = 0.7
    DEFAULT_CLOUD_TEMPERATURE = 0.7
    DEFAULT_OLLAMA_REASONING = False
    DEFAULT_OLLAMA_SEED = 42

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        engine=None,
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
        db_session = self.session_factory()
        try:
            canonical = db_session.get(ApplicationConfiguration, 1)
            if canonical is not None and all(
                payload_key in canonical.payload
                for payload_key in ("clinical_model", "text_extraction_model")
            ) and (
                bool(canonical.payload.get("use_cloud_models", False))
                or all(
                    self.normalize_optional_text(canonical.payload.get(payload_key))
                    for payload_key in ("clinical_model", "text_extraction_model")
                )
            ):
                payload = dict(canonical.payload)
                payload["updated_at"] = canonical.updated_at
                return ModelConfigSnapshot(
                    clinical_model=self.normalize_optional_text(
                        payload.get("clinical_model")
                    ),
                    text_extraction_model=self.normalize_optional_text(
                        payload.get("text_extraction_model")
                    ),
                    use_cloud_models=bool(payload.get("use_cloud_models", False)),
                    cloud_provider=(
                        self.normalize_optional_text(payload.get("cloud_provider"))
                        or "openai"
                    ),
                    cloud_model=self.normalize_optional_text(payload.get("cloud_model")),
                    ollama_temperature=self.normalize_temperature(
                        payload.get("ollama_temperature", self.DEFAULT_OLLAMA_TEMPERATURE)
                    ),
                    cloud_temperature=self.normalize_temperature(
                        payload.get("cloud_temperature", self.DEFAULT_CLOUD_TEMPERATURE)
                    ),
                    ollama_reasoning=bool(payload.get("ollama_reasoning", False)),
                    ollama_seed=self.normalize_optional_seed(payload.get("ollama_seed")),
                    rag_settings=(
                        payload.get("rag_settings")
                        if isinstance(payload.get("rag_settings"), dict)
                        else {}
                    ),
                    updated_at=canonical.updated_at,
                )
            rows = (
                db_session.execute(ModelConfigRepositoryQueries.select_all())
                .scalars()
                .all()
            )
            runtime_rows = (
                db_session.execute(
                    ModelConfigRepositoryQueries.select_runtime_settings()
                )
                .scalars()
                .all()
            )
            snapshot = self.build_snapshot_with_runtime(rows, runtime_rows)
            if rows or runtime_rows:
                self.application_configuration.save(asdict(snapshot))
            return snapshot
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    def save_snapshot(
        self,
        *,
        clinical_model: str | None | object = UNSET,
        text_extraction_model: str | None | object = UNSET,
        use_cloud_models: bool | object = UNSET,
        cloud_provider: str | None | object = UNSET,
        cloud_model: str | None | object = UNSET,
        ollama_temperature: float | object = UNSET,
        cloud_temperature: float | object = UNSET,
        ollama_reasoning: bool | object = UNSET,
        ollama_seed: int | None | object = UNSET,
        rag_settings: dict[str, object] | object = UNSET,
    ) -> ModelConfigSnapshot:
        db_session = self.session_factory()
        now = datetime.now()
        try:
            rows = (
                db_session.execute(ModelConfigRepositoryQueries.select_all())
                .scalars()
                .all()
            )
            role_map = {str(row.role_type): row for row in rows}

            if clinical_model is not UNSET:
                normalized_clinical_model = self.normalize_optional_text(clinical_model)
                clinical_row = self.ensure_role_row(db_session, role_map, "clinical")
                clinical_row.model_name = normalized_clinical_model
                clinical_row.provider = None
                clinical_row.is_active = normalized_clinical_model is not None
                clinical_row.updated_at = now

            if text_extraction_model is not UNSET:
                normalized_text_extraction_model = self.normalize_optional_text(
                    text_extraction_model
                )
                text_extraction_row = self.ensure_role_row(
                    db_session, role_map, "text_extraction"
                )
                text_extraction_row.model_name = normalized_text_extraction_model
                text_extraction_row.provider = None
                text_extraction_row.is_active = (
                    normalized_text_extraction_model is not None
                )
                text_extraction_row.updated_at = now

            cloud_fields_changed = any(
                field is not UNSET
                for field in (use_cloud_models, cloud_provider, cloud_model)
            )
            if cloud_fields_changed:
                cloud_row = self.ensure_role_row(db_session, role_map, "cloud")
                if cloud_provider is not UNSET:
                    cloud_row.provider = self.normalize_optional_text(cloud_provider)
                if cloud_model is not UNSET:
                    cloud_row.model_name = self.normalize_optional_text(cloud_model)
                if use_cloud_models is not UNSET:
                    cloud_row.is_active = bool(use_cloud_models)
                cloud_row.updated_at = now

            if ollama_temperature is not UNSET:
                self.upsert_runtime_setting(
                    db_session=db_session,
                    key=self.OLLAMA_TEMPERATURE_KEY,
                    value=f"{self.normalize_temperature(ollama_temperature):.2f}",
                    updated_at=now,
                )
            if cloud_temperature is not UNSET:
                self.upsert_runtime_setting(
                    db_session=db_session,
                    key=self.CLOUD_TEMPERATURE_KEY,
                    value=f"{self.normalize_temperature(cloud_temperature):.2f}",
                    updated_at=now,
                )
            if ollama_reasoning is not UNSET:
                self.upsert_runtime_setting(
                    db_session=db_session,
                    key=self.OLLAMA_REASONING_KEY,
                    value=self.normalize_bool_text(ollama_reasoning),
                    updated_at=now,
                )
            if ollama_seed is not UNSET:
                normalized_ollama_seed = self.normalize_optional_seed(ollama_seed)
                self.upsert_runtime_setting(
                    db_session=db_session,
                    key=self.OLLAMA_SEED_KEY,
                    value=""
                    if normalized_ollama_seed is None
                    else str(normalized_ollama_seed),
                    updated_at=now,
                )
            if rag_settings is not UNSET:
                self.upsert_rag_settings(
                    db_session=db_session,
                    settings=rag_settings if isinstance(rag_settings, dict) else {},
                    updated_at=now,
                )

            db_session.commit()
            refreshed_rows = (
                db_session.execute(ModelConfigRepositoryQueries.select_all())
                .scalars()
                .all()
            )
            refreshed_runtime_rows = (
                db_session.execute(
                    ModelConfigRepositoryQueries.select_runtime_settings()
                )
                .scalars()
                .all()
            )
            snapshot = self.build_snapshot_with_runtime(
                refreshed_rows, refreshed_runtime_rows
            )
            self.application_configuration.save(asdict(snapshot))
            return snapshot
        except Exception:
            db_session.rollback()
            raise
        finally:
            db_session.close()

    # -------------------------------------------------------------------------
    @staticmethod
    def ensure_role_row(
        db_session: Session,
        role_map: dict[str, ModelSelection],
        role_type: ModelRoleType,
    ) -> ModelSelection:
        existing = role_map.get(role_type)
        if existing is not None:
            return existing
        created = ModelSelection(
            role_type=role_type,
            provider=None,
            model_name=None,
            is_active=False,
        )
        db_session.add(created)
        role_map[role_type] = created
        return created

    # -------------------------------------------------------------------------
    @staticmethod
    def upsert_runtime_setting(
        *,
        db_session: Session,
        key: str,
        value: str,
        updated_at: datetime,
    ) -> None:
        existing = (
            db_session.query(RuntimeSetting)
            .filter(RuntimeSetting.setting_key == key)
            .one_or_none()
        )
        if existing is None:
            db_session.add(
                RuntimeSetting(
                    setting_key=key,
                    setting_value=value,
                    updated_at=updated_at,
                )
            )
            return
        existing.setting_value = value
        existing.updated_at = updated_at

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_temperature(value: object) -> float:
        if not isinstance(value, (int, float, str)):
            return 0.7
        try:
            parsed = float(value)
        except ValueError:
            parsed = 0.7
        return round(max(0.0, min(2.0, parsed)), 2)

    # -------------------------------------------------------------------------
    @classmethod
    def read_runtime_temperatures(
        cls, rows: list[RuntimeSetting]
    ) -> tuple[float, float]:
        values = {str(row.setting_key): row.setting_value for row in rows}
        return (
            cls.normalize_temperature(
                values.get(cls.OLLAMA_TEMPERATURE_KEY)
                if cls.OLLAMA_TEMPERATURE_KEY in values
                else cls.DEFAULT_OLLAMA_TEMPERATURE
            ),
            cls.normalize_temperature(
                values.get(cls.CLOUD_TEMPERATURE_KEY)
                if cls.CLOUD_TEMPERATURE_KEY in values
                else cls.DEFAULT_CLOUD_TEMPERATURE
            ),
        )

    # -------------------------------------------------------------------------
    @classmethod
    def read_runtime_reasoning(cls, rows: list[RuntimeSetting]) -> bool:
        values = {str(row.setting_key): row.setting_value for row in rows}
        raw_value = values.get(cls.OLLAMA_REASONING_KEY)
        if raw_value is None:
            return bool(cls.DEFAULT_OLLAMA_REASONING)
        normalized = str(raw_value).strip().lower()
        return normalized in {"1", "true", "yes", "on"}

    # -------------------------------------------------------------------------
    @classmethod
    def read_runtime_seed(cls, rows: list[RuntimeSetting]) -> int | None:
        values = {str(row.setting_key): row.setting_value for row in rows}
        raw_value = values.get(cls.OLLAMA_SEED_KEY, cls.DEFAULT_OLLAMA_SEED)
        if raw_value in {None, ""}:
            return None
        try:
            return max(0, int(str(raw_value)))
        except TypeError, ValueError:
            return cls.DEFAULT_OLLAMA_SEED

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_bool_text(value: object) -> str:
        return "true" if bool(value) else "false"

    # -------------------------------------------------------------------------
    @classmethod
    def upsert_rag_settings(
        cls,
        *,
        db_session: Session,
        settings: dict[str, object],
        updated_at: datetime,
    ) -> None:
        for key, value in settings.items():
            if key not in cls.RAG_SETTING_KEYS:
                continue
            cls.upsert_runtime_setting(
                db_session=db_session,
                key=f"{cls.RAG_SETTING_PREFIX}{key}",
                value=cls.serialize_rag_setting_value(value),
                updated_at=updated_at,
            )

    # -------------------------------------------------------------------------
    @staticmethod
    def serialize_rag_setting_value(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return ""
        return str(value)

    # -------------------------------------------------------------------------
    @classmethod
    def read_runtime_rag_settings(cls, rows: list[RuntimeSetting]) -> dict[str, object]:
        settings: dict[str, object] = {}
        for row in rows:
            raw_key = str(row.setting_key)
            if not raw_key.startswith(cls.RAG_SETTING_PREFIX):
                continue
            key = raw_key.removeprefix(cls.RAG_SETTING_PREFIX)
            if key not in cls.RAG_SETTING_KEYS:
                continue
            settings[key] = row.setting_value
        return settings

    # -------------------------------------------------------------------------
    @staticmethod
    def build_snapshot(rows: list[ModelSelection]) -> ModelConfigSnapshot:
        return ModelConfigSerializer.build_snapshot_with_runtime(rows, [])

    # -------------------------------------------------------------------------
    @classmethod
    def build_snapshot_with_runtime(
        cls,
        rows: list[ModelSelection],
        runtime_rows: list[RuntimeSetting],
    ) -> ModelConfigSnapshot:
        role_map = {str(row.role_type): row for row in rows}
        clinical = role_map.get("clinical")
        text_extraction = role_map.get("text_extraction")
        cloud = role_map.get("cloud")
        ollama_temperature, cloud_temperature = cls.read_runtime_temperatures(
            runtime_rows
        )
        ollama_reasoning = cls.read_runtime_reasoning(runtime_rows)
        ollama_seed = cls.read_runtime_seed(runtime_rows)
        rag_settings = cls.read_runtime_rag_settings(runtime_rows)
        updated_values = [
            row.updated_at
            for row in role_map.values()
            if isinstance(row.updated_at, datetime)
        ]
        updated_values.extend(
            row.updated_at
            for row in runtime_rows
            if isinstance(row.updated_at, datetime)
        )
        updated_at = max(updated_values) if updated_values else None
        return ModelConfigSnapshot(
            clinical_model=ModelConfigSerializer.normalize_optional_text(
                clinical.model_name if clinical else None
            ),
            text_extraction_model=ModelConfigSerializer.normalize_optional_text(
                text_extraction.model_name if text_extraction else None
            ),
            use_cloud_models=bool(cloud.is_active) if cloud is not None else False,
            cloud_provider=ModelConfigSerializer.normalize_optional_text(
                cloud.provider if cloud else None
            ),
            cloud_model=ModelConfigSerializer.normalize_optional_text(
                cloud.model_name if cloud else None
            ),
            ollama_temperature=ollama_temperature,
            cloud_temperature=cloud_temperature,
            ollama_reasoning=ollama_reasoning,
            ollama_seed=ollama_seed,
            rag_settings=rag_settings,
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
    @staticmethod
    def normalize_optional_seed(value: object) -> int | None:
        if value is None:
            return None
        try:
            return max(0, int(str(value).strip()))
        except TypeError, ValueError:
            return None
