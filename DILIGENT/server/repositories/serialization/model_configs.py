from __future__ import annotations

from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session, sessionmaker

from DILIGENT.server.configurations.runtime_state import LLMRuntimeConfig
from DILIGENT.server.domain.model_configs import ModelConfigSnapshot
from DILIGENT.server.repositories.queries.data import DataRepositoryQueries
from DILIGENT.server.repositories.queries.model_config import ModelConfigRepositoryQueries
from DILIGENT.server.repositories.schemas.models import ModelSelection, RuntimeSetting

ModelRoleType = Literal["clinical", "text_extraction", "cloud"]
UNSET = object()


###############################################################################
class ModelConfigSerializer:
    OLLAMA_TEMPERATURE_KEY = "ollama_temperature"
    CLOUD_TEMPERATURE_KEY = "cloud_temperature"

    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.engine = self.queries.database.backend.engine  # type: ignore[attr-defined]
        self.session_factory = sessionmaker(
            bind=self.engine,
            future=True,
        )

    # -------------------------------------------------------------------------
    def load_snapshot(self) -> ModelConfigSnapshot:
        db_session = self.session_factory()
        try:
            rows = db_session.execute(ModelConfigRepositoryQueries.select_all()).scalars().all()
            runtime_rows = (
                db_session.execute(ModelConfigRepositoryQueries.select_runtime_settings())
                .scalars()
                .all()
            )
            return self.build_snapshot_with_runtime(rows, runtime_rows)
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
    ) -> ModelConfigSnapshot:
        db_session = self.session_factory()
        now = datetime.now()
        try:
            rows = db_session.execute(ModelConfigRepositoryQueries.select_all()).scalars().all()
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
                text_extraction_row.is_active = normalized_text_extraction_model is not None
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
                    value=self.normalize_temperature(ollama_temperature),
                    updated_at=now,
                )
            if cloud_temperature is not UNSET:
                self.upsert_runtime_setting(
                    db_session=db_session,
                    key=self.CLOUD_TEMPERATURE_KEY,
                    value=self.normalize_temperature(cloud_temperature),
                    updated_at=now,
                )

            db_session.commit()
            refreshed_rows = (
                db_session.execute(ModelConfigRepositoryQueries.select_all()).scalars().all()
            )
            refreshed_runtime_rows = (
                db_session.execute(ModelConfigRepositoryQueries.select_runtime_settings())
                .scalars()
                .all()
            )
            return self.build_snapshot_with_runtime(refreshed_rows, refreshed_runtime_rows)
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
        value: float,
        updated_at: datetime,
    ) -> None:
        existing = (
            db_session.query(RuntimeSetting)
            .filter(RuntimeSetting.setting_key == key)
            .one_or_none()
        )
        serialized = f"{value:.2f}"
        if existing is None:
            db_session.add(
                RuntimeSetting(
                    setting_key=key,
                    setting_value=serialized,
                    updated_at=updated_at,
                )
            )
            return
        existing.setting_value = serialized
        existing.updated_at = updated_at

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_temperature(value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
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
                else LLMRuntimeConfig.get_ollama_temperature()
            ),
            cls.normalize_temperature(
                values.get(cls.CLOUD_TEMPERATURE_KEY)
                if cls.CLOUD_TEMPERATURE_KEY in values
                else LLMRuntimeConfig.get_cloud_temperature()
            ),
        )

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
        ollama_temperature, cloud_temperature = cls.read_runtime_temperatures(runtime_rows)
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
            updated_at=updated_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_optional_text(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None
