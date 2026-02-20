from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from DILIGENT.common.utils.logger import logger
from DILIGENT.server.repositories.queries.data import DataRepositoryQueries
from DILIGENT.server.repositories.schemas.models import ModelSelection

ModelRoleType = Literal["clinical", "text_extraction", "cloud"]
UNSET = object()


###############################################################################
@dataclass(frozen=True)
class ModelConfigSnapshot:
    clinical_model: str | None
    text_extraction_model: str | None
    use_cloud_models: bool
    cloud_provider: str | None
    cloud_model: str | None
    updated_at: datetime | None


###############################################################################
class ModelConfigSerializer:
    def __init__(self, queries: DataRepositoryQueries | None = None) -> None:
        self.queries = queries or DataRepositoryQueries()
        self.engine = self.queries.database.backend.engine  # type: ignore[attr-defined]
        self.session_factory = sessionmaker(
            bind=self.engine,
            future=True,
        )

    # -------------------------------------------------------------------------
    def load_snapshot(self) -> ModelConfigSnapshot:
        self.ensure_table()
        db_session = self.session_factory()
        try:
            rows = db_session.execute(select(ModelSelection)).scalars().all()
            return self.build_snapshot(rows)
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
    ) -> ModelConfigSnapshot:
        self.ensure_table()
        db_session = self.session_factory()
        now = datetime.now()
        try:
            rows = db_session.execute(select(ModelSelection)).scalars().all()
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

            db_session.commit()
            refreshed_rows = db_session.execute(select(ModelSelection)).scalars().all()
            return self.build_snapshot(refreshed_rows)
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
    def build_snapshot(rows: list[ModelSelection]) -> ModelConfigSnapshot:
        role_map = {str(row.role_type): row for row in rows}
        clinical = role_map.get("clinical")
        text_extraction = role_map.get("text_extraction")
        cloud = role_map.get("cloud")
        updated_values = [
            row.updated_at
            for row in role_map.values()
            if isinstance(row.updated_at, datetime)
        ]
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
    def ensure_table(self) -> None:
        try:
            ModelSelection.__table__.create(bind=self.engine, checkfirst=True)
        except Exception:
            logger.exception("Failed to ensure model_selections table exists")
            raise
