# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from functools import partial

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from domain.model_configs import ModelConfigSnapshot
from domain.settings.configuration import DEFAULT_SHARED_RAG_SETTINGS
from repositories.database.session import resolve_engine, resolve_session_factory
from repositories.schemas.configuration import ApplicationConfiguration
from repositories.serialization.application_configuration import (
    ApplicationConfigurationSerializer,
)

UNSET = object()

# Execution settings the singleton held before agents owned them. They are
# read once to seed the default agents and then pruned from the payload.
LEGACY_EXECUTION_KEYS: frozenset[str] = frozenset(
    {
        "use_cloud_models",
        "use_cloud_services",
        "cloud_provider",
        "llm_provider",
        "cloud_model",
        "text_extraction_model",
        "clinical_model",
        "revision_model",
        "timeline_model",
        "reasoning_level",
        "ollama_seed",
    }
)


###############################################################################
class ModelConfigSerializer:
    """Persist the shared (non-execution) configuration as one singleton."""

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
            return self.snapshot_from_payload(
                dict(row.payload), updated_at=row.updated_at
            )

    # -------------------------------------------------------------------------
    def load_raw_payload(self) -> dict[str, object] | None:
        """Return the stored payload unchanged (legacy keys included)."""
        payload, _ = self.application_configuration.load_with_metadata()
        return dict(payload) if isinstance(payload, dict) else None

    # -------------------------------------------------------------------------
    def ensure_defaults(self, payload: dict[str, object]) -> bool:
        """Fill missing shared defaults, preserve saved values, prune legacy keys."""
        defaults = self.prune_legacy_keys(payload)
        defaults["rag_settings"] = self.normalize_rag_settings(
            defaults.get("rag_settings")
        )

        current, _ = self.application_configuration.load_with_metadata()
        if current is None:
            return self.application_configuration.save_if_missing(defaults)

        merged = dict(defaults)
        merged.update(self.prune_legacy_keys(current))
        for block_name in (
            "jobs",
            "runtime",
            "ingestion",
            "session_pipeline",
            "clinical_language_detection",
            "drugs_matcher",
        ):
            default_block = defaults.get(block_name)
            current_block = current.get(block_name)
            if isinstance(default_block, dict):
                merged[block_name] = {
                    **default_block,
                    **(
                        current_block
                        if isinstance(current_block, dict)
                        else {}
                    ),
                }

        merged["rag_settings"] = {
            **self.normalize_rag_settings(defaults.get("rag_settings")),
            **self.normalize_rag_settings(current.get("rag_settings")),
        }
        if merged == current:
            return False
        self.application_configuration.save(merged)
        return True

    # -------------------------------------------------------------------------
    def save_snapshot(
        self,
        *,
        rag_settings: dict[str, object] | object = UNSET,
    ) -> ModelConfigSnapshot:
        updates: dict[str, object] = {}
        if rag_settings is not UNSET:
            updates["rag_settings"] = self.normalize_rag_settings(rag_settings)
        saved_payload, updated_at = self.application_configuration.update_payload(
            partial(self._merge_latest_configuration, updates),
            return_metadata=True,
        )
        return self.snapshot_from_payload(saved_payload, updated_at=updated_at)

    # -------------------------------------------------------------------------
    def _merge_latest_configuration(
        self, updates: dict[str, object], payload: dict[str, object]
    ) -> None:
        payload.pop("database", None)
        for key in LEGACY_EXECUTION_KEYS:
            payload.pop(key, None)
        payload["rag_settings"] = self.normalize_rag_settings(
            payload.get("rag_settings")
        )
        payload.update(updates)

    # -------------------------------------------------------------------------
    @classmethod
    def snapshot_from_payload(
        cls, payload: Mapping[str, object], *, updated_at: datetime | None
    ) -> ModelConfigSnapshot:
        return ModelConfigSnapshot(
            rag_settings=cls.normalize_rag_settings(payload.get("rag_settings")),
            updated_at=updated_at,
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def prune_legacy_keys(payload: Mapping[str, object]) -> dict[str, object]:
        return {
            key: value
            for key, value in payload.items()
            if key not in LEGACY_EXECUTION_KEYS
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def has_legacy_keys(payload: Mapping[str, object]) -> bool:
        return bool(LEGACY_EXECUTION_KEYS & set(payload))

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_rag_settings(value: object) -> dict[str, object]:
        """Keep only shared index settings; agent retrieval keys are dropped."""
        source = value if isinstance(value, Mapping) else {}
        return {
            key: source.get(key, default)
            for key, default in DEFAULT_SHARED_RAG_SETTINGS.items()
        }
