from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.paths import DOCS_PATH, VECTOR_DB_PATH
from configurations.startup import get_server_settings
from services.retrieval.settings import build_effective_rag_settings
from services.text.vocabulary import (
    deactivate_text_normalization_term_payload,
    invalidate_text_normalization_snapshot,
    list_text_normalization_term_payloads,
    upsert_text_normalization_term_payload,
)

###############################################################################
class InspectionUpdateConfigMixin:
    RAG_MANIFEST_FILE_NAME = "rag_index_manifest.json"

    # -------------------------------------------------------------------------
    def load_runtime_config(self) -> dict[str, Any]:
        return get_server_settings().model_dump()

    # -------------------------------------------------------------------------
    def rag_manifest_path(self) -> Path:
        return VECTOR_DB_PATH / self.RAG_MANIFEST_FILE_NAME

    # -------------------------------------------------------------------------
    def read_rag_manifest(self) -> dict[str, Any]:
        manifest_path = self.rag_manifest_path()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    # -------------------------------------------------------------------------
    def write_rag_manifest(
        self,
        *,
        documents_path: str,
        summary: dict[str, Any],
    ) -> None:
        manifest_path = self.rag_manifest_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents_path": documents_path,
            "documents": int(summary.get("documents", 0) or 0),
            "chunks": int(summary.get("chunks", 0) or 0),
            "supported_files": int(summary.get("supported_files", 0) or 0),
            "loaded_documents": int(summary.get("loaded_documents", 0) or 0),
            "built_at": datetime.now(UTC).isoformat(),
        }
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -------------------------------------------------------------------------
    def get_effective_rag_documents_path(self) -> str:
        manifest = self.read_rag_manifest()
        manifest_path = str(manifest.get("documents_path") or "").strip()
        if manifest_path:
            return manifest_path
        config = self.load_runtime_config()
        rag_cfg = config.get("rag", {}) if isinstance(config, dict) else {}
        return str(rag_cfg.get("documents_path", DOCS_PATH))

    # -------------------------------------------------------------------------
    def list_reference_catalog_runtime_observations(
        self, category: str | None = None
    ) -> list[dict[str, Any]]:
        return list_text_normalization_term_payloads(category=category)

    # -------------------------------------------------------------------------
    def upsert_reference_catalog_runtime_observation(
        self,
        *,
        category: str,
        term: str,
        replacement: str | None,
        source: str,
        is_active: bool,
    ) -> dict[str, Any]:
        payload = upsert_text_normalization_term_payload(
            category=category,
            term=term,
            replacement=replacement,
            source=source,
            is_active=is_active,
        )
        invalidate_text_normalization_snapshot()
        return payload

    # -------------------------------------------------------------------------
    def deactivate_reference_catalog_runtime_observation(
        self, *, category: str, term: str
    ) -> bool:
        updated = deactivate_text_normalization_term_payload(
            category=category,
            term=term,
        )
        if updated:
            invalidate_text_normalization_snapshot()
        return updated

    # -------------------------------------------------------------------------
    def build_update_config_response(self, target: str) -> dict[str, Any]:
        config = self.load_runtime_config()
        settings = get_server_settings()
        if target == "rxnav":
            source = config.get("runtime", {})
            defaults = {
                "rxnav_request_timeout": float(
                    source.get(
                        "rxnav_request_timeout",
                        settings.runtime.rxnav_request_timeout,
                    )
                ),
                "rxnav_max_concurrency": int(
                    source.get(
                        "rxnav_max_concurrency",
                        settings.runtime.rxnav_max_concurrency,
                    )
                ),
            }
            allowed_fields = list(defaults.keys())
        elif target == "livertox":
            source = config.get("runtime", {})
            defaults = {
                "livertox_monograph_max_workers": int(
                    source.get(
                        "livertox_monograph_max_workers",
                        settings.runtime.livertox_monograph_max_workers,
                    )
                ),
                "livertox_archive": str(
                    source.get(
                        "livertox_archive",
                        settings.runtime.livertox_archive,
                    )
                ),
                "redownload": False,
            }
            allowed_fields = list(defaults.keys())
        else:
            rag_settings = build_effective_rag_settings()
            defaults = {}
            allowed_fields = []
            summary = {
                "chunk_size": int(rag_settings.chunk_size),
                "chunk_overlap": int(rag_settings.chunk_overlap),
                "embedding_batch_size": int(rag_settings.embedding_batch_size),
                "vector_stream_batch_size": int(rag_settings.vector_stream_batch_size),
                "embedding_max_workers": int(rag_settings.embedding_max_workers),
                "embedding_backend": rag_settings.embedding_backend,
                "ollama_embedding_model": rag_settings.ollama_embedding_model,
                "hf_embedding_model": rag_settings.hf_embedding_model,
                "cloud_provider": rag_settings.cloud_provider,
                "cloud_embedding_model": rag_settings.cloud_embedding_model,
                "use_cloud_embeddings": bool(rag_settings.use_cloud_embeddings),
                "reset_vector_collection": bool(rag_settings.reset_vector_collection),
            }
            return {
                "target": target,
                "defaults": defaults,
                "allowed_fields": allowed_fields,
                "summary": summary,
                "read_only": True,
            }

        return {
            "target": target,
            "defaults": defaults,
            "allowed_fields": allowed_fields,
            "summary": {},
            "read_only": False,
        }
