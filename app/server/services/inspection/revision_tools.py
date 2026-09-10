from __future__ import annotations

from typing import Any

from repositories.context import RepositoryContext
from repositories.dilirank_repository import DiliRankRepository

###############################################################################
class RevisionToolRegistry:
    names = frozenset(
        {
            "read_session_context",
            "read_result_payload_path",
            "read_manual_edits",
            "read_version_lineage",
            "search_livertox_catalog",
            "get_livertox_excerpt",
            "search_dilirank_catalog",
            "get_dilirank_records",
            "get_drug_knowledge_bundle",
            "search_rag",
        }
    )

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        clinical_session_repository: Any,
        session_revision_repository: Any,
        knowledge_repository: Any,
        session: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        self.clinical_session_repository = clinical_session_repository
        self.session_revision_repository = session_revision_repository
        self.knowledge_repository = knowledge_repository
        repository_context = getattr(knowledge_repository, "context", None)
        self.dilirank_repository = (
            DiliRankRepository(repository_context)
            if isinstance(repository_context, RepositoryContext)
            else None
        )
        self.session = session
        self.context = context

    # -------------------------------------------------------------------------
    def manifest(self, allowed: list[str] | None) -> list[str]:
        return sorted(
            self.names if allowed is None else self.names.intersection(allowed)
        )

    # -------------------------------------------------------------------------
    def execute(
        self, name: str, arguments: dict[str, Any], allowed: list[str] | None
    ) -> dict[str, Any]:
        if name not in self.manifest(allowed):
            raise ValueError("Unknown or disallowed revision tool.")
        if not isinstance(arguments, dict) or any(
            key.startswith("_") for key in arguments
        ):
            raise ValueError("Malformed revision tool input.")
        if name == "read_session_context":
            return {"context": self.context}
        if name == "read_manual_edits":
            return {"items": self.session.get("manual_edit_history") or []}
        if name == "read_version_lineage":
            return {
                "items": self.session_revision_repository.list_session_versions(
                    int(self.session["session_id"])
                )
            }
        if name == "read_result_payload_path":
            return self._payload_path(str(arguments.get("path") or ""))
        if name == "get_livertox_excerpt":
            return {
                "item": self.knowledge_repository.get_livertox_excerpt(
                    self._positive_int(arguments.get("drug_id"))
                )
            }
        if name == "get_dilirank_records":
            return {
                "items": (
                    self.dilirank_repository.get_records_for_drug(
                        self._positive_int(arguments.get("drug_id"))
                    )
                    if self.dilirank_repository is not None
                    else []
                )
            }
        if name == "get_drug_knowledge_bundle":
            drug_id = self._positive_int(arguments.get("drug_id"))
            item = dict(self.knowledge_repository.get_drug_knowledge_bundle(drug_id))
            item["dilirank_records"] = (
                self.dilirank_repository.get_records_for_drug(drug_id)
                if self.dilirank_repository is not None
                else []
            )
            return {"item": item}
        if name == "search_livertox_catalog":
            return {
                "items": self.knowledge_repository.list_livertox_catalog(
                    search=str(arguments.get("query") or ""), offset=0, limit=10
                )[0]
            }
        if name == "search_dilirank_catalog":
            return {
                "items": (
                    self.dilirank_repository.list_catalog(
                        search=str(arguments.get("query") or ""),
                        offset=0,
                        limit=10,
                    )[0]
                    if self.dilirank_repository is not None
                    else []
                )
            }
        return {
            "available": False,
            "warning": "RAG retrieval is unavailable to this revision run.",
        }

    # -------------------------------------------------------------------------
    def _payload_path(self, path: str) -> dict[str, Any]:
        if (
            not path
            or ".." in path
            or not all(part.replace("_", "").isalnum() for part in path.split("."))
        ):
            raise ValueError("Invalid result payload path.")
        value: Any = self.session.get("result_payload") or {}
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return {"found": False, "path": path}
            value = value[part]
        return {"found": True, "path": path, "value": value}

    # -------------------------------------------------------------------------
    @staticmethod
    def _positive_int(value: Any) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError("Tool ids must be positive integers.") from None
        if number < 1:
            raise ValueError("Tool ids must be positive.")
        return number
