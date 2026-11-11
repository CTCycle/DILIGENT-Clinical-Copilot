# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from typing import Any, cast

from common.exceptions import (
    ServiceConflictError,
    ServiceNotFoundError,
    ServiceValidationError,
)
from domain.agents import (
    AGENT_CONFIG_SCHEMA_VERSION,
    AgentCreateRequest,
    AgentDuplicateRequest,
    AgentExecutionContext,
    AgentKind,
    AgentListResponse,
    AgentModelRef,
    AgentResponse,
    AgentRevisionListResponse,
    AgentRevisionResponse,
    AgentUpdateRequest,
    AgentValidationIssue,
    AgentValidationResponse,
    DiliAgentConfig,
    RevisionAgentConfig,
    agent_config_hash,
    parse_agent_config,
)
from repositories.agent_repository import (
    AgentArchivedError,
    AgentDefaultArchiveError,
    AgentNotFoundError,
    AgentRepository,
    AgentRevisionConflictError,
)
from repositories.context import RepositoryContext
from services.llm.model_config import ModelConfigService
from services.llm.provider_registry import provider_registry
from services.security.access_keys import AccessKeyService

AgentConfigModel = DiliAgentConfig | RevisionAgentConfig

###############################################################################
def _model_refs(config: AgentConfigModel) -> list[tuple[str, AgentModelRef]]:
    if isinstance(config, DiliAgentConfig):
        models = config.models
        refs: list[tuple[str, AgentModelRef]] = [
            ("models.extraction", models.extraction),
            ("models.clinical", models.clinical),
            ("models.timeline", models.timeline),
        ]
        refs.extend(
            (f"models.extraction_overrides.{stage}", item)
            for stage, item in models.extraction_overrides.items()
        )
        if models.report_synthesis is not None:
            refs.append(("models.report_synthesis", models.report_synthesis))
        return refs
    refs = [("models.planning", config.models.planning)]
    refs.extend(
        (f"models.stage_overrides.{stage}", item)
        for stage, item in config.models.stage_overrides.items()
    )
    return refs


###############################################################################
class AgentService:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        repository: AgentRepository | None = None,
        *,
        model_config_service: ModelConfigService | None = None,
        access_key_service: AccessKeyService | None = None,
    ) -> None:
        self.repository = repository or AgentRepository(RepositoryContext.create())
        self.model_config_service = model_config_service or ModelConfigService()
        self.access_key_service = access_key_service or AccessKeyService()

    # -------------------------------------------------------------------------
    @staticmethod
    def _to_response(record: dict[str, Any]) -> AgentResponse:
        revision = record["revision"]
        return AgentResponse(
            id=record["id"],
            kind=record["kind"],
            name=record["name"],
            description=record["description"],
            current_revision=record["current_revision"],
            is_default=record["is_default"],
            is_archived=record["is_archived"],
            created_at=record["created_at"],
            updated_at=record["updated_at"],
            revision=AgentService._revision_response(revision),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _revision_response(revision: dict[str, Any]) -> AgentRevisionResponse:
        return AgentRevisionResponse(
            revision_id=revision["id"],
            revision_number=revision["revision_number"],
            schema_version=revision["schema_version"],
            configuration_hash=revision["configuration_hash"],
            configuration=parse_agent_config(revision["configuration"]),
            created_at=revision["created_at"],
        )

    # -------------------------------------------------------------------------
    def _require(self, agent_id: str) -> dict[str, Any]:
        record = self.repository.get_agent(agent_id)
        if record is None:
            raise ServiceNotFoundError("Agent not found.")
        return record

    # -------------------------------------------------------------------------
    def list_agents(
        self, kind: AgentKind | None = None, *, include_archived: bool = False
    ) -> AgentListResponse:
        return AgentListResponse(
            agents=[
                self._to_response(item)
                for item in self.repository.list_agents(
                    kind, include_archived=include_archived
                )
            ]
        )

    # -------------------------------------------------------------------------
    def get_agent(self, agent_id: str) -> AgentResponse:
        return self._to_response(self._require(agent_id))

    # -------------------------------------------------------------------------
    def list_revisions(self, agent_id: str) -> AgentRevisionListResponse:
        self._require(agent_id)
        return AgentRevisionListResponse(
            revisions=[
                self._revision_response(item)
                for item in self.repository.list_revisions(agent_id)
            ]
        )

    # -------------------------------------------------------------------------
    def create_agent(self, request: AgentCreateRequest) -> AgentResponse:
        config = request.configuration
        self._raise_for_invalid_models(config)
        record = self.repository.create_agent(
            kind=config.kind,
            name=request.name.strip(),
            description=request.description.strip(),
            configuration=config.model_dump(mode="json"),
            configuration_hash=agent_config_hash(config),
            schema_version=AGENT_CONFIG_SCHEMA_VERSION,
            is_default=self.repository.get_default(config.kind) is None,
        )
        return self._to_response(record)

    # -------------------------------------------------------------------------
    def update_agent(self, agent_id: str, request: AgentUpdateRequest) -> AgentResponse:
        current = self._require(agent_id)
        config = request.configuration
        if config.kind != current["kind"]:
            raise ServiceValidationError(
                f"Agent kind cannot change from '{current['kind']}' to '{config.kind}'."
            )
        self._raise_for_invalid_models(config)
        try:
            record = self.repository.create_revision(
                agent_id=agent_id,
                expected_revision=request.expected_revision,
                name=request.name.strip(),
                description=request.description.strip(),
                configuration=config.model_dump(mode="json"),
                configuration_hash=agent_config_hash(config),
                schema_version=AGENT_CONFIG_SCHEMA_VERSION,
            )
        except AgentNotFoundError as exc:
            raise ServiceNotFoundError("Agent not found.") from exc
        except AgentArchivedError as exc:
            raise ServiceValidationError("Archived agents cannot be edited.") from exc
        except AgentRevisionConflictError as exc:
            raise ServiceConflictError(
                "The agent was changed since it was loaded. Reload and retry."
            ) from exc
        return self._to_response(record)

    # -------------------------------------------------------------------------
    def duplicate_agent(
        self, agent_id: str, request: AgentDuplicateRequest
    ) -> AgentResponse:
        source = self._require(agent_id)
        config = parse_agent_config(source["revision"]["configuration"])
        record = self.repository.create_agent(
            kind=source["kind"],
            name=(request.name or f"{source['name']} (copy)").strip()[:120],
            description=source["description"],
            configuration=config.model_dump(mode="json"),
            configuration_hash=agent_config_hash(config),
            schema_version=AGENT_CONFIG_SCHEMA_VERSION,
        )
        return self._to_response(record)

    # -------------------------------------------------------------------------
    def set_default(self, agent_id: str) -> AgentResponse:
        try:
            return self._to_response(self.repository.set_default(agent_id))
        except AgentNotFoundError as exc:
            raise ServiceNotFoundError("Agent not found.") from exc
        except AgentArchivedError as exc:
            raise ServiceValidationError(
                "Archived agents cannot become the default."
            ) from exc

    # -------------------------------------------------------------------------
    def archive_agent(self, agent_id: str) -> AgentResponse:
        try:
            return self._to_response(self.repository.archive_agent(agent_id))
        except AgentNotFoundError as exc:
            raise ServiceNotFoundError("Agent not found.") from exc
        except AgentDefaultArchiveError as exc:
            raise ServiceConflictError(
                "Choose another default agent before archiving this one."
            ) from exc

    # -------------------------------------------------------------------------
    def validate_agent(self, agent_id: str) -> AgentValidationResponse:
        record = self._require(agent_id)
        config = parse_agent_config(record["revision"]["configuration"])
        issues = self.collect_issues(config)
        if record["is_archived"]:
            issues.append(
                AgentValidationIssue(
                    code="agent_archived", message="This agent is archived."
                )
            )
        return AgentValidationResponse(valid=not issues, issues=issues)

    # -------------------------------------------------------------------------
    def collect_issues(self, config: AgentConfigModel) -> list[AgentValidationIssue]:
        """Deterministic readiness checks; never contacts a provider."""

        issues = self._model_issues(config)
        checked_scopes: set[str] = set()
        for field, ref in _model_refs(config):
            if ref.provider == "ollama":
                continue
            try:
                scope = provider_registry.get(ref.provider).credential_scope
            except ValueError:
                continue
            if scope in checked_scopes:
                continue
            checked_scopes.add(scope)
            active = [
                item
                for item in self.access_key_service.list_access_keys(cast(Any, scope))
                if item.is_active
            ]
            if not active:
                issues.append(
                    AgentValidationIssue(
                        code="provider_key_missing",
                        field=field,
                        message=(
                            f"Configure an active {ref.provider} access key "
                            "before running this agent."
                        ),
                    )
                )
        return issues

    # -------------------------------------------------------------------------
    def _model_issues(self, config: AgentConfigModel) -> list[AgentValidationIssue]:
        issues: list[AgentValidationIssue] = []
        local_models: set[str] | None = None
        for field, ref in _model_refs(config):
            if ref.provider == "ollama":
                if local_models is None:
                    local_models = self.model_config_service.known_local_model_names()
                known = local_models
            else:
                try:
                    known = self.model_config_service.known_provider_model_names(
                        ref.provider
                    )
                except ValueError:
                    issues.append(
                        AgentValidationIssue(
                            code="provider_unsupported",
                            field=field,
                            message=f"Provider '{ref.provider}' is not supported.",
                        )
                    )
                    continue
            # An unloaded cloud catalog cannot prove a model invalid.
            if known and ref.model not in known:
                issues.append(
                    AgentValidationIssue(
                        code="model_unavailable",
                        field=field,
                        message=(
                            f"Model '{ref.model}' is not available from "
                            f"provider '{ref.provider}'."
                        ),
                    )
                )
        if isinstance(config, RevisionAgentConfig):
            # Imported lazily: services.inspection imports this module.
            from services.inspection.revision_tools import RevisionToolRegistry

            unknown = sorted(
                set(config.limits.allowed_tools or []) - RevisionToolRegistry.names
            )
            if unknown:
                issues.append(
                    AgentValidationIssue(
                        code="revision_tool_unknown",
                        field="limits.allowed_tools",
                        message="Unknown revision tools: " + ", ".join(unknown),
                    )
                )
        return issues

    # -------------------------------------------------------------------------
    def _raise_for_invalid_models(self, config: AgentConfigModel) -> None:
        issues = self._model_issues(config)
        if issues:
            raise ServiceValidationError("; ".join(item.message for item in issues))

    # -------------------------------------------------------------------------
    def resolve_for_execution(
        self,
        *,
        kind: AgentKind,
        agent_id: str | None = None,
        expected_revision: int | None = None,
    ) -> AgentExecutionContext:
        """Resolve one immutable agent revision for a launch.

        ``agent_id`` None selects the default agent of ``kind``.
        ``expected_revision`` guards against an edit between preflight and
        launch silently changing the configuration that was checked.
        """

        record = (
            self.repository.get_agent(agent_id)
            if agent_id
            else self.repository.get_default(kind)
        )
        if record is None:
            raise ServiceNotFoundError(
                "Agent not found." if agent_id else f"No default {kind} agent exists."
            )
        if record["kind"] != kind:
            raise ServiceValidationError(
                f"Agent '{record['name']}' is a {record['kind']} agent, not {kind}."
            )
        if record["is_archived"]:
            raise ServiceValidationError("Archived agents cannot run new sessions.")
        if (
            expected_revision is not None
            and int(expected_revision) != record["current_revision"]
        ):
            raise ServiceConflictError(
                "The selected agent changed after it was checked. Review it and retry."
            )
        return self._context(record["revision"], record["id"], record["name"], kind)

    # -------------------------------------------------------------------------
    def resolve_revision(self, revision_id: int) -> AgentExecutionContext:
        revision = self.repository.get_revision(revision_id)
        if revision is None:
            raise ServiceNotFoundError("Agent revision not found.")
        return self._context(
            revision, revision["agent_id"], revision["agent_name"], revision["kind"]
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _context(
        revision: dict[str, Any], agent_id: str, agent_name: str, kind: AgentKind
    ) -> AgentExecutionContext:
        config = parse_agent_config(revision["configuration"])
        if agent_config_hash(config) != revision["configuration_hash"]:
            raise ServiceValidationError(
                "Stored agent revision failed its integrity check."
            )
        return AgentExecutionContext(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_revision_id=int(revision["id"]),
            revision_number=int(revision["revision_number"]),
            kind=kind,
            config_hash=revision["configuration_hash"],
            configuration=config,
        )

