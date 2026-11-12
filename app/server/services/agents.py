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
    AgentConfig,
    AgentCreateRequest,
    AssessmentConfig,
    CapabilityConfig,
    RevisionConfig,
    AgentDuplicateRequest,
    AgentExecutionContext,
    AgentLifecycleRequest,
    AgentListResponse,
    AgentModelRef,
    AgentResponse,
    AgentRevisionListResponse,
    AgentRevisionResponse,
    AgentUpdateRequest,
    AgentValidationIssue,
    Capability,
    agent_config_hash,
    agent_is_editable,
    agent_is_selectable,
    parse_agent_config,
)
from repositories.agent_repository import (
    AgentDefaultRetireError,
    AgentDeleteBlockedError,
    AgentInactiveError,
    AgentNotEditableError,
    AgentNotFoundError,
    AgentRepository,
    AgentRevisionConflictError,
)
from repositories.context import RepositoryContext
from services.llm.model_config import ModelConfigService
from services.llm.provider_registry import provider_registry
from services.security.access_keys import AccessKeyService


###############################################################################
def _model_refs(
    capability: Capability, config: CapabilityConfig
) -> list[tuple[str, AgentModelRef]]:
    """Model references of one capability slice, keyed by editor field path."""

    if isinstance(config, AssessmentConfig):
        refs: list[tuple[str, AgentModelRef]] = [
            ("assessment.models.extraction", config.models.extraction),
            ("assessment.models.clinical", config.models.clinical),
        ]
        refs.extend(
            (f"assessment.models.extraction_overrides.{stage}", item)
            for stage, item in config.models.extraction_overrides.items()
        )
        return refs
    if isinstance(config, RevisionConfig):
        refs = [("revision.models.planning", config.models.planning)]
        refs.extend(
            (f"revision.models.stage_overrides.{stage}", item)
            for stage, item in config.models.stage_overrides.items()
        )
        return refs
    return [("timeline.model", config.model)]


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
            name=record["name"],
            description=record["description"],
            current_revision=record["current_revision"],
            is_default=record["is_default"],
            is_active=record["is_active"],
            is_locked=record["is_locked"],
            is_archived=record["is_archived"],
            editable=agent_is_editable(
                is_active=record["is_active"],
                is_locked=record["is_locked"],
                is_archived=record["is_archived"],
            ),
            capabilities=parse_agent_config(
                revision["configuration"]
            ).enabled_capabilities(),
            reference_count=record["reference_count"],
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
        self,
        *,
        include_archived: bool = False,
        selectable_for: Capability | None = None,
    ) -> AgentListResponse:
        records = self.repository.list_agents(include_archived=include_archived)
        if selectable_for is not None:
            records = [
                item
                for item in records
                if agent_is_selectable(
                    parse_agent_config(item["revision"]["configuration"]),
                    selectable_for,
                    is_active=item["is_active"],
                    is_archived=item["is_archived"],
                )
            ]
        return AgentListResponse(agents=[self._to_response(item) for item in records])

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
            name=request.name.strip(),
            description=request.description.strip(),
            configuration=config.model_dump(mode="json"),
            configuration_hash=agent_config_hash(config),
            schema_version=AGENT_CONFIG_SCHEMA_VERSION,
            is_default=self.repository.get_default() is None,
        )
        return self._to_response(record)

    # -------------------------------------------------------------------------
    def update_agent(self, agent_id: str, request: AgentUpdateRequest) -> AgentResponse:
        self._require(agent_id)
        config = request.configuration
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
        except AgentNotEditableError as exc:
            raise ServiceConflictError(
                "The agent must be active, unlocked and not archived to be edited."
            ) from exc
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
        except AgentInactiveError as exc:
            raise ServiceConflictError(
                "Inactive or archived agents cannot become the default."
            ) from exc

    # -------------------------------------------------------------------------
    def set_lifecycle(
        self, agent_id: str, request: AgentLifecycleRequest
    ) -> AgentResponse:
        try:
            return self._to_response(
                self.repository.set_lifecycle(
                    agent_id,
                    is_active=request.is_active,
                    is_locked=request.is_locked,
                    is_archived=request.is_archived,
                )
            )
        except AgentNotFoundError as exc:
            raise ServiceNotFoundError("Agent not found.") from exc
        except AgentDefaultRetireError as exc:
            raise ServiceConflictError(
                "Choose another default agent before deactivating or archiving this one."
            ) from exc

    # -------------------------------------------------------------------------
    def delete_agent(self, agent_id: str) -> None:
        try:
            self.repository.delete_agent(agent_id)
        except AgentNotFoundError as exc:
            raise ServiceNotFoundError("Agent not found.") from exc
        except AgentDeleteBlockedError as exc:
            raise ServiceConflictError(exc.reason) from exc

    # -------------------------------------------------------------------------
    def collect_context_issues(
        self, context: AgentExecutionContext
    ) -> list[AgentValidationIssue]:
        return self.collect_issues(context.capability, context.configuration)

    # -------------------------------------------------------------------------
    def collect_issues(
        self, capability: Capability, config: CapabilityConfig
    ) -> list[AgentValidationIssue]:
        """Deterministic readiness checks; never contacts a provider."""

        issues = self._model_issues(capability, config)
        checked_scopes: set[str] = set()
        for field, ref in _model_refs(capability, config):
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
    def _model_issues(
        self, capability: Capability, config: CapabilityConfig
    ) -> list[AgentValidationIssue]:
        issues: list[AgentValidationIssue] = []
        local_models: set[str] | None = None
        for field, ref in _model_refs(capability, config):
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
        if isinstance(config, RevisionConfig):
            # Imported lazily: services.inspection imports this module.
            from services.inspection.revision_tools import RevisionToolRegistry

            unknown = sorted(
                set(config.limits.allowed_tools or []) - RevisionToolRegistry.names
            )
            if unknown:
                issues.append(
                    AgentValidationIssue(
                        code="revision_tool_unknown",
                        field="revision.limits.allowed_tools",
                        message="Unknown revision tools: " + ", ".join(unknown),
                    )
                )
        return issues

    # -------------------------------------------------------------------------
    def _raise_for_invalid_models(self, config: AgentConfig) -> None:
        issues: list[AgentValidationIssue] = []
        for capability in config.enabled_capabilities():
            issues.extend(
                self._model_issues(capability, config.capability(capability))
            )
        if issues:
            raise ServiceValidationError("; ".join(item.message for item in issues))

    # -------------------------------------------------------------------------
    def resolve_for_execution(
        self,
        *,
        capability: Capability,
        agent_id: str | None = None,
        expected_revision: int | None = None,
    ) -> AgentExecutionContext:
        """Resolve one immutable agent revision for a launch.

        ``agent_id`` None selects the default agent. The agent must be active,
        not archived and have ``capability`` enabled. ``expected_revision``
        guards against an edit between preflight and launch silently changing
        the configuration that was checked.
        """

        record = (
            self.repository.get_agent(agent_id)
            if agent_id
            else self.repository.get_default()
        )
        if record is None:
            raise ServiceNotFoundError(
                "Agent not found." if agent_id else "No default agent exists."
            )
        config = parse_agent_config(record["revision"]["configuration"])
        if record["is_archived"]:
            raise ServiceValidationError("Archived agents cannot run new operations.")
        if not record["is_active"]:
            raise ServiceValidationError("Inactive agents cannot run new operations.")
        if not config.capability(capability).enabled:
            raise ServiceValidationError(
                f"Agent '{record['name']}' does not have the {capability} "
                "capability enabled."
            )
        if (
            expected_revision is not None
            and int(expected_revision) != record["current_revision"]
        ):
            raise ServiceConflictError(
                "The selected agent changed after it was checked. Review it and retry."
            )
        return self._context(
            record["revision"], record["id"], record["name"], capability
        )

    # -------------------------------------------------------------------------
    def resolve_revision(
        self, revision_id: int, capability: Capability
    ) -> AgentExecutionContext:
        """Replay a stored revision; lifecycle state is intentionally ignored."""

        revision = self.repository.get_revision(revision_id)
        if revision is None:
            raise ServiceNotFoundError("Agent revision not found.")
        return self._context(
            revision, revision["agent_id"], revision["agent_name"], capability
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _context(
        revision: dict[str, Any],
        agent_id: str,
        agent_name: str,
        capability: Capability,
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
            capability=capability,
            config_hash=revision["configuration_hash"],
            configuration=config.capability(capability),
        )

