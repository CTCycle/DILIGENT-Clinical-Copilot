from __future__ import annotations

from typing import Any

from repositories.context import RepositoryContext
from repositories import values as repository_values
from repositories.serialization import session_revision_artifacts, session_revision_data, session_revision_steps, session_result_data


class SessionRevisionRepository:
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    def __get_revision_data(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return getattr(session_revision_data, name)(self, *args, **kwargs)

    def list_session_versions(self, session_id: int): return self.__get_revision_data("list_session_versions", session_id)
    def get_session_version_detail(self, session_id: int, **kwargs: Any): return self.__get_revision_data("get_session_version_detail", session_id, **kwargs)
    def get_version_record_for_session(self, session_id: int): return self.__get_revision_data("get_version_record_for_session", session_id)
    def get_latest_version_record_for_session(self, session_id: int): return self.__get_revision_data("get_latest_version_record_for_session", session_id)
    def get_next_session_version(self, root_session_id: int): return session_result_data.get_next_session_version(self, root_session_id)
    def list_manual_report_edits(self, session_id: int): return self.__get_revision_data("list_manual_report_edits", session_id)
    def update_current_report_text_with_manual_audit(self, session_id: int, **kwargs: Any): return self.__get_revision_data("update_current_report_text_with_manual_audit", session_id, **kwargs)
    def create_revision_version_shell(self, session_id: int, **kwargs: Any): return self.__get_revision_data("create_revision_version_shell", session_id, **kwargs)
    def create_or_update_revision_run(self, **kwargs: Any): return self.__get_revision_data("create_or_update_revision_run", **kwargs)
    def get_revision_run(self, pipeline_run_id: str): return self.__get_revision_data("get_revision_run", pipeline_run_id)
    def get_revision_run_by_job_id(self, job_id: str): return self.__get_revision_data("get_revision_run_by_job_id", job_id)
    def fail_revision_run(self, **kwargs: Any): return self.__get_revision_data("fail_revision_run", **kwargs)
    def cancel_revision_run(self, **kwargs: Any): return self.__get_revision_data("cancel_revision_run", **kwargs)
    def start_revision_step(self, **kwargs: Any): return session_revision_steps.start_revision_step(self, **kwargs)
    def complete_revision_step(self, **kwargs: Any): return session_revision_steps.complete_revision_step(self, **kwargs)
    def fail_revision_step(self, **kwargs: Any): return session_revision_steps.fail_revision_step(self, **kwargs)
    def list_revision_steps(self, pipeline_run_id: str): return self.__get_revision_data("list_revision_steps", pipeline_run_id)
    def persist_revision_artifact(self, **kwargs: Any): return session_revision_artifacts.persist_revision_artifact(self, **kwargs)
    def persist_revision_agent_issue_scan(self, **kwargs: Any): return session_revision_artifacts.persist_revision_agent_issue_scan(self, **kwargs)
    def list_revision_artifacts_for_version(self, **kwargs: Any): return session_revision_artifacts.list_revision_artifacts_for_version(self, **kwargs)
    def list_revision_entities_for_version(self, **kwargs: Any): return session_revision_artifacts.list_revision_entities_for_version(self, **kwargs)
    def list_revision_reviews_for_version(self, **kwargs: Any): return session_revision_steps.list_revision_reviews_for_version(self, **kwargs)
    def record_revision_review_action(self, **kwargs: Any): return session_revision_steps.record_revision_review_action(self, **kwargs)
    def persist_revision_entities(self, **kwargs: Any): return session_revision_artifacts.persist_revision_entities(self, **kwargs)
    def finalize_revision_version(self, **kwargs: Any): return self.__get_revision_data("finalize_revision_version", **kwargs)

    def get_session_detail(self, session_id: int): return session_result_data.get_session_detail(self, session_id)
    def get_session_result_payload(self, session_id: int): return session_result_data.get_session_result_payload(self, session_id)
    def normalize_string(self, value: Any) -> str | None: return repository_values.normalize_string(value)
    def normalize_session_status(self, value: Any) -> str: return repository_values.normalize_session_status(value)
    def parse_session_result_payload(self, value: str | None): return session_result_data.parse_session_result_payload(self, value)
    def serialize_json_payload(self, value: Any) -> str | None: return session_result_data.serialize_json_payload(self, value)
