from __future__ import annotations

from typing import Any

from repositories.context import RepositoryContext
from repositories.serialization import session_result_data, session_timelines


class SessionTimelineRepository:
    def __init__(self, context: RepositoryContext) -> None:
        self.context = context
        self.engine = context.engine
        self.session_factory = context.session_factory

    def list_session_timelines(self, session_id: int):
        return session_timelines.list_session_timelines(self, session_id)

    def get_session_timeline_record(self, session_id: int, timeline_id: int):
        return session_timelines.get_session_timeline_record(self, session_id, timeline_id)

    def get_latest_session_timeline_record(self, session_id: int):
        return session_timelines.get_latest_session_timeline_record(self, session_id)

    def create_session_timeline_record(self, session_id: int, payload: dict[str, Any]):
        return session_timelines.create_session_timeline_record(self, session_id, payload)

    def delete_session_timeline_record(self, session_id: int, timeline_id: int) -> bool:
        return session_timelines.delete_session_timeline_record(self, session_id, timeline_id)

    def get_session_timeline_source(self, session_id: int):
        return session_timelines.get_session_timeline_source(self, session_id)

    def normalize_string(self, value: Any) -> str | None:
        return session_result_data.normalize_string(self, value)

    def parse_session_result_payload(self, value: str | None):
        return session_result_data.parse_session_result_payload(self, value)

    def serialize_json_payload(self, value: Any) -> str | None:
        return session_result_data.serialize_json_payload(self, value)
