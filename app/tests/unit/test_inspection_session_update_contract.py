from __future__ import annotations

from api.inspection.sessions import InspectionSessionEndpoint
from services.inspection.service import DataInspectionService


###############################################################################
class FakeSerializer:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.report_calls: list[dict[str, object]] = []
        self.metadata_calls: list[dict[str, object]] = []

    # -------------------------------------------------------------------------
    def get_session_detail(self, session_id: int) -> dict[str, object]:
        return {"session_id": session_id, "path": "clinical"}

    # -------------------------------------------------------------------------
    def list_manual_report_edits(self, session_id: int) -> list[dict[str, object]]:
        _ = session_id
        return []

    # -------------------------------------------------------------------------
    def update_current_report_text_with_manual_audit(
        self,
        session_id: int,
        *,
        report_text: str,
        edited_fields: list[str] | None,
        reviewer_note: str | None,
        edited_by: str | None,
        metadata: dict[str, object] | None,
    ) -> dict[str, object]:
        call = {
            "session_id": session_id,
            "report_text": report_text,
            "edited_fields": edited_fields,
            "reviewer_note": reviewer_note,
            "edited_by": edited_by,
            "metadata": metadata,
        }
        self.report_calls.append(call)
        return {"session": {"session_id": session_id, "path": "report"}}

    # -------------------------------------------------------------------------
    def update_session_metadata(
        self,
        session_id: int,
        *,
        metadata: dict[str, object] | None,
    ) -> dict[str, object]:
        call = {
            "session_id": session_id,
            "metadata": metadata,
        }
        self.metadata_calls.append(call)
        return self.get_session_detail(session_id)


###############################################################################
class FakeRouter:
    # -------------------------------------------------------------------------
    def add_api_route(self, *args: object, **kwargs: object) -> None:
        _ = (args, kwargs)


###############################################################################
class FakeEndpointService:
    # -------------------------------------------------------------------------
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    # -------------------------------------------------------------------------
    def update_session(
        self,
        session_id: int,
        *,
        report_text: str | None = None,
        edited_fields: list[str] | None = None,
        reviewer_note: str | None = None,
        edited_by: str | None = None,
        metadata: dict[str, object] | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "session_id": session_id,
                "report_text": report_text,
                "edited_fields": edited_fields,
                "reviewer_note": reviewer_note,
                "edited_by": edited_by,
                "metadata": metadata,
            }
        )
        return {
            "session_id": session_id,
            "status": "successful",
            "metadata": metadata or {},
        }


###############################################################################
class FakeRequest:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        session_text: str | None = None,
        report_text: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.session_text = session_text
        self.report_text = report_text
        self.edited_fields = None
        self.reviewer_note = None
        self.edited_by = None
        self.metadata = metadata


###############################################################################
def build_service(serializer: FakeSerializer) -> DataInspectionService:
    return DataInspectionService(
        clinical_session_repository=serializer,
        drug_catalog_repository=serializer,
        knowledge_repository=serializer,
        session_timeline_repository=serializer,
        session_revision_repository=serializer,
        timeline_extractor=object(),
        jobs=object(),  # type: ignore[arg-type]
    )


###############################################################################
def test_update_session_without_report_text_updates_metadata_only() -> None:
    serializer = FakeSerializer()
    service = build_service(serializer)

    payload = service.update_session(
        7,
        report_text=None,
        metadata={"source": "manual"},
    )

    assert payload == {
        "session_id": 7,
        "path": "clinical",
        "manual_edit_history": [],
    }
    assert serializer.report_calls == []
    assert serializer.metadata_calls == [
        {"session_id": 7, "metadata": {"source": "manual"}}
    ]


###############################################################################
def test_update_session_with_report_text_updates_report_only() -> None:
    serializer = FakeSerializer()
    service = build_service(serializer)

    payload = service.update_session(
        9,
        report_text="new report",
        metadata={"source": "manual"},
    )

    assert payload == {
        "session_id": 9,
        "path": "clinical",
        "manual_edit_history": [],
    }
    assert serializer.report_calls == [
        {
            "session_id": 9,
            "report_text": "new report",
            "edited_fields": None,
            "reviewer_note": None,
            "edited_by": None,
            "metadata": {"source": "manual"},
        }
    ]
    assert serializer.metadata_calls == []


###############################################################################
def test_session_endpoint_does_not_treat_session_text_as_report_text() -> None:
    service = FakeEndpointService()
    endpoint = InspectionSessionEndpoint(router=FakeRouter(), service=service)  # type: ignore[arg-type]

    payload = endpoint.update_session(
        11,
        FakeRequest(
            session_text="legacy alias",
            report_text=None,
            metadata={"source": "endpoint"},
        ),
    )

    assert payload.session_id == 11
    assert service.calls == [
        {
            "session_id": 11,
            "report_text": None,
            "edited_fields": None,
            "reviewer_note": None,
            "edited_by": None,
            "metadata": {"source": "endpoint"},
        }
    ]
