from __future__ import annotations

from typing import Any

SESSION_REVISION_DISABLED_MESSAGE = (
    "Session revision workflow has been intentionally removed and is pending rewrite."
)


class SessionRevisionNotImplementedError(NotImplementedError):
    pass


class InspectionRevisionScaffoldMixin:

    # -------------------------------------------------------------------------
    def raise_session_revision_not_implemented(self) -> None:
        raise SessionRevisionNotImplementedError(SESSION_REVISION_DISABLED_MESSAGE)

    # -------------------------------------------------------------------------
    def start_revision_job(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.raise_session_revision_not_implemented()

    # -------------------------------------------------------------------------
    def retry_revision_job(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        self.raise_session_revision_not_implemented()

    # -------------------------------------------------------------------------
    def get_revision_run(self, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        self.raise_session_revision_not_implemented()

    # -------------------------------------------------------------------------
    def list_revision_steps(self, *_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        self.raise_session_revision_not_implemented()

    # -------------------------------------------------------------------------
    def list_revision_artifacts(
        self, *_args: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        self.raise_session_revision_not_implemented()

    # -------------------------------------------------------------------------
    def list_revision_entities(
        self, *_args: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        self.raise_session_revision_not_implemented()

    # -------------------------------------------------------------------------
    def list_revision_reviews(
        self, *_args: Any, **_kwargs: Any
    ) -> list[dict[str, Any]]:
        self.raise_session_revision_not_implemented()

    # -------------------------------------------------------------------------
    def update_revision_clinical_review(
        self, *_args: Any, **_kwargs: Any
    ) -> dict[str, Any] | None:
        self.raise_session_revision_not_implemented()
