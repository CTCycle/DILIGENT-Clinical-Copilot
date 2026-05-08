from __future__ import annotations

from typing import Any


###############################################################################
class ServiceError(Exception):
    status_code = 500
    retryable = True
    default_detail = "Operation failed unexpectedly. Please retry."

    def __init__(
        self,
        detail: Any | None = None,
        *,
        status_code: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.detail = self.default_detail if detail is None else detail
        if status_code is not None:
            self.status_code = int(status_code)
        if retryable is not None:
            self.retryable = bool(retryable)
        super().__init__(str(self.detail))


###############################################################################
class ServiceValidationError(ServiceError):
    status_code = 422
    retryable = False
    default_detail = "Request validation failed."


###############################################################################
class ServiceNotFoundError(ServiceError):
    status_code = 404
    retryable = False
    default_detail = "Required resource was not found."


###############################################################################
class ServiceConflictError(ServiceError):
    status_code = 409
    retryable = False
    default_detail = "Operation conflicts with current resource state."


###############################################################################
class ServiceDependencyError(ServiceError):
    status_code = 503
    retryable = True
    default_detail = "Service dependency unavailable. Please retry shortly."
