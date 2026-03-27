from __future__ import annotations

import asyncio
import uuid

import httpx
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from DILIGENT.server.common.utils.logger import logger


REQUEST_ID_HEADER = "X-Request-ID"
GENERIC_FAILURE_MESSAGE = "Request could not be completed. Please retry."
TIMEOUT_FAILURE_MESSAGE = "Request timed out. Please retry."
DEPENDENCY_FAILURE_MESSAGE = "Service dependency unavailable. Please retry shortly."
MISSING_RESOURCE_MESSAGE = "Required resource was not found."


###############################################################################
def resolve_request_id(request: Request) -> str:
    candidate = getattr(request.state, "request_id", "")
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    return "n/a"


###############################################################################
def build_error_payload(
    *,
    detail: str,
    request_id: str,
    retryable: bool,
) -> dict[str, object]:
    return {
        "detail": detail,
        "request_id": request_id,
        "retryable": retryable,
    }


###############################################################################
class RequestIdMiddleware(BaseHTTPMiddleware):
    # -------------------------------------------------------------------------
    async def dispatch(self, request: Request, call_next):
        request.state.request_id = uuid.uuid4().hex[:12]
        response = await call_next(request)
        response.headers.setdefault(REQUEST_ID_HEADER, resolve_request_id(request))
        return response


###############################################################################
def request_validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = resolve_request_id(request)
    logger.info(
        "Request validation failed request_id=%s method=%s path=%s",
        request_id,
        request.method,
        request.url.path,
    )
    payload: dict[str, object] = {
        "detail": exc.errors(),
        "request_id": request_id,
        "retryable": False,
    }
    response = JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=payload,
    )
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    return response


###############################################################################
def timeout_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = resolve_request_id(request)
    logger.warning(
        "Timeout request_id=%s method=%s path=%s type=%s",
        request_id,
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    response = JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content=build_error_payload(
            detail=TIMEOUT_FAILURE_MESSAGE,
            request_id=request_id,
            retryable=True,
        ),
    )
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    return response


###############################################################################
def dependency_error_handler(
    request: Request,
    exc: httpx.RequestError,
) -> JSONResponse:
    request_id = resolve_request_id(request)
    logger.warning(
        "Dependency failure request_id=%s method=%s path=%s type=%s",
        request_id,
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    response = JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=build_error_payload(
            detail=DEPENDENCY_FAILURE_MESSAGE,
            request_id=request_id,
            retryable=True,
        ),
    )
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    return response


###############################################################################
def missing_resource_error_handler(
    request: Request,
    exc: FileNotFoundError,
) -> JSONResponse:
    request_id = resolve_request_id(request)
    logger.warning(
        "Missing resource request_id=%s method=%s path=%s type=%s",
        request_id,
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    response = JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=build_error_payload(
            detail=MISSING_RESOURCE_MESSAGE,
            request_id=request_id,
            retryable=False,
        ),
    )
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    return response


###############################################################################
def unhandled_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = resolve_request_id(request)
    logger.exception(
        "Unhandled API exception request_id=%s method=%s path=%s type=%s",
        request_id,
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    response = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=build_error_payload(
            detail=GENERIC_FAILURE_MESSAGE,
            request_id=request_id,
            retryable=True,
        ),
    )
    response.headers.setdefault(REQUEST_ID_HEADER, request_id)
    return response


###############################################################################
def register_error_handling(app: FastAPI) -> None:
    app.add_middleware(RequestIdMiddleware)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)
    app.add_exception_handler(TimeoutError, timeout_error_handler)
    app.add_exception_handler(asyncio.TimeoutError, timeout_error_handler)
    app.add_exception_handler(httpx.TimeoutException, timeout_error_handler)
    app.add_exception_handler(httpx.RequestError, dependency_error_handler)
    app.add_exception_handler(FileNotFoundError, missing_resource_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
