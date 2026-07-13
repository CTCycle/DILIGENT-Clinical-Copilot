from __future__ import annotations

from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel

from domain.llm.providers import CloudModelDescriptor

T = TypeVar("T", bound=BaseModel)


###############################################################################
class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]]
    options: dict[str, Any] = {}
    json_mode: bool = False


###############################################################################
class ChatResult(BaseModel):
    content: str
    reasoning_content: str | None = None


###############################################################################
class StructuredRequest(BaseModel, Generic[T]):
    model: str
    messages: list[dict[str, str]]
    schema_type: type[T]


###############################################################################
class ConnectivityResult(BaseModel):
    ok: bool
    response_preview: str | None = None
    error: str | None = None


###############################################################################
class EmbeddingRequest(BaseModel):
    model: str
    inputs: list[str]


###############################################################################
class CloudTransport(Protocol):

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult: ...

    # -------------------------------------------------------------------------
    async def structured(self, request: StructuredRequest[T]) -> T: ...

    # -------------------------------------------------------------------------
    async def list_models(self) -> list[CloudModelDescriptor]: ...

    # -------------------------------------------------------------------------
    async def check_connectivity(self, model: str) -> ConnectivityResult: ...

    # -------------------------------------------------------------------------
    async def close(self) -> None: ...


###############################################################################
class EmbeddingTransport(Protocol):

    # -------------------------------------------------------------------------
    async def embed(self, request: EmbeddingRequest) -> list[list[float]]: ...


###############################################################################
class StructuredTransportMixin:

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    async def structured(self, request: StructuredRequest[T]) -> T:
        result = await self.chat(
            ChatRequest(
                model=request.model,
                messages=request.messages,
                json_mode=True,
            )
        )
        return request.schema_type.model_validate_json(result.content)
