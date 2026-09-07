from __future__ import annotations

from typing import Protocol

from domain.llm.providers import CloudModelDescriptor
from domain.llm.transports import (
    T,
    ChatRequest,
    ChatResult,
    ConnectivityResult,
    EmbeddingRequest,
    StructuredRequest,
)

###############################################################################
class CloudTransport(Protocol):

    # -------------------------------------------------------------------------
    async def chat(self, request: ChatRequest) -> ChatResult: ...

    # -------------------------------------------------------------------------
    async def structured(self, request: StructuredRequest[T]) -> T: ...

    # -------------------------------------------------------------------------
    async def list_models(
        self, *, force_refresh: bool = False
    ) -> list[CloudModelDescriptor]: ...

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
                options=request.options,
                json_mode=True,
                reasoning_level=request.reasoning_level,
                reasoning_parameter=request.reasoning_parameter,
                reasoning_reserve=request.reasoning_reserve,
                output_token_limit=request.output_token_limit,
            )
        )
        return request.schema_type.model_validate_json(result.content)
