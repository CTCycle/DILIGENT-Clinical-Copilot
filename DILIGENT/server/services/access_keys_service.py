from __future__ import annotations

from typing import Literal

from DILIGENT.server.domain.keys import AccessKeyResponse
from DILIGENT.server.repositories.serialization.access_keys import AccessKeySerializer
from DILIGENT.server.repositories.schemas.models import AccessKey, ResearchAccessKey

ProviderName = Literal["openai", "gemini", "tavily"]


###############################################################################
class AccessKeyService:
    def __init__(self, serializer: AccessKeySerializer | None = None) -> None:
        self.serializer = serializer or AccessKeySerializer()

    # -------------------------------------------------------------------------
    @staticmethod
    def to_response(row: AccessKey | ResearchAccessKey) -> AccessKeyResponse:
        return AccessKeyResponse(
            id=int(row.id),
            provider=str(row.provider),  # type: ignore[arg-type]
            is_active=bool(row.is_active),
            fingerprint=str(row.fingerprint),
            created_at=row.created_at,
            updated_at=row.updated_at,
            last_used_at=row.last_used_at,
        )

    # -------------------------------------------------------------------------
    def list_access_keys(self, provider: ProviderName) -> list[AccessKeyResponse]:
        rows = self.serializer.list_keys(provider)
        return [self.to_response(row) for row in rows]

    # -------------------------------------------------------------------------
    def create_access_key(self, provider: ProviderName, access_key: str) -> AccessKeyResponse:
        created = self.serializer.create_key(provider, access_key)
        return self.to_response(created)

    # -------------------------------------------------------------------------
    def activate_access_key(self, key_id: int, provider: ProviderName) -> AccessKeyResponse:
        row = self.serializer.activate_key(key_id, provider=provider)
        return self.to_response(row)

    # -------------------------------------------------------------------------
    def delete_access_key(self, key_id: int, provider: ProviderName) -> bool:
        return bool(self.serializer.delete_key(key_id, provider=provider))
