from __future__ import annotations

from domain.keys import (
    AccessKeyRecord,
    AccessKeyResponse,
    ProviderName,
    normalize_provider_name,
)
from repositories.serialization.access_keys import AccessKeySerializer


###############################################################################
class AccessKeyService:
    def __init__(self, serializer: AccessKeySerializer | None = None) -> None:
        self.serializer = serializer or AccessKeySerializer()

    # -------------------------------------------------------------------------
    @staticmethod
    def to_response(row: AccessKeyRecord) -> AccessKeyResponse:
        return AccessKeyResponse(
            id=int(row.id),
            provider=normalize_provider_name(str(row.provider)),
            is_active=bool(row.is_active),
            fingerprint=str(row.key_fingerprint),
            created_at=row.created_at,
            updated_at=row.updated_at,
            last_used_at=row.last_used_at,
        )

    # -------------------------------------------------------------------------
    def list_access_keys(self, provider: ProviderName) -> list[AccessKeyResponse]:
        rows = self.serializer.list_keys(provider)
        return [self.to_response(row) for row in rows]

    # -------------------------------------------------------------------------
    def create_access_key(
        self, provider: ProviderName, access_key: str
    ) -> AccessKeyResponse:
        created = self.serializer.create_key(provider, access_key)
        return self.to_response(created)

    # -------------------------------------------------------------------------
    def activate_access_key(
        self, key_id: int, provider: ProviderName
    ) -> AccessKeyResponse:
        row = self.serializer.activate_key(key_id, provider=provider)
        return self.to_response(row)

    # -------------------------------------------------------------------------
    def delete_access_key(self, key_id: int, provider: ProviderName) -> bool:
        return bool(self.serializer.delete_key(key_id, provider=provider))

