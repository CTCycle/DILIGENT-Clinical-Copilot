from __future__ import annotations

from services.catalogs.runtime import get_reference_catalog_snapshot


def get_sensitive_error_tokens() -> tuple[str, ...]:
    snapshot = get_reference_catalog_snapshot()
    return tuple(
        value.lower()
        for value in snapshot.values(
            "security_text_filters",
            "sensitive_error_tokens",
            key="default",
        )
    )
