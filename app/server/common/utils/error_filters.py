# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from common.catalogs.provider import get_catalog_provider

###############################################################################
def get_sensitive_error_tokens() -> tuple[str, ...]:
    snapshot = get_catalog_provider().get_snapshot()
    return tuple(
        value.lower()
        for value in snapshot.values(
            "security_text_filters",
            "sensitive_error_tokens",
            key="default",
        )
    )
