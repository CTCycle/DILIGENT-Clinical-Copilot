# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import os
from urllib.parse import urlsplit


def parse_cors_allow_origins(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated list of exact HTTP(S) origins."""
    if value is None or not value.strip():
        return ()

    origins: list[str] = []
    seen: set[str] = set()
    for entry in value.split(","):
        candidate = entry.strip()
        if not candidate:
            continue
        if candidate == "*":
            raise ValueError("CORS_ALLOW_ORIGINS must contain explicit origins, not '*'.")

        try:
            parsed = urlsplit(candidate)
            port = parsed.port
            hostname = parsed.hostname
        except ValueError as exc:
            raise ValueError(
                "CORS_ALLOW_ORIGINS must contain valid HTTP(S) origins."
            ) from exc

        scheme = parsed.scheme.casefold()
        if (
            scheme not in {"http", "https"}
            or not parsed.netloc
            or not hostname
            or "*" in parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "CORS_ALLOW_ORIGINS must contain valid HTTP(S) origins without paths."
            )

        normalized_host = hostname.casefold()
        if ":" in normalized_host:
            normalized_host = f"[{normalized_host}]"
        default_port = 80 if scheme == "http" else 443
        port_suffix = f":{port}" if port is not None and port != default_port else ""
        origin = f"{scheme}://{normalized_host}{port_suffix}"
        if origin not in seen:
            origins.append(origin)
            seen.add(origin)
    return tuple(origins)


def get_cors_allow_origins() -> tuple[str, ...]:
    return parse_cors_allow_origins(os.getenv("CORS_ALLOW_ORIGINS"))
