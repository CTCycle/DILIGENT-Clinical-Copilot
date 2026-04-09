from __future__ import annotations

from DILIGENT.server.common.utils.types import coerce_int, coerce_str_or_none

OLLAMA_DEFAULT_HOST = "localhost"
OLLAMA_DEFAULT_PORT = 11434
OLLAMA_DEFAULT_SCHEME = "http"

###############################################################################
def _resolve_ollama_url_with_scheme(
    normalized_host: str,
    *,
    port_value: int | None,
) -> str:
    scheme, host_port = normalized_host.split("://", maxsplit=1)
    if ":" in host_port:
        host_only, parsed_port = host_port.split(":", maxsplit=1)
        resolved_port = (
            port_value
            if port_value is not None
            else coerce_int(parsed_port, OLLAMA_DEFAULT_PORT, minimum=1, maximum=65535)
        )
        return f"{scheme}://{host_only}:{resolved_port}"
    resolved_port = port_value if port_value is not None else OLLAMA_DEFAULT_PORT
    return f"{scheme}://{host_port}:{resolved_port}"

###############################################################################
def resolve_ollama_base_url(
    *,
    ollama_url: str | None,
    ollama_host: str | None,
    ollama_port: int | None,
    fallback: str = f"{OLLAMA_DEFAULT_SCHEME}://{OLLAMA_DEFAULT_HOST}:{OLLAMA_DEFAULT_PORT}",
) -> str:
    if ollama_url:
        return ollama_url.rstrip("/")
    host_value = coerce_str_or_none(ollama_host)
    port_value = ollama_port
    if host_value:
        normalized_host = host_value.strip().rstrip("/")
        if "://" in normalized_host:
            return _resolve_ollama_url_with_scheme(normalized_host, port_value=port_value)
        resolved_port = port_value if port_value is not None else OLLAMA_DEFAULT_PORT
        return f"{OLLAMA_DEFAULT_SCHEME}://{normalized_host}:{resolved_port}"
    if port_value is not None:
        return f"{OLLAMA_DEFAULT_SCHEME}://{OLLAMA_DEFAULT_HOST}:{port_value}"
    return fallback.rstrip("/")
