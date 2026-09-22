from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from services.updater.livertox_download import (
    UpstreamHumanVerificationRequired,
    resolve_master_list_from_bookshelf,
    resolve_master_list_url,
)


def test_bookshelf_resolver_uses_current_master_list_link() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        if request.method == "HEAD" and request.url.path.endswith(
            "masterlist02-26.xlsx"
        ):
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                },
                request=request,
            )
        if request.method == "GET" and request.url.path == "/books/NBK571102/":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                text=(
                    '<a href="/books/NBK571102/bin/masterlist02-26.xlsx">'
                    "Current Master List</a>"
                ),
                request=request,
            )
        return httpx.Response(405, request=request)

    async def resolve() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await resolve_master_list_from_bookshelf(
                SimpleNamespace(http_headers={}), client
            )

    result = asyncio.run(resolve())

    assert result == (
        "https://www.ncbi.nlm.nih.gov/books/NBK571102/bin/masterlist02-26.xlsx"
    )
    assert ("GET", "https://www.ncbi.nlm.nih.gov/books/NBK571102/") in requests
    assert not any("report=excel" in url for _, url in requests)


def test_bookshelf_captcha_stops_legacy_source_fallbacks() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, str(request.url)))
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text=(
                '<base href="https://www.google.com/recaptcha/challengepage/">'
                "verification required"
            ),
            request=request,
        )

    async def resolve() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await resolve_master_list_url(
                SimpleNamespace(
                    http_headers={},
                    base_url="https://ftp.ncbi.nlm.nih.gov/pub/litarch/29/31/",
                ),
                client,
            )

    with pytest.raises(UpstreamHumanVerificationRequired, match="CAPTCHA"):
        asyncio.run(resolve())

    assert requests
    assert all("/books/NBK571102/" in url for _, url in requests)
    assert not any("catalog.data.gov" in url or "/bin/" in url for _, url in requests)
