# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from openpyxl import Workbook

from common.constants import DEFAULT_NCBI_CONTACT_EMAIL
from services.updater import livertox_download
from services.updater.livertox_common import build_ncbi_http_headers
from services.updater.ncbi_client import (
    NCBIClientError,
    resolve_books_oai_master_list_url,
    search_bookshelf_record,
)


MASTER_URL = "https://www.ncbi.nlm.nih.gov/books/NBK571102/bin/masterlist99-27.xlsx"
ARCHIVE_PATH = "cc/8d/livertox_NBK547852.tar.gz"
ARCHIVE_URL = f"https://ftp.ncbi.nlm.nih.gov/pub/litarch/{ARCHIVE_PATH}"


def _response_json(request: httpx.Request, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"Content-Type": "application/json"},
        content=json.dumps(payload).encode(),
        request=request,
    )


def _master_xlsx_bytes() -> bytes:
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["LiverTox Master List"])
    sheet.append(
        [
            "Count",
            "Ingredient",
            "Brand Name",
            "Likelihood Score",
            "Chapter Title",
            "Last Update",
            "Year Approved",
            "Type of Agent",
            "In LiverTox",
            "Primary Classification",
            "Secondary Classification",
        ]
    )
    sheet.append(
        [
            1,
            "Acetaminophen",
            "Tylenol",
            "A",
            "Acetaminophen",
            "2026-01-01",
            1955,
            "Drug",
            "Yes",
            "Analgesic",
            "Other",
        ]
    )
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def _archive_bytes(*, include_master: bool = False, unsafe_member: bool = False) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        monograph = tarfile.TarInfo("NBK000001.html")
        monograph_payload = b"<title>Acetaminophen</title><p>LiverTox excerpt</p>"
        monograph.size = len(monograph_payload)
        archive.addfile(monograph, io.BytesIO(monograph_payload))
        if include_master:
            master = tarfile.TarInfo("data/masterlist99-27.xlsx")
            master_payload = _master_xlsx_bytes()
            master.size = len(master_payload)
            archive.addfile(master, io.BytesIO(master_payload))
        if unsafe_member:
            unsafe = tarfile.TarInfo("../masterlist-unsafe.xlsx")
            unsafe_payload = b"unsafe"
            unsafe.size = len(unsafe_payload)
            archive.addfile(unsafe, io.BytesIO(unsafe_payload))
    return output.getvalue()


def _updater(tmp_path: Path, *, email: str = DEFAULT_NCBI_CONTACT_EMAIL) -> SimpleNamespace:
    return SimpleNamespace(
        http_headers=build_ncbi_http_headers(email),
        ncbi_contact_email=email,
        sources_path=str(tmp_path),
        redownload=True,
        delay=0,
        chunk_size=64,
        header_row=1,
        file_name="livertox_NBK547852.tar.gz",
        master_list_path=str(tmp_path / "LiverTox_Master_List.xlsx"),
        master_list_metadata_path=str(tmp_path / "master.metadata.json"),
        archive_metadata_path=str(tmp_path / "archive.metadata.json"),
        supported_extensions=(".html", ".htm", ".xhtml", ".xml", ".nxml", ".pdf"),
        resolved_archive_url=None,
    )


def test_esearch_and_esummary_use_ncbi_identity_and_exact_rid() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("esearch.fcgi"):
            return _response_json(request, {"esearchresult": {"idlist": ["987"]}})
        if request.url.path.endswith("esummary.fcgi"):
            return _response_json(
                request,
                {"result": {"uids": ["987"], "987": {"RID": "NBK571102"}}},
            )
        return httpx.Response(404, request=request)

    async def resolve() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await search_bookshelf_record(
                client,
                contact_email="developer@example.org",
            )

    record = asyncio.run(resolve())

    assert record["RID"] == "NBK571102"
    assert requests[0].url.params["db"] == "books"
    assert requests[0].url.params["term"] == "NBK571102"
    assert requests[0].url.params["tool"] == "DILIGENTClinicalCopilot"
    assert requests[0].url.params["email"] == "developer@example.org"
    assert requests[1].url.params["email"] == "developer@example.org"


def test_esummary_rejects_missing_or_conflicting_rid() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return _response_json(request, {"esearchresult": {"idlist": ["1", "2"]}})
        return _response_json(
            request,
            {
                "result": {
                    "uids": ["1", "2"],
                    "1": {"RID": "NBK571102"},
                    "2": {"RID": "NBK571102"},
                }
            },
        )

    async def resolve() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await search_bookshelf_record(client, contact_email=DEFAULT_NCBI_CONTACT_EMAIL)

    with pytest.raises(NCBIClientError, match="conflicting"):
        asyncio.run(resolve())


def _oai_handler(
    xlsx: bytes,
    *,
    link: str = MASTER_URL,
) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return _response_json(request, {"esearchresult": {"idlist": ["987"]}})
        if request.url.path.endswith("esummary.fcgi"):
            return _response_json(
                request,
                {"result": {"uids": ["987"], "987": {"rid": "NBK571102"}}},
            )
        if request.url.path == "/lit/oai/books/":
            if request.url.params["verb"] == "ListMetadataFormats":
                return httpx.Response(
                    200,
                    text=(
                        "<OAI-PMH><ListMetadataFormats>"
                        "<metadataPrefix>nbk_meta</metadataPrefix>"
                        "<metadataPrefix>nbk_ftext</metadataPrefix>"
                        "</ListMetadataFormats></OAI-PMH>"
                    ),
                    request=request,
                )
            return httpx.Response(
                200,
                text=(
                    "<OAI-PMH><GetRecord><record><metadata>"
                    f'<supplementary-material href="{link}" />'
                    "</metadata></record></GetRecord></OAI-PMH>"
                ),
                request=request,
            )
        if request.method == "HEAD" and request.url == httpx.URL(MASTER_URL):
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(xlsx)), "Last-Modified": "Wed, 01 Jan 2026 00:00:00 GMT"},
                request=request,
            )
        if request.method == "GET" and request.url == httpx.URL(MASTER_URL):
            return httpx.Response(200, content=xlsx, request=request)
        return httpx.Response(404, request=request)

    return handler


def test_books_oai_resolves_dynamic_master_list_link_and_identifier() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _oai_handler(b"")(request)

    async def resolve() -> str | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await resolve_books_oai_master_list_url(client)

    assert asyncio.run(resolve()) == MASTER_URL
    assert requests[0].url.params["identifier"] == "oai:books.ncbi.nlm.nih.gov:571102"
    assert requests[1].url.params["identifier"] == "oai:books.ncbi.nlm.nih.gov:571102"
    assert requests[1].url.params["metadataPrefix"] == "nbk_ftext"


def test_books_oai_resolves_safe_relative_master_list_link() -> None:
    async def resolve() -> str | None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                _oai_handler(b"", link="masterlist02-26.xlsx")
            )
        ) as client:
            return await resolve_books_oai_master_list_url(client)

    assert asyncio.run(resolve()) == MASTER_URL.replace(
        "masterlist99-27", "masterlist02-26"
    )


@pytest.mark.parametrize(
    "link",
    [
        "https://evil.example/masterlist.xlsx",
        "https://www.ncbi.nlm.nih.gov/books/NBK000001/bin/masterlist.xlsx",
    ],
)
def test_books_oai_rejects_foreign_or_wrong_book_links(link: str) -> None:
    async def resolve() -> str | None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_oai_handler(b"", link=link))
        ) as client:
            return await resolve_books_oai_master_list_url(client)

    assert asyncio.run(resolve()) is None


def test_missing_oai_spreadsheet_uses_litarch_fallback(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    archive_payload = _archive_bytes(include_master=True)
    updater = _updater(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("esearch.fcgi"):
            return _response_json(request, {"esearchresult": {"idlist": ["987"]}})
        if request.url.path.endswith("esummary.fcgi"):
            return _response_json(
                request,
                {"result": {"uids": ["987"], "987": {"rid": "NBK571102"}}},
            )
        if request.url.path == "/lit/oai/books/":
            if request.url.params["verb"] == "ListMetadataFormats":
                return httpx.Response(
                    200,
                    text="<OAI-PMH><metadataPrefix>nbk_ftext</metadataPrefix></OAI-PMH>",
                    request=request,
                )
            return httpx.Response(
                200,
                text='<OAI-PMH><link href="https://evil.example/masterlist.xlsx" /></OAI-PMH>',
                request=request,
            )
        if request.url == httpx.URL(livertox_download.NLM_LITARCH_FILE_LIST_URL):
            return httpx.Response(
                200,
                text=f"path,title,accession\n{ARCHIVE_PATH},LiverTox,NBK547852\n",
                request=request,
            )
        if request.method == "HEAD" and request.url == httpx.URL(ARCHIVE_URL):
            return httpx.Response(
                200,
                headers={
                    "Content-Length": str(len(archive_payload)),
                    "Last-Modified": "Wed, 01 Jan 2026 00:00:00 GMT",
                },
                request=request,
            )
        if request.method == "GET" and request.url == httpx.URL(ARCHIVE_URL):
            return httpx.Response(200, content=archive_payload, request=request)
        return httpx.Response(404, request=request)

    monkeypatch.setattr(
        livertox_download,
        "_http_client",
        lambda _self: httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ),
    )
    result = asyncio.run(livertox_download.download_master_list(updater))

    assert result["downloaded"] is True
    assert result["source_member"] == "data/masterlist99-27.xlsx"
    assert Path(result["file_path"]).is_file()
    assert not any(
        request.url.path == "/books/NBK571102/" for request in requests
    )


def test_litarch_file_list_resolves_exact_accession_and_dynamic_directory() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=f"path,title,accession\n{ARCHIVE_PATH},LiverTox,NBK547852\n",
            request=request,
        )

    async def resolve() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await livertox_download.resolve_litarch_archive(client)

    assert asyncio.run(resolve()) == ARCHIVE_URL


def test_litarch_file_list_rejects_conflicting_and_unsafe_paths() -> None:
    async def resolve(csv_payload: str) -> str:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=csv_payload, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await livertox_download.resolve_litarch_archive(client)

    with pytest.raises(RuntimeError, match="conflicting"):
        asyncio.run(
            resolve(
                "path,title,accession\n"
                "aa/aa/livertox_NBK547852.tar.gz,LiverTox,NBK547852\n"
                "bb/bb/livertox_NBK547852.tar.gz,LiverTox,NBK547852\n"
            )
        )
    with pytest.raises(RuntimeError, match="unsafe"):
        asyncio.run(resolve("../livertox_NBK547852.tar.gz,LiverTox,NBK547852\n"))


def test_master_list_extraction_rejects_traversal_and_ambiguity(tmp_path: Path) -> None:
    safe_archive = tmp_path / "safe.tar.gz"
    safe_archive.write_bytes(_archive_bytes(include_master=True))
    assert livertox_download.find_master_list_member(safe_archive) == "data/masterlist99-27.xlsx"

    unsafe_archive = tmp_path / "unsafe.tar.gz"
    unsafe_archive.write_bytes(_archive_bytes(unsafe_member=True))
    with pytest.raises(RuntimeError, match="unsafe"):
        livertox_download.find_master_list_member(unsafe_archive)

    ambiguous = tmp_path / "ambiguous.tar.gz"
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name in ("a/masterlist-one.xlsx", "b/masterlist-two.xlsx"):
            member = tarfile.TarInfo(name)
            payload = _master_xlsx_bytes()
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    ambiguous.write_bytes(output.getvalue())
    with pytest.raises(RuntimeError, match="conflicting"):
        livertox_download.find_master_list_member(ambiguous)


def test_archive_download_is_dynamic_and_skips_unchanged_payload(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    archive_payload = _archive_bytes()
    requests: list[httpx.Request] = []
    updater = _updater(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url == httpx.URL(livertox_download.NLM_LITARCH_FILE_LIST_URL):
            return httpx.Response(200, text=f"{ARCHIVE_PATH},LiverTox,NBK547852\n", request=request)
        if request.method == "HEAD" and request.url == httpx.URL(ARCHIVE_URL):
            return httpx.Response(
                200,
                headers={"Content-Length": str(len(archive_payload)), "Last-Modified": "Wed, 01 Jan 2026 00:00:00 GMT"},
                request=request,
            )
        if request.method == "GET" and request.url == httpx.URL(ARCHIVE_URL):
            return httpx.Response(200, content=archive_payload, request=request)
        return httpx.Response(404, request=request)

    def client_factory(_self: object) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)

    monkeypatch.setattr(livertox_download, "_http_client", client_factory)
    first = asyncio.run(livertox_download.download_bulk_data(updater, str(tmp_path)))
    updater.redownload = False
    second = asyncio.run(livertox_download.download_bulk_data(updater, str(tmp_path)))

    assert first["downloaded"] is True
    assert second["downloaded"] is False
    assert sum(request.method == "GET" and request.url == httpx.URL(ARCHIVE_URL) for request in requests) == 1
    assert not (tmp_path / "livertox_NBK547852.tar.gz.part").exists()


def test_partial_archive_download_preserves_last_good_file(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    destination = tmp_path / "livertox_NBK547852.tar.gz"
    original = _archive_bytes()
    destination.write_bytes(original)
    updater = _updater(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url == httpx.URL(livertox_download.NLM_LITARCH_FILE_LIST_URL):
            return httpx.Response(200, text=f"{ARCHIVE_PATH},LiverTox,NBK547852\n", request=request)
        if request.method == "HEAD" and request.url == httpx.URL(ARCHIVE_URL):
            return httpx.Response(200, headers={"Content-Length": "8"}, request=request)
        if request.method == "GET" and request.url == httpx.URL(ARCHIVE_URL):
            return httpx.Response(200, content=b"not-a-tar", request=request)
        return httpx.Response(404, request=request)

    monkeypatch.setattr(
        livertox_download,
        "_http_client",
        lambda _self: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RuntimeError, match="safely"):
        asyncio.run(livertox_download.download_bulk_data(updater, str(tmp_path)))
    assert destination.read_bytes() == original
    assert not (tmp_path / "livertox_NBK547852.tar.gz.part").exists()


def test_partial_master_list_download_preserves_last_good_file(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    destination = tmp_path / "LiverTox_Master_List.xlsx"
    original = _master_xlsx_bytes()
    destination.write_bytes(original)
    updater = _updater(tmp_path)
    updater.redownload = True

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return _response_json(request, {"esearchresult": {"idlist": ["987"]}})
        if request.url.path.endswith("esummary.fcgi"):
            return _response_json(request, {"result": {"uids": ["987"], "987": {"rid": "NBK571102"}}})
        if request.url.path == "/lit/oai/books/":
            if request.url.params["verb"] == "ListMetadataFormats":
                return httpx.Response(200, text="<OAI-PMH><metadataPrefix>nbk_ftext</metadataPrefix></OAI-PMH>", request=request)
            return httpx.Response(200, text=f'<OAI-PMH><link href="{MASTER_URL}" /></OAI-PMH>', request=request)
        if request.method == "HEAD" and request.url == httpx.URL(MASTER_URL):
            return httpx.Response(200, headers={"Content-Length": "8"}, request=request)
        if request.method == "GET" and request.url == httpx.URL(MASTER_URL):
            return httpx.Response(200, content=b"not-an-xlsx", request=request)
        return httpx.Response(404, request=request)

    monkeypatch.setattr(
        livertox_download,
        "_http_client",
        lambda _self: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(RuntimeError, match="file-list"):
        asyncio.run(livertox_download.download_master_list(updater))
    assert destination.read_bytes() == original
    assert not (tmp_path / "LiverTox_Master_List.xlsx.part").exists()


def test_ncbi_failures_are_sanitized_and_do_not_return_contact_email() -> None:
    contact = "developer@example.org"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json", request=request)

    async def resolve() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await search_bookshelf_record(client, contact_email=contact)

    with pytest.raises(NCBIClientError) as error:
        asyncio.run(resolve())
    assert contact not in str(error.value)
    assert contact in build_ncbi_http_headers(contact)["User-Agent"]


def test_ncbi_contact_is_not_written_to_discovery_logs(caplog) -> None:  # type: ignore[no-untyped-def]
    contact = "developer@example.org"
    updater = SimpleNamespace(ncbi_contact_email=contact)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return _response_json(request, {"esearchresult": {"idlist": ["987"]}})
        if request.url.path.endswith("esummary.fcgi"):
            return _response_json(
                request,
                {"result": {"uids": ["987"], "987": {"rid": "NBK571102"}}},
            )
        return httpx.Response(503, request=request)

    async def resolve() -> str | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await livertox_download.resolve_master_list_url(updater, client)

    assert asyncio.run(resolve()) is None
    assert contact not in caplog.text
