# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx

from common.constants import (
    LIVERTOX_MASTER_LIST_ACCESSION,
    NCBI_BOOKS_OAI_BASE_URL,
    NCBI_EUTILS_BASE_URL,
    NCBI_TOOL_NAME,
)


class NCBIClientError(RuntimeError):
    """Raised when an official NCBI machine-access response is unusable."""


###############################################################################
async def search_bookshelf_record(
    client: httpx.AsyncClient,
    *,
    contact_email: str,
    accession: str = LIVERTOX_MASTER_LIST_ACCESSION,
) -> dict[str, Any]:
    params = {
        "db": "books",
        "term": accession,
        "retmode": "json",
        "tool": NCBI_TOOL_NAME,
        "email": contact_email,
    }
    payload = await _get_json(client, "esearch.fcgi", params)
    id_list = _extract_id_list(payload)
    if not id_list:
        raise NCBIClientError("NCBI Bookshelf returned no matching record.")
    return await summarize_bookshelf_record(
        client,
        id_list,
        contact_email=contact_email,
        accession=accession,
    )


###############################################################################
async def summarize_bookshelf_record(
    client: httpx.AsyncClient,
    uids: Iterable[str],
    *,
    contact_email: str,
    accession: str = LIVERTOX_MASTER_LIST_ACCESSION,
) -> dict[str, Any]:
    normalized_uids = [str(uid).strip() for uid in uids if str(uid).strip()]
    if not normalized_uids:
        raise NCBIClientError("NCBI Bookshelf returned no record identifiers.")
    payload = await _get_json(
        client,
        "esummary.fcgi",
        {
            "db": "books",
            "id": ",".join(normalized_uids),
            "retmode": "json",
            "tool": NCBI_TOOL_NAME,
            "email": contact_email,
        },
    )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise NCBIClientError("NCBI Bookshelf returned an invalid summary.")

    matches: list[dict[str, Any]] = []
    for uid in normalized_uids:
        record = result.get(uid)
        if not isinstance(record, dict):
            continue
        rid = record.get("RID", record.get("rid"))
        if isinstance(rid, str) and rid.strip().upper() == accession.upper():
            matches.append(record)
    if not matches:
        raise NCBIClientError("NCBI Bookshelf returned an unexpected record.")
    if len(matches) != 1:
        raise NCBIClientError("NCBI Bookshelf returned conflicting records.")
    return matches[0]


###############################################################################
async def resolve_books_oai_master_list_url(
    client: httpx.AsyncClient,
    *,
    accession: str = LIVERTOX_MASTER_LIST_ACCESSION,
) -> str | None:
    identifier = f"oai:books.ncbi.nlm.nih.gov:{_numeric_accession(accession)}"
    formats = await list_metadata_formats(client, identifier=identifier)
    if "nbk_ftext" not in formats:
        return None
    xml_payload = await get_books_oai_record(
        client,
        identifier=identifier,
        metadata_prefix="nbk_ftext",
    )
    candidates = _extract_master_list_candidates(xml_payload, accession=accession)
    if len(candidates) > 1:
        raise NCBIClientError("Books-OAI returned conflicting master-list links.")
    return next(iter(candidates), None)


###############################################################################
async def list_metadata_formats(
    client: httpx.AsyncClient,
    *,
    identifier: str,
) -> set[str]:
    xml_payload = await _get_text(
        client,
        NCBI_BOOKS_OAI_BASE_URL,
        {"verb": "ListMetadataFormats", "identifier": identifier},
    )
    root = _parse_xml(xml_payload, "Books-OAI metadata formats")
    return {
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element.tag) == "metadataPrefix" and (element.text or "").strip()
    }


###############################################################################
async def get_books_oai_record(
    client: httpx.AsyncClient,
    *,
    identifier: str,
    metadata_prefix: str,
) -> str:
    return await _get_text(
        client,
        NCBI_BOOKS_OAI_BASE_URL,
        {
            "verb": "GetRecord",
            "identifier": identifier,
            "metadataPrefix": metadata_prefix,
        },
    )


###############################################################################
def _extract_master_list_candidates(xml_payload: str, *, accession: str) -> set[str]:
    root = _parse_xml(xml_payload, "Books-OAI full-text record")
    candidates: set[str] = set()
    for element in root.iter():
        values = list(element.attrib.values())
        if _local_name(element.tag) in {
            "link",
            "ext-link",
            "media",
            "supplementary-material",
            "uri",
            "url",
        }:
            values.append(element.text or "")
        for value in values:
            raw_value = value.strip()
            if (
                raw_value
                and not raw_value.startswith(("/", "//"))
                and not urlsplit(raw_value).scheme
            ):
                raw_value = urljoin(
                    f"https://www.ncbi.nlm.nih.gov/books/{accession}/bin/",
                    raw_value,
                )
            candidate = _validate_master_list_url(raw_value, accession=accession)
            if candidate is not None:
                candidates.add(candidate)
    return candidates


###############################################################################
def _validate_master_list_url(value: str, *, accession: str) -> str | None:
    raw_value = value.strip()
    if not raw_value:
        return None
    try:
        parsed = urlsplit(raw_value)
    except ValueError:
        return None
    if parsed.scheme.lower() != "https" or parsed.hostname != "www.ncbi.nlm.nih.gov":
        return None
    expected_prefix = f"/books/{accession}/"
    if not parsed.path.startswith(expected_prefix):
        return None
    filename = unquote(parsed.path.rsplit("/", maxsplit=1)[-1])
    if not filename.lower().startswith("masterlist") or not filename.lower().endswith(
        ".xlsx"
    ):
        return None
    return urlunsplit(parsed)


###############################################################################
async def _get_json(
    client: httpx.AsyncClient,
    endpoint: str,
    params: dict[str, str],
) -> dict[str, Any]:
    try:
        response = await client.get(f"{NCBI_EUTILS_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise NCBIClientError(
            "NCBI E-utilities returned an unusable response."
        ) from exc
    if not isinstance(payload, dict):
        raise NCBIClientError("NCBI E-utilities returned an invalid JSON object.")
    return payload


###############################################################################
async def _get_text(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, str],
) -> str:
    try:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as exc:
        raise NCBIClientError("NCBI machine-access request failed.") from exc


###############################################################################
def _extract_id_list(payload: dict[str, Any]) -> list[str]:
    result = payload.get("esearchresult")
    if not isinstance(result, dict):
        raise NCBIClientError("NCBI ESearch returned an invalid result.")
    id_list = result.get("idlist")
    if not isinstance(id_list, list) or not all(
        isinstance(item, str) for item in id_list
    ):
        raise NCBIClientError("NCBI ESearch returned an invalid identifier list.")
    return [item.strip() for item in id_list if item.strip()]


###############################################################################
def _numeric_accession(accession: str) -> str:
    if not accession.upper().startswith("NBK") or not accession[3:].isdigit():
        raise NCBIClientError("NCBI Bookshelf accession is invalid.")
    return accession[3:]


###############################################################################
def _parse_xml(payload: str, description: str) -> ElementTree.Element:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise NCBIClientError(f"{description} was malformed.") from exc
    if any(_local_name(element.tag) == "error" for element in root.iter()):
        raise NCBIClientError(f"{description} returned an error.")
    return root


###############################################################################
def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]
