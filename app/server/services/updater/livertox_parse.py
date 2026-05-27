from __future__ import annotations

import html
import io
import multiprocessing
import os
import re
import tarfile
import unicodedata
from collections.abc import Callable
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from typing import Any, cast

import pandas as pd
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader
from tqdm import tqdm

from common.utils.logger import logger
from services.text.normalization import normalize_whitespace
from services.updater import livertox_common


# -----------------------------------------------------------------------------
def process_monograph_payload(
    member_name: str,
    data: bytes,
) -> dict[str, str] | None:
    return process_monograph_member(member_name, data)


###############################################################################

# Extracted from the facade module; functions intentionally accept the facade instance.

def sanitize_livertox_master_list(self, data: pd.DataFrame) -> pd.DataFrame | None:
    if data.empty:
        return

    column_mapping = {
        "Count": "reference_count",
        "Ingredient": "ingredient",
        "Brand Name": "brand_name",
        "Likelihood Score": "likelihood_score",
        "Chapter Title": "chapter_title",
        "Last Update": "last_update",
        "Year Approved": "year_approved",
        "Type of Agent": "agent_classification",
        "In LiverTox": "include_in_livertox",
        "Primary Classification": "primary_classification",
        "Secondary Classification": "secondary_classification",
    }

    data = cast(
        pd.DataFrame,
        data.rename(columns=lambda s: re.sub(r"\s+", " ", s).strip()),
    )
    data = cast(pd.DataFrame, data.rename(columns=column_mapping))

    required_columns = list(column_mapping.values())
    for column in required_columns:
        if column not in data.columns:
            data[column] = pd.NA

    data = cast(pd.DataFrame, data[required_columns])

    text_columns = [
        "ingredient",
        "brand_name",
        "likelihood_score",
        "chapter_title",
        "agent_classification",
        "primary_classification",
        "secondary_classification",
    ]
    for column in text_columns:
        data[column] = clean_master_list_column(self, cast(pd.Series, data[column]))

    data = cast(pd.DataFrame, data.dropna(subset=["chapter_title"]))

    invalid_headers = {
        "ingredient": {"ingredient", "count"},
        "brand_name": {"brand name"},
    }
    for column, values in invalid_headers.items():
        column_values = cast(pd.Series, data[column]).fillna("").str.lower()
        data = cast(pd.DataFrame, data[~column_values.isin(values)])

    data["last_update"] = pd.to_datetime(data["last_update"], errors="coerce")
    data["reference_count"] = pd.to_numeric(
        data["reference_count"], errors="coerce"
    )
    data["year_approved"] = pd.to_numeric(data["year_approved"], errors="coerce")

    data = cast(
        pd.DataFrame,
        data.drop_duplicates(subset=["ingredient", "brand_name"], keep="last"),
    )

    return data.reset_index(drop=True)

def clean_master_list_column(self, series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str).str.strip()
    cleaned = cleaned.replace("", pd.NA)
    return cleaned

def collect_monographs(
    self,
    archive_path: str | None = None,
    *,
    should_stop: Callable[[], bool] | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> list[dict[str, str]]:
    tar_path = archive_path or self.tar_file_path
    normalized_path = os.path.abspath(tar_path)
    if not os.path.isfile(normalized_path):
        raise FileNotFoundError(f"LiverTox archive missing at {normalized_path}")
    if not tarfile.is_tarfile(normalized_path):
        raise RuntimeError(f"Invalid LiverTox archive at {normalized_path}")

    collected: list[dict[str, str]] = []
    max_workers = self.monograph_max_workers

    # Stage 1: scan only archive metadata and keep one member per basename.
    selected_members: list[tarfile.TarInfo] = []
    selected_basenames: set[str] = set()
    with tarfile.open(normalized_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            normalized_name = os.path.normpath(member.name)
            if os.path.isabs(normalized_name) or normalized_name.startswith(".."):
                logger.warning("Skipping unsafe archive member: %s", member.name)
                continue
            extension = os.path.splitext(normalized_name.lower())[1]
            if extension not in self.supported_extensions:
                continue
            base_name = os.path.basename(member.name).lower()
            if base_name in selected_basenames:
                continue
            selected_basenames.add(base_name)
            selected_members.append(member)

    if not selected_members:
        return collected

    total_payloads = len(selected_members)
    processed_count = 0
    worker_budget = min(max_workers, total_payloads) or 1

    if worker_budget == 1:
        with tarfile.open(normalized_path, "r:gz") as archive:
            for member in tqdm(
                selected_members,
                desc="Processing LiverTox files",
                total=total_payloads,
            ):
                if livertox_common.should_cancel(should_stop):
                    raise RuntimeError("LiverTox update cancelled by user request")
                extracted = archive.extractfile(member)
                if extracted is None:
                    processed_count += 1
                    emit_monograph_progress(
                        self,
                        progress_callback=progress_callback,
                        processed_count=processed_count,
                        total_payloads=total_payloads,
                    )
                    continue
                try:
                    data = extracted.read()
                finally:
                    extracted.close()
                if data:
                    record = process_monograph_member(
                        member.name, data
                    )
                    if record:
                        collected.append(record)
                processed_count += 1
                emit_monograph_progress(
                    self,
                    progress_callback=progress_callback,
                    processed_count=processed_count,
                    total_payloads=total_payloads,
                )
        return sort_monograph_records(self, collected)

    ctx = multiprocessing.get_context("spawn")
    max_in_flight = max(1, worker_budget * 2)
    with ProcessPoolExecutor(max_workers=worker_budget, mp_context=ctx) as executor:
        in_flight: dict[Future[dict[str, str] | None], str] = {}
        with tarfile.open(normalized_path, "r:gz") as archive:
            for member in tqdm(
                selected_members,
                desc="Reading LiverTox files",
                total=total_payloads,
            ):
                if livertox_common.should_cancel(should_stop):
                    raise RuntimeError("LiverTox update cancelled by user request")
                extracted = archive.extractfile(member)
                if extracted is None:
                    processed_count += 1
                    emit_monograph_progress(
                        self,
                        progress_callback=progress_callback,
                        processed_count=processed_count,
                        total_payloads=total_payloads,
                    )
                    continue
                try:
                    data = extracted.read()
                finally:
                    extracted.close()
                if not data:
                    processed_count += 1
                    emit_monograph_progress(
                        self,
                        progress_callback=progress_callback,
                        processed_count=processed_count,
                        total_payloads=total_payloads,
                    )
                    continue
                future = executor.submit(
                    process_monograph_payload, member.name, data
                )
                in_flight[future] = member.name
                if len(in_flight) >= max_in_flight:
                    processed_count = drain_monograph_futures(
                        self,
                        in_flight=in_flight,
                        collected=collected,
                        processed_count=processed_count,
                        total_payloads=total_payloads,
                        progress_callback=progress_callback,
                        wait_for_one=True,
                    )

        while in_flight:
            if livertox_common.should_cancel(should_stop):
                raise RuntimeError("LiverTox update cancelled by user request")
            processed_count = drain_monograph_futures(
                self,
                in_flight=in_flight,
                collected=collected,
                processed_count=processed_count,
                total_payloads=total_payloads,
                progress_callback=progress_callback,
                wait_for_one=False,
            )
    return sort_monograph_records(self, collected)

def emit_monograph_progress(
    self,
    *,
    progress_callback: Callable[[float, str], None] | None,
    processed_count: int,
    total_payloads: int,
) -> None:
    ratio = min(1.0, max(0.0, processed_count / max(total_payloads, 1)))
    livertox_common.emit_progress(
        progress_callback,
        progress=35.0 + (ratio * 33.0),
        message=f"Processed {processed_count}/{total_payloads} LiverTox files",
    )

def drain_monograph_futures(
    self,
    *,
    in_flight: dict[Future[dict[str, str] | None], str],
    collected: list[dict[str, str]],
    processed_count: int,
    total_payloads: int,
    progress_callback: Callable[[float, str], None] | None,
    wait_for_one: bool,
) -> int:
    if not in_flight:
        return processed_count
    return_when = FIRST_COMPLETED if wait_for_one else ALL_COMPLETED
    done, _ = wait(set(in_flight), return_when=return_when)
    for future in done:
        member_name = in_flight.pop(future, "")
        base_name = os.path.basename(member_name).lower()
        try:
            record = future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to process LiverTox monograph '%s': %s",
                base_name or "<unknown>",
                exc,
            )
            processed_count += 1
            emit_monograph_progress(
                self,
                progress_callback=progress_callback,
                processed_count=processed_count,
                total_payloads=total_payloads,
            )
            continue
        if record:
            collected.append(record)
        processed_count += 1
        emit_monograph_progress(
            self,
            progress_callback=progress_callback,
            processed_count=processed_count,
            total_payloads=total_payloads,
        )
    return processed_count

def sort_monograph_records(
    self, records: list[dict[str, str]]
) -> list[dict[str, str]]:
    return sorted(
        records,
        key=lambda item: (
            str(item.get("drug_name", "")).casefold(),
            str(item.get("nbk_id", "")).casefold(),
            str(item.get("excerpt", "")).casefold(),
        ),
    )

def process_monograph_member(
    member_name: str,
    data: bytes,
) -> dict[str, str] | None:
    payload = convert_member_bytes(member_name, data)
    if payload is None:
        return None
    plain_text, markup_text = payload
    if not plain_text:
        return None
    nbk_id = extract_nbk(member_name, markup_text or plain_text)
    record_nbk = nbk_id or derive_identifier(member_name)
    if not record_nbk:
        return None
    drug_name = extract_title(
        markup_text or "",
        plain_text,
        record_nbk,
    )
    cleaned_text = plain_text.strip()
    if not drug_name or not cleaned_text:
        return None
    return {
        "nbk_id": record_nbk,
        "drug_name": drug_name,
        "excerpt": cleaned_text,
    }

def convert_member_bytes(
    member_name: str, data: bytes
) -> tuple[str, str | None] | None:
    lower_name = member_name.lower()
    if lower_name.endswith(".pdf"):
        text = pdf_to_text(data)
        if text.strip():
            return text, None
        decoded = decode_markup(data)
        return decoded, decoded
    markup = decode_markup(data)
    text = html_to_text(markup)
    return text, markup

def decode_markup(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")

def pdf_to_text(data: bytes) -> str:
    buffer = io.BytesIO(data)
    if pdfminer_extract_text is not None:
        try:
            buffer.seek(0)
            text = pdfminer_extract_text(buffer)
            if text:
                return text
        except Exception:
            buffer.seek(0)
    if PdfReader is not None:
        try:
            buffer.seek(0)
            reader = PdfReader(buffer)
            collected: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    collected.append(page_text)
            if collected:
                return "\n".join(collected)
        except Exception:
            buffer.seek(0)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="ignore")

def extract_nbk(member_name: str, content: str) -> str | None:
    match = re.search(r"NBK\d+", member_name, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    match = re.search(r"NBK\d+", content, re.IGNORECASE)
    if match:
        return match.group(0).upper()
    return None

def derive_identifier(member_name: str) -> str:
    base = os.path.basename(member_name)
    stem = os.path.splitext(base)[0]
    cleaned = normalize_whitespace(strip_punctuation(stem))
    return cleaned or base

def extract_title(html_text: str, plain_text: str, default: str) -> str:
    patterns = (
        r"<title-group[^>]*>\s*<title[^>]*>(.*?)</title>",
        r"<article-title[^>]*>(.*?)</article-title>",
        r"<h1[^>]*>(.*?)</h1>",
        r"<title[^>]*>(.*?)</title>",
    )
    for pattern in patterns:
        for match in re.finditer(
            pattern, html_text, flags=re.IGNORECASE | re.DOTALL
        ):
            fragment = clean_fragment(match.group(1))
            normalized = normalize_extracted_title(fragment)
            if normalized:
                return normalized
    for line in plain_text.splitlines():
        stripped = line.strip()
        if stripped:
            normalized = normalize_extracted_title(stripped)
            if normalized:
                return normalized
    return normalize_extracted_title(default) or default

def clean_fragment(fragment: str) -> str:
    return html_to_text(fragment)

def normalize_extracted_title(value: str) -> str:
    cleaned = normalize_whitespace(value)
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"\s*[-|:]\s*(?:LiverTox|NCBI Bookshelf)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return normalize_whitespace(cleaned)

def html_to_text(html_text: str) -> str:
    # Tempered dot avoids runaway backtracking on malformed HTML.
    stripped = re.sub(
        r"(?is)<(script|style)[^>]*>(?:(?!</\1>).)*</\1>", " ", html_text
    )
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    unescaped = html.unescape(stripped)
    return normalize_whitespace(unescaped)

def strip_punctuation(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    folded = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[-_,.;:()\[\]{}\/\\]", " ", folded)

def sanitize_records(self, entries: list[dict[str, Any]]) -> pd.DataFrame:
    sanitized = self.serializer.sanitize_livertox_records(entries)
    if sanitized.empty:
        sanitized = pd.DataFrame(
            columns=["nbk_id", "drug_name", "excerpt", "synonyms"]
        )
    sanitized = sanitized.copy()
    drug_names = cast(pd.Series, sanitized["drug_name"]).astype(str).str.strip()
    sanitized["drug_name"] = drug_names
    sanitized = cast(pd.DataFrame, sanitized[drug_names != ""])
    numeric_mask = cast(pd.Series, sanitized["drug_name"]).str.fullmatch(r"\d+")
    sanitized = cast(pd.DataFrame, sanitized[~numeric_mask])
    sanitized["excerpt"] = cast(pd.Series, sanitized["excerpt"]).apply(
        lambda value: sanitize_excerpt(self, value)
    )
    sanitized.loc[sanitized["excerpt"] == "", "excerpt"] = pd.NA
    if "synonyms" not in sanitized.columns:
        sanitized["synonyms"] = pd.NA
    synonyms = cast(pd.Series, sanitized["synonyms"])
    sanitized["synonyms"] = synonyms.where(
        pd.notnull(synonyms), pd.NA
    )
    return sanitized.reset_index(drop=True)

def sanitize_excerpt(self, value: Any) -> str | Any:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return pd.NA
    cleaned = self.excerpt_sanitizer.sanitize(text)
    if not cleaned:
        return pd.NA
    return cleaned

def normalize_nbk_id(self, value: Any) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    normalized = str(value).strip().upper()
    if not normalized:
        return None
    if not livertox_common.NBK_ID_PATTERN.fullmatch(normalized):
        return None
    return normalized

def contains_symbol(self, value: str) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.search(r"[^A-Za-z0-9\s\-/(),'.+]", value))
