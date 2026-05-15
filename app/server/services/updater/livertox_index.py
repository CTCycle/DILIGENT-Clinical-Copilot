from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, cast
from collections.abc import Callable

import httpx
import pandas as pd
from tqdm import tqdm

from configurations.startup import server_settings
from common.utils.logger import logger
from services.updater import livertox_parse

SUPPORTED_MONOGRAPH_EXTENSIONS = (".html", ".htm", ".xhtml", ".xml", ".nxml", ".pdf")
NBK_ID_PATTERN = re.compile(r"^NBK\d+$", re.IGNORECASE)

DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "DILIGENTClinicalCopilot/1.0 (contact=clinical-copilot@pharmagent.local)"
    )
}

DOWNLOAD_CHUNK_SIZE = 262_144


# -----------------------------------------------------------------------------
def load_json(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


# -----------------------------------------------------------------------------
def save_masterlist_metadata(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


# -----------------------------------------------------------------------------
def metadata_matches(stored: dict[str, Any], remote: dict[str, Any]) -> bool:
    return stored.get("last_modified") == remote.get("last_modified") and int(
        stored.get("size", 0)
    ) == int(remote.get("size", 0))


# -----------------------------------------------------------------------------
def process_monograph_payload(
    member_name: str,
    data: bytes,
) -> dict[str, str] | None:
    return livertox_parse.process_monograph_member(member_name, data)


###############################################################################
async def download_file(
    client: httpx.AsyncClient,
    url: str,
    destination: str,
    total_size: int,
    label: str,
    *,
    chunk_size: int,
) -> None:
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        with (
            open(destination, "wb") as output,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=label,
                ncols=80,
            ) as progress,
        ):
            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))


###############################################################################

# Extracted from the facade module; functions intentionally accept the facade instance.

def update_from_livertox(
    self,
    *,
    progress_callback: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    logger.info("Starting LiverTox update")
    if self.should_cancel(should_stop):
        raise RuntimeError("LiverTox update cancelled by user request")
    self.emit_progress(
        progress_callback,
        progress=5.0,
        message="Refreshing LiverTox master list",
    )
    master_metadata, master_frame = self.refresh_master_list()

    logger.info("Checking LiverTox archive metadata")
    if self.should_cancel(should_stop):
        raise RuntimeError("LiverTox update cancelled by user request")
    self.emit_progress(
        progress_callback,
        progress=20.0,
        message="Downloading LiverTox archive metadata",
    )
    archive_metadata = asyncio.run(self.download_bulk_data(self.sources_path))
    archive_path = archive_metadata.get("file_path") or os.path.join(
        self.sources_path, server_settings.external_data.livertox_archive
    )

    local_info = self.collect_local_archive_info(archive_path)
    logger.info("Extracting LiverTox monographs from %s", archive_path)
    self.emit_progress(
        progress_callback,
        progress=35.0,
        message="Extracting LiverTox monographs",
    )
    extracted = self.collect_monographs(
        archive_path,
        should_stop=should_stop,
        progress_callback=progress_callback,
    )
    if self.should_cancel(should_stop):
        raise RuntimeError("LiverTox update cancelled by user request")
    logger.info("Sanitizing %d extracted entries", len(extracted))
    self.emit_progress(
        progress_callback,
        progress=70.0,
        message="Sanitizing extracted LiverTox entries",
    )
    monograph_df = self.sanitize_records(extracted)
    logger.info("Combining LiverTox datasets")
    self.emit_progress(
        progress_callback,
        progress=80.0,
        message="Combining master list and monograph excerpts",
    )
    unified = self.build_unified_dataset(
        monograph_df,
        master_frame,
        master_metadata,
    )
    logger.info("Finalizing sanitized dataset")
    self.emit_progress(
        progress_callback,
        progress=88.0,
        message="Finalizing LiverTox dataset",
    )
    final_dataset = self.finalize_dataset(unified)
    logger.info("Persisting finalized records to database")
    if self.should_cancel(should_stop):
        raise RuntimeError("LiverTox update cancelled by user request")
    self.emit_progress(
        progress_callback,
        progress=95.0,
        message="Persisting LiverTox records",
    )
    self.serializer.save_livertox_records(final_dataset)

    payload = {**master_metadata, **archive_metadata, **local_info}
    payload["processed_entries"] = len(final_dataset.index)
    payload["records"] = len(final_dataset.index)
    logger.info("LiverTox update completed successfully")
    self.emit_progress(
        progress_callback,
        progress=99.0,
        message="LiverTox update completed",
    )

    return payload

def build_unified_dataset(
    self,
    monographs: pd.DataFrame,
    master_frame: pd.DataFrame,
    master_metadata: dict[str, Any],
) -> pd.DataFrame:
    base_columns = [
        "drug_name",
        "ingredient",
        "brand_name",
        "likelihood_score",
        "last_update",
        "reference_count",
        "year_approved",
        "agent_classification",
        "primary_classification",
        "secondary_classification",
        "include_in_livertox",
        "source_url",
        "source_last_modified",
    ]
    monograph_columns = ["drug_name", "nbk_id", "excerpt", "synonyms"]
    final_columns = base_columns + ["nbk_id", "excerpt", "synonyms"]

    if master_frame is None or master_frame.empty:
        master = pd.DataFrame(columns=base_columns)
    else:
        master = master_frame.copy()
        if "chapter_title" in master.columns:
            master = cast(
                pd.DataFrame,
                master.rename(columns={"chapter_title": "drug_name"}),
            )
        if "drug_name" not in master.columns:
            master["drug_name"] = pd.NA
        master_drug_names = cast(pd.Series, master["drug_name"]).astype(str).str.strip()
        master["drug_name"] = master_drug_names
        master = cast(pd.DataFrame, master[master_drug_names != ""])
        for column in base_columns:
            if column not in master.columns:
                master[column] = pd.NA
        if master.empty and master_metadata.get("source_url"):
            master = pd.DataFrame(columns=base_columns)
        else:
            metadata_source_url = master_metadata.get("source_url")
            metadata_last_modified = master_metadata.get("last_modified")
            if metadata_source_url is not None:
                master["source_url"] = cast(pd.Series, master["source_url"]).fillna(
                    metadata_source_url
                )
            if metadata_last_modified is not None:
                master["source_last_modified"] = cast(
                    pd.Series, master["source_last_modified"]
                ).fillna(metadata_last_modified)
        master = cast(pd.DataFrame, master[base_columns])

    if monographs.empty:
        monograph_df = pd.DataFrame(columns=monograph_columns)
    else:
        monograph_df = monographs.copy()
    for column in monograph_columns:
        if column not in monograph_df.columns:
            monograph_df[column] = pd.NA
    monograph_df = cast(pd.DataFrame, monograph_df[monograph_columns])

    if master.empty:
        dataset = monograph_df.copy()
        for column in base_columns:
            if column not in dataset.columns:
                dataset[column] = pd.NA
        dataset = cast(pd.DataFrame, dataset[final_columns])
        return self.sanitize_unified_dataset(dataset)

    dataset = cast(pd.DataFrame, master.merge(monograph_df, on="drug_name", how="left"))
    if not monograph_df.empty:
        matched = cast(pd.Series, dataset["drug_name"]).unique().tolist()
        unmatched = cast(
            pd.DataFrame,
            monograph_df[~cast(pd.Series, monograph_df["drug_name"]).isin(matched)],
        )
        if not unmatched.empty:
            filler = unmatched.copy()
            for column in base_columns:
                if column not in filler.columns:
                    filler[column] = pd.NA
            filler = cast(pd.DataFrame, filler[dataset.columns])
            dataset = cast(pd.DataFrame, pd.concat([dataset, filler], ignore_index=True))
    dataset = cast(pd.DataFrame, dataset[final_columns])
    return self.sanitize_unified_dataset(dataset)

def sanitize_unified_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sanitized = frame.copy()
    sanitized_drug_names = cast(pd.Series, sanitized["drug_name"]).astype(str).str.strip()
    sanitized["drug_name"] = sanitized_drug_names
    sanitized = cast(pd.DataFrame, sanitized[sanitized_drug_names != ""])
    numeric_mask = cast(pd.Series, sanitized["drug_name"]).str.fullmatch(r"\d+")
    sanitized = cast(pd.DataFrame, sanitized[~numeric_mask])
    symbol_mask = cast(pd.Series, sanitized["drug_name"]).apply(self.contains_symbol)
    sanitized = cast(pd.DataFrame, sanitized[~symbol_mask])

    for column in ("ingredient", "brand_name"):
        if column not in sanitized.columns:
            sanitized[column] = pd.NA
        column_values = cast(pd.Series, sanitized[column])
        sanitized[column] = column_values.where(
            pd.notnull(column_values), pd.NA
        )
        column_values = cast(pd.Series, sanitized[column]).astype(str).str.strip()
        sanitized[column] = column_values
        sanitized.loc[
            column_values.isin(["", "nan", "None", "<NA>"]), column
        ] = pd.NA
        column_values = cast(pd.Series, sanitized[column])
        invalid_mask = column_values.notna() & column_values.apply(
            self.contains_symbol
        )
        invalid_mask = invalid_mask.fillna(False)
        sanitized = cast(pd.DataFrame, sanitized[~invalid_mask])

    excerpt_values = cast(pd.Series, sanitized["excerpt"]).astype(str).str.strip()
    sanitized["excerpt"] = excerpt_values
    sanitized.loc[
        excerpt_values.isin(["", "nan", "None", "NaT"]), "excerpt"
    ] = pd.NA
    sanitized.loc[cast(pd.Series, sanitized["excerpt"]).isna(), "excerpt"] = pd.NA
    return sanitized.reset_index(drop=True)

def finalize_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    finalized = self.sanitize_unified_dataset(frame)
    for column in finalized.columns:
        if column == "drug_name":
            continue
        column_values = cast(pd.Series, finalized[column])
        finalized[column] = column_values.where(
            pd.notnull(column_values), pd.NA
        )
        column_values = cast(pd.Series, finalized[column]).astype(str).str.strip()
        finalized[column] = column_values
        finalized.loc[
            column_values.isin(["", "nan", "NaT", "None", "<NA>"]), column
        ] = pd.NA
    finalized["synonyms"] = cast(pd.Series, finalized["synonyms"]).apply(
        lambda value: (
            value.strip()
            if isinstance(value, str)
            and value.strip()
            and value.strip().lower() not in {"<na>", "nan", "nat", "none"}
            else pd.NA
        )
    )
    excerpt_values = cast(pd.Series, finalized["excerpt"]).astype(str).str.strip()
    finalized["excerpt"] = excerpt_values
    finalized.loc[
        excerpt_values.isin(["", "nan", "NaT", "None", "<NA>"]), "excerpt"
    ] = pd.NA
    if "nbk_id" not in finalized.columns:
        finalized["nbk_id"] = pd.NA
    nbk_series = cast(pd.Series, finalized["nbk_id"]).apply(self.normalize_nbk_id)
    present_nbk_series = cast(pd.Series, nbk_series.dropna())
    counts = cast(pd.Series, present_nbk_series.value_counts())
    unique_counts = cast(pd.Series, counts[counts == 1])
    safe_nbk_values = set(unique_counts.index.tolist())
    safe_mask = nbk_series.isin(list(safe_nbk_values))
    nulled_nbk_count = int(cast(Any, (nbk_series.notna() & ~safe_mask).sum()))
    safe_nbk_count = int(cast(Any, safe_mask.sum()))
    finalized["nbk_id"] = nbk_series.where(safe_mask, pd.NA)
    logger.info(
        "LiverTox NBK audit: total_rows=%d safe_nbk_count=%d nulled_nbk_count=%d",
        len(finalized.index),
        safe_nbk_count,
        nulled_nbk_count,
    )

    for column in ("source_last_modified", "source_url", "last_update"):
        if column not in finalized.columns:
            finalized[column] = pd.NA
    sort_frame = finalized.assign(
        _drug_name_sort=finalized["drug_name"].astype(str).str.casefold(),
        _source_last_modified_sort=finalized["source_last_modified"]
        .fillna("")
        .astype(str)
        .str.casefold(),
        _source_url_sort=finalized["source_url"]
        .fillna("")
        .astype(str)
        .str.casefold(),
        _last_update_sort=finalized["last_update"]
        .fillna("")
        .astype(str)
        .str.casefold(),
    )
    sort_frame = sort_frame.sort_values(
        by=[
            "_drug_name_sort",
            "_source_last_modified_sort",
            "_source_url_sort",
            "_last_update_sort",
        ],
        kind="mergesort",
    )
    deduped = sort_frame.drop_duplicates(
        subset=["drug_name", "ingredient", "brand_name"],
        keep="first",
    )
    deduped = deduped.drop(
        columns=[
            "_drug_name_sort",
            "_source_last_modified_sort",
            "_source_url_sort",
            "_last_update_sort",
        ]
    )
    return deduped.reset_index(drop=True)
