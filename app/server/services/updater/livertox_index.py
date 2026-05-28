from __future__ import annotations

from typing import Any, cast

import pandas as pd

from common.utils.logger import logger
from services.updater import livertox_parse


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
        return sanitize_unified_dataset(self, dataset)

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
            dataset = cast(
                pd.DataFrame, pd.concat([dataset, filler], ignore_index=True)
            )
    dataset = cast(pd.DataFrame, dataset[final_columns])
    return sanitize_unified_dataset(self, dataset)


def sanitize_unified_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sanitized = frame.copy()
    sanitized_drug_names = (
        cast(pd.Series, sanitized["drug_name"]).astype(str).str.strip()
    )
    sanitized["drug_name"] = sanitized_drug_names
    sanitized = cast(pd.DataFrame, sanitized[sanitized_drug_names != ""])
    numeric_mask = cast(pd.Series, sanitized["drug_name"]).str.fullmatch(r"\d+")
    sanitized = cast(pd.DataFrame, sanitized[~numeric_mask])
    symbol_mask = cast(pd.Series, sanitized["drug_name"]).apply(
        lambda value: livertox_parse.contains_symbol(self, value)
    )
    sanitized = cast(pd.DataFrame, sanitized[~symbol_mask])

    for column in ("ingredient", "brand_name"):
        if column not in sanitized.columns:
            sanitized[column] = pd.NA
        column_values = cast(pd.Series, sanitized[column])
        sanitized[column] = column_values.where(pd.notnull(column_values), pd.NA)
        column_values = cast(pd.Series, sanitized[column]).astype(str).str.strip()
        sanitized[column] = column_values
        sanitized.loc[column_values.isin(["", "nan", "None", "<NA>"]), column] = pd.NA
        column_values = cast(pd.Series, sanitized[column])
        invalid_mask = column_values.notna() & column_values.apply(
            lambda value: livertox_parse.contains_symbol(self, value)
        )
        invalid_mask = invalid_mask.fillna(False)
        sanitized = cast(pd.DataFrame, sanitized[~invalid_mask])

    excerpt_values = cast(pd.Series, sanitized["excerpt"]).astype(str).str.strip()
    sanitized["excerpt"] = excerpt_values
    sanitized.loc[excerpt_values.isin(["", "nan", "None", "NaT"]), "excerpt"] = pd.NA
    sanitized.loc[cast(pd.Series, sanitized["excerpt"]).isna(), "excerpt"] = pd.NA
    return sanitized.reset_index(drop=True)


def finalize_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    finalized = sanitize_unified_dataset(self, frame)
    for column in finalized.columns:
        if column == "drug_name":
            continue
        column_values = cast(pd.Series, finalized[column])
        finalized[column] = column_values.where(pd.notnull(column_values), pd.NA)
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
    nbk_series = cast(pd.Series, finalized["nbk_id"]).apply(
        lambda value: livertox_parse.normalize_nbk_id(self, value)
    )
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
        _source_url_sort=finalized["source_url"].fillna("").astype(str).str.casefold(),
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
