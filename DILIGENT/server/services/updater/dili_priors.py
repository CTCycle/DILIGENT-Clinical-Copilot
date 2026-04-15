from __future__ import annotations

import io
from typing import Any

import httpx
import pandas as pd

from DILIGENT.server.common.constants import (
    DILIRANK_SOURCE_URL,
    DILIST_SOURCE_URL,
)
from DILIGENT.server.configurations.startup import server_settings
from DILIGENT.server.repositories.serialization.data import DataSerializer
from DILIGENT.server.services.text.normalization import normalize_drug_name


###############################################################################
class DiliPriorUpdater:
    def __init__(
        self,
        *,
        serializer: DataSerializer | None = None,
        request_timeout: float | None = None,
    ) -> None:
        self.serializer = serializer or DataSerializer()
        configured = (
            float(request_timeout)
            if request_timeout is not None
            else float(server_settings.external_data.dili_priors_request_timeout)
        )
        self.request_timeout = max(configured, 1.0)

    # -------------------------------------------------------------------------
    def update_from_sources(self, *, redownload: bool = False) -> dict[str, dict[str, int]]:
        _ = redownload
        dilirank_frame = self.parse_dilirank(self.download_dilirank())
        dilist_frame = self.parse_dilist(self.download_dilist())
        dilirank_matched, dilirank_summary = self.match_rows_to_drugs(
            dilirank_frame,
            source_dataset="dilirank",
        )
        dilist_matched, dilist_summary = self.match_rows_to_drugs(
            dilist_frame,
            source_dataset="dilist",
        )
        unified = pd.concat([dilirank_matched, dilist_matched], ignore_index=True)
        self.persist_annotations(unified)
        return {
            "dilirank": dilirank_summary,
            "dilist": dilist_summary,
        }

    # -------------------------------------------------------------------------
    def download_dilirank(self) -> pd.DataFrame:
        return self.download_tabular_source(DILIRANK_SOURCE_URL)

    # -------------------------------------------------------------------------
    def download_dilist(self) -> pd.DataFrame:
        return self.download_tabular_source(DILIST_SOURCE_URL)

    # -------------------------------------------------------------------------
    def download_tabular_source(self, source_url: str) -> pd.DataFrame:
        with httpx.Client(timeout=self.request_timeout, follow_redirects=True) as client:
            response = client.get(source_url)
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            content = response.content
        if "csv" in content_type:
            return pd.read_csv(io.BytesIO(content))
        if "excel" in content_type or "spreadsheet" in content_type:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception:
            pass
        try:
            return pd.read_excel(io.BytesIO(content), engine="openpyxl")
        except Exception:
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    def parse_dilirank(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return self.empty_annotation_frame()
        working = frame.copy().where(pd.notnull(frame), None)
        lowered = {str(column).strip().lower(): column for column in working.columns}
        name_column = self.pick_column(lowered, ["drug", "drug_name", "name"])
        record_column = self.pick_column(lowered, ["id", "dilirank_id", "record_id"])
        classification_column = self.pick_column(
            lowered,
            ["dilirank", "classification", "class"],
        )
        severity_column = self.pick_column(lowered, ["severity", "severity_class"])
        concern_column = self.pick_column(lowered, ["concern", "concern_class"])
        comment_column = self.pick_column(lowered, ["comment", "notes"])
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(working.to_dict(orient="records"), start=1):
            source_name = self.clean_text(row.get(name_column))
            if source_name is None:
                continue
            source_record_id = self.clean_text(row.get(record_column)) or f"dilirank::{index}"
            rows.append(
                {
                    "drug_id": None,
                    "source_dataset": "dilirank",
                    "source_record_id": source_record_id,
                    "source_name": source_name,
                    "source_name_norm": normalize_drug_name(source_name),
                    "classification": self.clean_text(row.get(classification_column)),
                    "severity_class": self.clean_text(row.get(severity_column)),
                    "concern_class": self.clean_text(row.get(concern_column)),
                    "label_section": None,
                    "routes": None,
                    "comment": self.clean_text(row.get(comment_column)),
                    "source_url": DILIRANK_SOURCE_URL,
                    "source_last_modified": None,
                }
            )
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    def parse_dilist(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return self.empty_annotation_frame()
        working = frame.copy().where(pd.notnull(frame), None)
        lowered = {str(column).strip().lower(): column for column in working.columns}
        name_column = self.pick_column(lowered, ["drug", "drug_name", "name"])
        record_column = self.pick_column(lowered, ["id", "dilist_id", "record_id"])
        classification_column = self.pick_column(lowered, ["classification", "class", "category"])
        concern_column = self.pick_column(lowered, ["concern", "concern_class"])
        route_column = self.pick_column(lowered, ["route", "routes"])
        label_section_column = self.pick_column(lowered, ["label_section", "section"])
        comment_column = self.pick_column(lowered, ["comment", "notes"])
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(working.to_dict(orient="records"), start=1):
            source_name = self.clean_text(row.get(name_column))
            if source_name is None:
                continue
            source_record_id = self.clean_text(row.get(record_column)) or f"dilist::{index}"
            rows.append(
                {
                    "drug_id": None,
                    "source_dataset": "dilist",
                    "source_record_id": source_record_id,
                    "source_name": source_name,
                    "source_name_norm": normalize_drug_name(source_name),
                    "classification": self.clean_text(row.get(classification_column)),
                    "severity_class": None,
                    "concern_class": self.clean_text(row.get(concern_column)),
                    "label_section": self.clean_text(row.get(label_section_column)),
                    "routes": self.clean_text(row.get(route_column)),
                    "comment": self.clean_text(row.get(comment_column)),
                    "source_url": DILIST_SOURCE_URL,
                    "source_last_modified": None,
                }
            )
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    def match_rows_to_drugs(
        self,
        frame: pd.DataFrame,
        *,
        source_dataset: str,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        if frame.empty:
            return frame.copy(), {
                "downloaded_rows": 0,
                "linked_rows": 0,
                "unmatched_rows": 0,
                "ambiguous_rows": 0,
            }
        by_canonical: dict[str, set[int]] = {}
        by_alias: dict[str, set[int]] = {}
        stream = self.serializer.stream_drugs_catalog()
        for row in stream.to_dict(orient="records"):
            drug_id = self.serializer.to_int(row.get("drug_id"))
            if drug_id is None:
                continue
            canonical = self.clean_text(row.get("canonical_name_norm"))
            if canonical:
                by_canonical.setdefault(canonical, set()).add(drug_id)
            for alias in row.get("aliases", []) or []:
                if not isinstance(alias, dict):
                    continue
                alias_norm = self.clean_text(alias.get("alias_norm"))
                if alias_norm:
                    by_alias.setdefault(alias_norm, set()).add(drug_id)
        linked_rows = 0
        ambiguous_rows = 0
        unmatched_rows = 0
        rows: list[dict[str, Any]] = []
        for row in frame.to_dict(orient="records"):
            source_name_norm = self.clean_text(row.get("source_name_norm"))
            matched_ids = set(by_canonical.get(source_name_norm or "", set()))
            if not matched_ids:
                matched_ids = set(by_alias.get(source_name_norm or "", set()))
            matched_id: int | None = None
            if len(matched_ids) == 1:
                matched_id = next(iter(matched_ids))
                linked_rows += 1
            elif len(matched_ids) > 1:
                ambiguous_rows += 1
            else:
                unmatched_rows += 1
            rows.append(
                {
                    **row,
                    "source_dataset": source_dataset,
                    "drug_id": matched_id,
                }
            )
        return pd.DataFrame(rows), {
            "downloaded_rows": len(rows),
            "linked_rows": linked_rows,
            "unmatched_rows": unmatched_rows,
            "ambiguous_rows": ambiguous_rows,
        }

    # -------------------------------------------------------------------------
    def persist_annotations(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        self.serializer.save_dili_annotations(frame)

    # -------------------------------------------------------------------------
    @staticmethod
    def clean_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    # -------------------------------------------------------------------------
    @staticmethod
    def pick_column(mapping: dict[str, Any], candidates: list[str]) -> Any:
        for candidate in candidates:
            if candidate in mapping:
                return mapping[candidate]
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def empty_annotation_frame() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "drug_id",
                "source_dataset",
                "source_record_id",
                "source_name",
                "source_name_norm",
                "classification",
                "severity_class",
                "concern_class",
                "label_section",
                "routes",
                "comment",
                "source_url",
                "source_last_modified",
            ]
        )

