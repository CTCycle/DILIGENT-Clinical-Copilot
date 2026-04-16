from __future__ import annotations

import io
from html import unescape
from html.parser import HTMLParser
import json
import re
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
class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[tuple[str, str]]]] = []
        self._in_table = False
        self._current_table: list[list[tuple[str, str]]] = []
        self._current_row: list[tuple[str, str]] | None = None
        self._current_cell_tag: str | None = None
        self._current_cell_parts: list[str] = []

    # ---------------------------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        _ = attrs
        tag = tag.casefold()
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif self._in_table and tag == "tr":
            self._current_row = []
        elif self._in_table and tag in {"th", "td"}:
            self._current_cell_tag = tag
            self._current_cell_parts = []

    # ---------------------------------------------------------------------
    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if not self._in_table:
            return
        if tag in {"th", "td"} and self._current_cell_tag == tag:
            if self._current_row is not None:
                self._current_row.append((tag, unescape("".join(self._current_cell_parts).strip())))
            self._current_cell_tag = None
            self._current_cell_parts = []
        elif tag == "tr":
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            if self._current_table:
                self.tables.append(self._current_table)
            self._in_table = False
            self._current_table = []
            self._current_row = None
            self._current_cell_tag = None
            self._current_cell_parts = []

    # ---------------------------------------------------------------------
    def handle_data(self, data: str) -> None:
        if self._current_cell_tag is not None:
            self._current_cell_parts.append(data)


###############################################################################
class DiliPriorUpdater:
    HEADER_TOKENS = {
        "ltkbid",
        "compoundname",
        "severityclass",
        "labelsection",
        "vdiliconcern",
        "comment",
        "dilistid",
        "dilistclassification",
        "routesofadministration",
        "compoundname",
        "compound",
        "routeofadministration",
    }

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
            text = response.text
        if "html" in content_type or "<table" in text.casefold():
            tables = self.parse_html_tables(text)
            if tables:
                return max(tables, key=lambda frame: (int(frame.shape[0]), int(frame.shape[1])))
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
    def parse_html_tables(self, text: str) -> list[pd.DataFrame]:
        parser = _HTMLTableParser()
        parser.feed(text)
        frames: list[pd.DataFrame] = []
        for table in parser.tables:
            if not table:
                continue
            header_row = table[0]
            has_header = bool(header_row) and all(cell_tag == "th" for cell_tag, _ in header_row)
            if not has_header:
                has_header = self.looks_like_header_row(header_row)
            if has_header:
                columns = [cell_text or f"column_{index + 1}" for index, (_, cell_text) in enumerate(header_row)]
                rows = [
                    [cell_text for _, cell_text in row]
                    for row in table[1:]
                    if any(cell_text for _, cell_text in row)
                ]
                frame = pd.DataFrame(rows, columns=columns)
            else:
                max_columns = max(len(row) for row in table)
                rows = [
                    [cell_text for _, cell_text in row] + [None] * (max_columns - len(row))
                    for row in table
                    if any(cell_text for _, cell_text in row)
                ]
                frame = pd.DataFrame(rows, columns=[f"column_{index + 1}" for index in range(max_columns)])
            if not frame.empty:
                frames.append(frame)
        return frames

    # -------------------------------------------------------------------------
    def looks_like_header_row(self, row: list[tuple[str, str]]) -> bool:
        normalized = [self.normalize_header_token(text) for _, text in row if text]
        if len(normalized) < 2:
            return False
        header_hits = sum(1 for token in normalized if token in self.HEADER_TOKENS)
        return header_hits >= max(2, len(normalized) - 1)

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_header_token(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    # -------------------------------------------------------------------------
    def parse_dilirank(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return self.empty_annotation_frame()
        working = frame.copy().where(pd.notnull(frame), None)
        lowered = {str(column).strip().lower(): column for column in working.columns}
        name_column = self.pick_column(
            lowered,
            ["drug", "drug_name", "name", "compoundname", "compound name", "compound"],
        )
        record_column = self.pick_column(
            lowered,
            ["id", "dilirank_id", "record_id", "ltkbid", "ltkb_id"],
        )
        classification_column = self.pick_column(
            lowered,
            [
                "dilirank",
                "classification",
                "class",
                "vdili-concern",
                "vdili concern",
                "vdili_concern",
            ],
        )
        severity_column = self.pick_column(lowered, ["severity", "severity_class", "severityclass"])
        concern_column = self.pick_column(
            lowered,
            ["concern", "concern_class", "dililst", "dilist", "dilist classification"],
        )
        label_section_column = self.pick_column(lowered, ["label_section", "labelsection", "label section"])
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
                    "label_section": self.clean_text(row.get(label_section_column)),
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
        name_column = self.pick_column(
            lowered,
            ["drug", "drug_name", "name", "compoundname", "compound name", "compound"],
        )
        record_column = self.pick_column(
            lowered,
            ["id", "dilist_id", "record_id", "ltkbid", "ltkb_id"],
        )
        classification_column = self.pick_column(
            lowered,
            [
                "classification",
                "class",
                "category",
                "dilist",
                "dilist classification",
                "dilist_classification",
            ],
        )
        concern_column = self.pick_column(lowered, ["concern", "concern_class"])
        route_column = self.pick_column(
            lowered,
            ["route", "routes", "route of administration", "routes of administration"],
        )
        label_section_column = self.pick_column(lowered, ["label_section", "section", "label section"])
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
        db_session = self.serializer.session_factory()
        linked_rows = 0
        ambiguous_rows = 0
        unmatched_rows = 0
        rows: list[dict[str, Any]] = []
        try:
            for row in frame.to_dict(orient="records"):
                source_name = self.clean_text(row.get("source_name"))
                matched_id = self.serializer.resolve_drug_id(
                    db_session,
                    matched_drug_name=source_name,
                    rxcui=None,
                    nbk_id=None,
                )
                if matched_id is not None:
                    linked_rows += 1
                else:
                    unmatched_rows += 1
                rows.append(
                    {
                        **row,
                        "source_dataset": source_dataset,
                        "drug_id": matched_id,
                    }
                )
        finally:
            db_session.close()
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

