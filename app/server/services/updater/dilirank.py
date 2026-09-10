from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from common.paths import ARCHIVES_PATH
from repositories import values as repository_values
from repositories.dilirank_repository import DiliRankRepository

DILIRANK_SOURCE_URL = "https://www.fda.gov/media/113052/download?attachment="
DILIRANK_FILE_NAME = "DILIrank2.0.xlsx"
DILIRANK_METADATA_FILE_NAME = "dilirank2.metadata.json"
DILIRANK_REQUIRED_COLUMNS = (
    "LTKBID",
    "CompoundName",
    "SeverityClass",
    "LabelSection",
    "vDILI-Concern",
    "Comment",
)
DILIRANK_CONCERN_CANONICAL_BY_CASEFOLD = {
    "vmost-dili-concern": "vMost-DILI-concern",
    "vless-dili-concern": "vLess-DILI-concern",
    "vno-dili-concern": "vNo-DILI-concern",
    "ambiguous-dili-concern": "Ambiguous-DILI-concern",
}

###############################################################################
class DiliRankUpdater:

    # -------------------------------------------------------------------------
    def __init__(
        self,
        *,
        repository: DiliRankRepository,
        archives_path: Path = ARCHIVES_PATH,
        redownload: bool = False,
        request_timeout: float = 60.0,
    ) -> None:
        self.repository = repository
        self.archives_path = Path(archives_path)
        self.redownload = bool(redownload)
        self.request_timeout = max(float(request_timeout), 1.0)
        self.workbook_path = self.archives_path / DILIRANK_FILE_NAME
        self.metadata_path = self.archives_path / DILIRANK_METADATA_FILE_NAME

    # -------------------------------------------------------------------------
    def update_from_fda(
        self,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        self._raise_if_cancelled(should_stop)
        self._emit(progress_callback, 5.0, "Refreshing FDA DILIrank 2.0 source")
        source_metadata = self._download_workbook()
        self._raise_if_cancelled(should_stop)
        self._emit(progress_callback, 40.0, "Validating DILIrank 2.0 workbook")
        try:
            frame = pd.read_excel(self.workbook_path, engine="openpyxl")
            records = self._parse_records(frame, source_metadata)
        except Exception:
            if source_metadata.get("downloaded"):
                self.workbook_path.unlink(missing_ok=True)
                self.metadata_path.unlink(missing_ok=True)
            raise
        self._raise_if_cancelled(should_stop)
        self._emit(progress_callback, 75.0, "Linking DILIrank records to local drugs")
        summary = self.repository.replace_records(records)
        self._emit(progress_callback, 98.0, "DILIrank 2.0 update completed")
        return {
            **summary,
            "downloaded": bool(source_metadata.get("downloaded")),
            "source_url": source_metadata.get("source_url"),
            "source_last_modified": source_metadata.get("last_modified"),
            "etag": source_metadata.get("etag"),
        }

    # -------------------------------------------------------------------------
    def _download_workbook(self) -> dict[str, Any]:
        self.archives_path.mkdir(parents=True, exist_ok=True)
        stored = self._load_metadata()
        headers = {
            "Accept": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*",
            "User-Agent": "DILIGENT/3 DILIrank updater",
        }
        if not self.redownload and self.workbook_path.is_file():
            etag = str(stored.get("etag") or "").strip()
            last_modified = str(stored.get("last_modified") or "").strip()
            if etag:
                headers["If-None-Match"] = etag
            if last_modified:
                headers["If-Modified-Since"] = last_modified

        with httpx.Client(
            timeout=self.request_timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            response = client.get(DILIRANK_SOURCE_URL)
        if response.status_code == 304 and self.workbook_path.is_file():
            return {
                **stored,
                "downloaded": False,
                "source_url": stored.get("source_url") or DILIRANK_SOURCE_URL,
            }
        response.raise_for_status()
        if not response.content:
            raise RuntimeError("FDA DILIrank 2.0 download returned an empty workbook.")
        if not response.content.startswith(b"PK"):
            raise RuntimeError("FDA DILIrank 2.0 download was not a valid XLSX payload.")
        temporary_path = self.workbook_path.with_suffix(".xlsx.tmp")
        temporary_path.write_bytes(response.content)
        temporary_path.replace(self.workbook_path)
        metadata = {
            "source_url": str(response.url),
            "last_modified": response.headers.get("Last-Modified"),
            "etag": response.headers.get("ETag"),
            "size": len(response.content),
        }
        self._save_metadata(metadata)
        return {**metadata, "downloaded": True}

    # -------------------------------------------------------------------------
    def _parse_records(
        self,
        frame: pd.DataFrame,
        source_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        missing_columns = [
            column for column in DILIRANK_REQUIRED_COLUMNS if column not in frame.columns
        ]
        if missing_columns:
            raise RuntimeError(
                "FDA DILIrank 2.0 workbook schema changed; missing column(s): "
                + ", ".join(missing_columns)
            )
        records: list[dict[str, Any]] = []
        unknown_concerns: set[str] = set()
        seen_ltkb_ids: set[str] = set()
        for row in frame.to_dict(orient="records"):
            ltkb_id = self._cell_text(row.get("LTKBID"))
            compound_name = self._cell_text(row.get("CompoundName"))
            raw_dili_concern = self._cell_text(row.get("vDILI-Concern"))
            if ltkb_id is None or compound_name is None or raw_dili_concern is None:
                continue
            dili_concern = DILIRANK_CONCERN_CANONICAL_BY_CASEFOLD.get(
                raw_dili_concern.casefold()
            )
            if ltkb_id in seen_ltkb_ids:
                raise RuntimeError(f"Duplicate FDA DILIrank identifier encountered: {ltkb_id}")
            seen_ltkb_ids.add(ltkb_id)
            if dili_concern is None:
                unknown_concerns.add(raw_dili_concern)
                continue
            records.append(
                {
                    "ltkb_id": ltkb_id,
                    "compound_name": compound_name,
                    "severity_class": self._cell_int(row.get("SeverityClass")),
                    "label_section": self._cell_text(row.get("LabelSection")),
                    "dili_concern": dili_concern,
                    "comment": self._cell_text(row.get("Comment")),
                    "source_url": source_metadata.get("source_url")
                    or DILIRANK_SOURCE_URL,
                    "source_last_modified": source_metadata.get("last_modified"),
                }
            )
        if unknown_concerns:
            raise RuntimeError(
                "FDA DILIrank 2.0 introduced unsupported concern category value(s): "
                + ", ".join(sorted(unknown_concerns))
            )
        if not records:
            raise RuntimeError("FDA DILIrank 2.0 workbook contained no usable records.")
        return records

    # -------------------------------------------------------------------------
    @staticmethod
    def _cell_int(value: Any) -> int | None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"FDA DILIrank 2.0 severity class is not numeric: {value!r}"
            ) from exc
        if not number.is_integer():
            raise RuntimeError(
                f"FDA DILIrank 2.0 severity class is not an integer: {value!r}"
            )
        return int(number)

    # -------------------------------------------------------------------------
    @staticmethod
    def _cell_text(value: Any) -> str | None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return repository_values.normalize_string(value)

    # -------------------------------------------------------------------------
    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            return {}
        try:
            payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    # -------------------------------------------------------------------------
    def _save_metadata(self, payload: dict[str, Any]) -> None:
        temporary_path = self.metadata_path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(self.metadata_path)

    # -------------------------------------------------------------------------
    @staticmethod
    def _raise_if_cancelled(should_stop: Callable[[], bool] | None) -> None:
        if should_stop is not None and should_stop():
            raise RuntimeError("DILIrank 2.0 update cancelled by user request")

    # -------------------------------------------------------------------------
    @staticmethod
    def _emit(
        progress_callback: Callable[[float, str], None] | None,
        progress: float,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(progress, message)
