from __future__ import annotations

import asyncio
import io
import re
import zipfile
from collections.abc import Callable
from typing import Any
from xml.etree import ElementTree

import httpx
import pandas as pd

from DILIGENT.server.common.constants import (
    DAILYMED_LABEL_XML_BASE_URL,
    DAILYMED_RXNORM_SETID_MAPPING_URL,
    DAILYMED_SECTION_WHITELIST,
    HEPATIC_KEYWORDS,
)
from DILIGENT.server.configurations.startup import server_settings
from DILIGENT.server.repositories.serialization.data import DataSerializer

ProgressCallback = Callable[[float, str], None]


###############################################################################
class DailyMedLabelUpdater:
    XML_NS = {
        "hl7": "urn:hl7-org:v3",
    }
    SECTION_KEY_LOOKUP = {
        "boxed_warning": "boxed_warning",
        "boxed warning": "boxed_warning",
        "box warning": "boxed_warning",
        "warnings and precautions": "warnings_and_precautions",
        "adverse reactions": "adverse_reactions",
        "contraindications": "contraindications",
        "use in specific populations": "use_in_specific_populations",
    }

    def __init__(
        self,
        *,
        serializer: DataSerializer | None = None,
        request_timeout: float | None = None,
        max_concurrency: int | None = None,
        section_max_length: int | None = None,
        max_sections_per_drug: int | None = None,
    ) -> None:
        self.serializer = serializer or DataSerializer()
        settings = server_settings.external_data
        self.request_timeout = max(
            float(request_timeout)
            if request_timeout is not None
            else float(settings.dailymed_request_timeout),
            1.0,
        )
        self.max_concurrency = max(
            int(max_concurrency)
            if max_concurrency is not None
            else int(settings.dailymed_max_concurrency),
            1,
        )
        self.section_max_length = max(
            int(section_max_length)
            if section_max_length is not None
            else int(settings.dailymed_section_max_length),
            128,
        )
        self.max_sections_per_drug = max(
            int(max_sections_per_drug)
            if max_sections_per_drug is not None
            else int(settings.dailymed_max_sections_per_drug),
            1,
        )
        self.keyword_tokens = {token.casefold() for token in HEPATIC_KEYWORDS}

    # -------------------------------------------------------------------------
    def update_labels(
        self,
        *,
        redownload: bool = False,
        progress_callback: ProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, int]:
        _ = redownload
        self.emit_progress(progress_callback, progress=13.0, message="Loading drug catalog")
        targets = self.load_target_drugs(
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        self.emit_progress(
            progress_callback,
            progress=20.0,
            message=f"Resolved {len(targets)} DailyMed target drugs",
        )
        if self.should_cancel(should_stop):
            raise RuntimeError("Drug labels update cancelled by user request")
        self.emit_progress(
            progress_callback,
            progress=24.0,
            message="Loading DailyMed RxNorm mapping",
        )
        mapping = self.load_rxnorm_to_setid_mapping()
        mapping_rows = sum(len(items) for items in mapping.values())
        self.emit_progress(
            progress_callback,
            progress=30.0,
            message=f"Loaded {mapping_rows} DailyMed mapping rows",
        )
        records = self.persist_documents(
            targets,
            mapping,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        return {
            "target_drugs": len(targets),
            "mapped_drugs": sum(1 for target in targets if target["rxcui"] in mapping),
            "documents_persisted": len(records),
        }

    # -------------------------------------------------------------------------
    def load_target_drugs(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        catalog = self.serializer.get_drugs_catalog()
        if catalog.empty:
            return []
        db_session = self.serializer.session_factory()
        rxcui_to_drug_id: dict[str, int | None] = {}
        rows: list[dict[str, Any]] = []
        try:
            total_rows = max(len(catalog), 1)
            report_interval = max(1, total_rows // 25)
            for index, row in enumerate(catalog.itertuples(index=False), start=1):
                if self.should_cancel(should_stop):
                    raise RuntimeError("Drug labels update cancelled by user request")
                rxcui = self.clean_text(getattr(row, "rxcui", None))
                drug_name = self.clean_text(getattr(row, "name", None)) or self.clean_text(
                    getattr(row, "raw_name", None)
                )
                if rxcui is None or drug_name is None:
                    continue
                if rxcui not in rxcui_to_drug_id:
                    rxcui_to_drug_id[rxcui] = self.serializer.resolve_drug_id(
                        db_session,
                        matched_drug_name=drug_name,
                        rxcui=rxcui,
                        nbk_id=None,
                    )
                drug_id = rxcui_to_drug_id[rxcui]
                if drug_id is None:
                    continue
                rows.append({"drug_id": drug_id, "drug_name": drug_name, "rxcui": rxcui})
                if progress_callback is not None and (
                    index % report_interval == 0 or index == total_rows
                ):
                    progress = 13.0 + (index / total_rows) * 7.0
                    self.emit_progress(
                        progress_callback,
                        progress=progress,
                        message=(
                            f"Resolved {len(rows)} DailyMed target drugs "
                            f"({index}/{total_rows})"
                        ),
                    )
        finally:
            db_session.close()
        return rows

    # -------------------------------------------------------------------------
    def load_rxnorm_to_setid_mapping(self) -> dict[str, list[dict[str, Any]]]:
        with httpx.Client(timeout=self.request_timeout, follow_redirects=True) as client:
            response = client.get(DAILYMED_RXNORM_SETID_MAPPING_URL)
            response.raise_for_status()
            content = response.content
            content_type = (response.headers.get("content-type") or "").lower()
        frame = self.parse_mapping_frame(content, content_type)
        by_rxcui: dict[str, list[dict[str, Any]]] = {}
        if frame.empty:
            return by_rxcui
        lowered = {str(column).strip().lower(): column for column in frame.columns}
        rxcui_col = self.pick_column(lowered, ["rxcui", "rxnorm", "rxnorm_rxcui"])
        setid_col = self.pick_column(lowered, ["setid", "set_id"])
        spl_col = self.pick_column(lowered, ["spl_version", "version"])
        effective_col = self.pick_column(lowered, ["effective_date", "effective time", "effective"])
        status_col = self.pick_column(lowered, ["status", "state"])
        title_col = self.pick_column(lowered, ["title"])
        labeler_col = self.pick_column(lowered, ["labeler", "labeler_name"])
        for row in frame.to_dict(orient="records"):
            rxcui = self.clean_text(row.get(rxcui_col))
            set_id = self.clean_text(row.get(setid_col))
            if rxcui is None or set_id is None:
                continue
            item = {
                "rxcui": rxcui,
                "set_id": set_id,
                "spl_version": self.serializer.to_int(row.get(spl_col)) or 1,
                "effective_date": self.serializer.normalize_date(row.get(effective_col)),
                "status": self.clean_text(row.get(status_col)),
                "title": self.clean_text(row.get(title_col)),
                "labeler": self.clean_text(row.get(labeler_col)),
            }
            by_rxcui.setdefault(rxcui, []).append(item)
        return by_rxcui

    # -------------------------------------------------------------------------
    def parse_mapping_frame(self, content: bytes, content_type: str) -> pd.DataFrame:
        if content.startswith(b"PK"):
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    for name in archive.namelist():
                        lowered = name.lower()
                        if lowered.endswith(".csv"):
                            return pd.read_csv(io.BytesIO(archive.read(name)))
                        if lowered.endswith((".txt", ".tsv")):
                            text = archive.read(name).decode("utf-8", errors="ignore")
                            return pd.read_csv(io.StringIO(text), sep="|")
            except Exception:
                return pd.DataFrame()
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
    def select_best_label_for_drug(
        self,
        rxcui: str,
        mapping: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any] | None:
        candidates = mapping.get(rxcui, [])
        if not candidates:
            return None
        active = [
            item
            for item in candidates
            if (self.clean_text(item.get("status")) or "active").casefold() == "active"
        ]
        pool = active or candidates
        sorted_pool = sorted(
            pool,
            key=lambda item: (
                self.serializer.to_sortable_text(item.get("effective_date")),
                int(item.get("spl_version") or 0),
                self.serializer.to_sortable_text(item.get("set_id")),
            ),
            reverse=True,
        )
        return sorted_pool[0] if sorted_pool else None

    # -------------------------------------------------------------------------
    async def download_label_xml(self, set_id: str, *, client: httpx.AsyncClient) -> str:
        url = f"{DAILYMED_LABEL_XML_BASE_URL}/{set_id}.xml"
        response = await client.get(url)
        response.raise_for_status()
        return response.text

    # -------------------------------------------------------------------------
    def extract_relevant_sections(self, xml_text: str) -> list[dict[str, Any]]:
        if not xml_text.strip():
            return []
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            return []
        collected: list[dict[str, Any]] = []
        for section in root.findall(".//hl7:section", namespaces=self.XML_NS):
            title_node = section.find(".//hl7:title", namespaces=self.XML_NS)
            title_text = ""
            if title_node is not None:
                title_text = self.normalize_whitespace(
                    " ".join(part for part in title_node.itertext() if part)
                )
            title_key = self.map_section_key(title_text)
            if title_key is None or title_key not in DAILYMED_SECTION_WHITELIST:
                continue
            paragraphs = [
                self.normalize_whitespace(" ".join(part for part in node.itertext() if part))
                for node in section.findall(".//hl7:paragraph", namespaces=self.XML_NS)
            ]
            paragraphs = [item for item in paragraphs if item]
            if title_key == "boxed_warning":
                kept_text = self.normalize_whitespace("\n".join(paragraphs)[: self.section_max_length])
                if kept_text:
                    collected.append(
                        {
                            "section_key": title_key,
                            "section_title": title_text or title_key.replace("_", " ").title(),
                            "text": kept_text,
                            "contains_hepatic_keywords": self.contains_hepatic_keywords(kept_text),
                        }
                    )
                continue
            kept = [item for item in paragraphs if self.contains_hepatic_keywords(item)]
            joined = self.normalize_whitespace("\n".join(kept))
            if not joined:
                continue
            truncated = joined[: self.section_max_length].strip()
            if truncated:
                collected.append(
                    {
                        "section_key": title_key,
                        "section_title": title_text or title_key.replace("_", " ").title(),
                        "text": truncated,
                        "contains_hepatic_keywords": True,
                    }
                )
        sorted_sections = sorted(
            collected,
            key=lambda item: DAILYMED_SECTION_WHITELIST.index(item["section_key"]),
        )
        trimmed = sorted_sections[: self.max_sections_per_drug]
        for index, item in enumerate(trimmed):
            item["display_order"] = index
        return trimmed

    # -------------------------------------------------------------------------
    def persist_documents(
        self,
        targets: list[dict[str, Any]],
        mapping: dict[str, list[dict[str, Any]]],
        *,
        progress_callback: ProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rows = loop.run_until_complete(
                self._persist_documents_async(
                    targets=targets,
                    mapping=mapping,
                    progress_callback=progress_callback,
                    should_stop=should_stop,
                )
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        self.serializer.replace_drug_label_documents(rows)
        return rows

    # -------------------------------------------------------------------------
    async def _persist_documents_async(
        self,
        *,
        targets: list[dict[str, Any]],
        mapping: dict[str, list[dict[str, Any]]],
        progress_callback: ProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        limits = httpx.Limits(
            max_connections=self.max_concurrency,
            max_keepalive_connections=self.max_concurrency,
        )
        async with httpx.AsyncClient(
            timeout=self.request_timeout,
            follow_redirects=True,
            limits=limits,
        ) as client:
            return await self.gather_persistable_rows(
                targets=targets,
                mapping=mapping,
                client=client,
                progress_callback=progress_callback,
                should_stop=should_stop,
            )

    # -------------------------------------------------------------------------
    async def gather_persistable_rows(
        self,
        *,
        targets: list[dict[str, Any]],
        mapping: dict[str, list[dict[str, Any]]],
        client: httpx.AsyncClient,
        progress_callback: ProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        total = len(targets)
        if total == 0:
            return []
        results: list[dict[str, Any] | None] = [None] * total
        queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()
        for index, target in enumerate(targets):
            queue.put_nowait((index, target))
        worker_count = min(self.max_concurrency, total)
        for _ in range(worker_count):
            queue.put_nowait(None)
        progress_state = {"completed": 0}
        progress_lock = asyncio.Lock()
        report_interval = max(1, total // 50)
        error_message = "Drug labels update cancelled by user request"

        await asyncio.gather(
            *(
                self.gather_persistable_rows_worker(
                    queue=queue,
                    results=results,
                    mapping=mapping,
                    client=client,
                    total=total,
                    report_interval=report_interval,
                    progress_callback=progress_callback,
                    should_stop=should_stop,
                    progress_state=progress_state,
                    progress_lock=progress_lock,
                    error_message=error_message,
                )
                for _ in range(worker_count)
            )
        )
        return [item for item in results if item is not None]

    # -------------------------------------------------------------------------
    async def gather_persistable_rows_worker(
        self,
        *,
        queue: asyncio.Queue[tuple[int, dict[str, Any]] | None],
        results: list[dict[str, Any] | None],
        mapping: dict[str, list[dict[str, Any]]],
        client: httpx.AsyncClient,
        total: int,
        report_interval: int,
        progress_callback: ProgressCallback | None,
        should_stop: Callable[[], bool] | None,
        progress_state: dict[str, int],
        progress_lock: asyncio.Lock,
        error_message: str,
    ) -> None:
        while True:
            item = await queue.get()
            if item is None:
                return
            index, target = item
            if self.should_cancel(should_stop):
                raise RuntimeError(error_message)
            result = await self.load_document_row(
                target=target,
                mapping=mapping,
                client=client,
            )
            results[index] = result
            async with progress_lock:
                progress_state["completed"] += 1
                completed = progress_state["completed"]
            if progress_callback is not None and (
                completed % report_interval == 0 or completed == total
            ):
                progress = 30.0 + (completed / total) * 58.0
                self.emit_progress(
                    progress_callback,
                    progress=progress,
                    message=f"Downloaded {completed}/{total} DailyMed labels",
                )

    # -------------------------------------------------------------------------
    async def load_document_row(
        self,
        *,
        target: dict[str, Any],
        mapping: dict[str, list[dict[str, Any]]],
        client: httpx.AsyncClient,
    ) -> dict[str, Any] | None:
        best = self.select_best_label_for_drug(target["rxcui"], mapping)
        if best is None:
            return None
        set_id = self.clean_text(best.get("set_id"))
        if set_id is None:
            return None
        xml_text = await self.download_label_xml(set_id, client=client)
        sections = self.extract_relevant_sections(xml_text)
        if not sections:
            return None
        return {
            "drug_id": target["drug_id"],
            "source": "dailymed",
            "set_id": set_id,
            "spl_version": int(best.get("spl_version") or 1),
            "title": self.clean_text(best.get("title")),
            "labeler": self.clean_text(best.get("labeler")),
            "effective_date": self.serializer.normalize_date(best.get("effective_date")),
            "source_url": f"{DAILYMED_LABEL_XML_BASE_URL}/{set_id}.xml",
            "source_last_modified": None,
            "sections": sections,
        }

    # -------------------------------------------------------------------------
    def contains_hepatic_keywords(self, text: str) -> bool:
        candidate = text.casefold()
        return any(token in candidate for token in self.keyword_tokens)

    # -------------------------------------------------------------------------
    def map_section_key(self, raw_title: str) -> str | None:
        normalized = raw_title.casefold()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized in self.SECTION_KEY_LOOKUP:
            return self.SECTION_KEY_LOOKUP[normalized]
        for key, value in self.SECTION_KEY_LOOKUP.items():
            if key in normalized:
                return value
        return None

    # -------------------------------------------------------------------------
    def emit_progress(
        self,
        progress_callback: ProgressCallback | None,
        *,
        progress: float,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(progress, message)

    # -------------------------------------------------------------------------
    @staticmethod
    def should_cancel(should_stop: Callable[[], bool] | None) -> bool:
        return bool(should_stop and should_stop())

    # -------------------------------------------------------------------------
    @staticmethod
    def normalize_whitespace(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

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

