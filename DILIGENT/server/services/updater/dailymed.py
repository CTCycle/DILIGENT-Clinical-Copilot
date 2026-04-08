from __future__ import annotations

import asyncio
import io
import re
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
from DILIGENT.server.configurations.bootstrap import server_settings
from DILIGENT.server.repositories.serialization.data import DataSerializer


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
    def update_labels(self, *, redownload: bool = False) -> dict[str, int]:
        _ = redownload
        targets = self.load_target_drugs()
        mapping = self.load_rxnorm_to_setid_mapping()
        records = self.persist_documents(targets, mapping)
        return {
            "target_drugs": len(targets),
            "mapped_drugs": sum(1 for target in targets if target["rxcui"] in mapping),
            "documents_persisted": len(records),
        }

    # -------------------------------------------------------------------------
    def load_target_drugs(self) -> list[dict[str, Any]]:
        catalog = self.serializer.stream_drugs_catalog()
        if catalog.empty:
            return []
        rows: list[dict[str, Any]] = []
        for row in catalog.to_dict(orient="records"):
            drug_id = self.serializer.to_int(row.get("drug_id"))
            rxcui = self.clean_text(row.get("rxcui"))
            drug_name = self.clean_text(row.get("canonical_name"))
            if drug_id is None or rxcui is None or drug_name is None:
                continue
            rows.append({"drug_id": drug_id, "drug_name": drug_name, "rxcui": rxcui})
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
    async def download_label_xml(self, set_id: str) -> str:
        url = f"{DAILYMED_LABEL_XML_BASE_URL}/{set_id}.xml"
        async with httpx.AsyncClient(timeout=self.request_timeout, follow_redirects=True) as client:
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
    ) -> list[dict[str, Any]]:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def load_one(target: dict[str, Any]) -> dict[str, Any] | None:
            best = self.select_best_label_for_drug(target["rxcui"], mapping)
            if best is None:
                return None
            set_id = self.clean_text(best.get("set_id"))
            if set_id is None:
                return None
            async with semaphore:
                xml_text = await self.download_label_xml(set_id)
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

        async def gather_all() -> list[dict[str, Any]]:
            tasks = [asyncio.create_task(load_one(target)) for target in targets]
            rows: list[dict[str, Any]] = []
            for task in tasks:
                result = await task
                if result is not None:
                    rows.append(result)
            return rows

        try:
            rows = loop.run_until_complete(gather_all())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        self.serializer.replace_drug_label_documents(rows)
        return rows

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
