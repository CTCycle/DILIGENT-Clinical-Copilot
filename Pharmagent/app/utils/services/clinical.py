from __future__ import annotations

import re
from xml.etree import ElementTree as ET

import httpx
from Pharmagent.app.api.models.prompts import (
    HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
    HEPATOTOXICITY_ANALYSIS_USER_PROMPT,
)
from Pharmagent.app.api.models.providers import initialize_llm_client
from Pharmagent.app.api.schemas.clinical import (
    DrugHepatotoxicityAnalysis,
    DrugToxicityFindings,
    HepatotoxicityPatternScore,
    PatientData,
    PatientDrugToxicityBundle,
    PatientDrugs,
)
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.logger import logger


###############################################################################
class HepatotoxicityPatternAnalyzer:
    
    # -------------------------------------------------------------------------
    def analyze(self, payload: PatientData) -> HepatotoxicityPatternScore:
        alt_value = self._parse_marker_value(payload.alt)
        alt_max_value = self._parse_marker_value(payload.alt_max)
        alp_value = self._parse_marker_value(payload.alp)
        alp_max_value = self._parse_marker_value(payload.alp_max)

        alt_multiple = self._safe_ratio(alt_value, alt_max_value)
        alp_multiple = self._safe_ratio(alp_value, alp_max_value)

        r_score: float | None = None
        if alt_multiple is not None and alp_multiple not in (None, 0.0):
            r_score = alt_multiple / alp_multiple

        classification = "indeterminate"
        if r_score is not None:
            if r_score > 5:
                classification = "hepatocellular"
            elif r_score < 2:
                classification = "cholestatic"
            else:
                classification = "mixed"

        return HepatotoxicityPatternScore(
            alt_multiple=alt_multiple,
            alp_multiple=alp_multiple,
            r_score=r_score,
            classification=classification,
        )

    # -------------------------------------------------------------------------
    def _parse_marker_value(self, raw: str | None) -> float | None:
        if raw is None:
            return None
        normalized = raw.replace(",", ".")
        match = re.search(r"[-+]?\d*\.?\d+", normalized)
        if not match:
            return None
        try:
            return float(match.group())
        except ValueError:
            return None

    # -------------------------------------------------------------------------
    def _safe_ratio(self, value: float | None, reference: float | None) -> float | None:
        if value is None or reference is None:
            return None
        if reference == 0:
            return None
        return value / reference


###############################################################################
class DrugToxicityEssay:
    def __init__(self, drugs: PatientDrugs, *, timeout_s: float = 300.0) -> None:
        self.drugs = drugs
        self.timeout_s = float(timeout_s)
        self.client = initialize_llm_client(purpose="agent", timeout_s=self.timeout_s)
        self.model = ClientRuntimeConfig.get_agent_model()
        self.max_prompt_chars = 6000
        self.http_timeout = httpx.Timeout(30.0)
        self._search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self._fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # -------------------------------------------------------------------------
    async def run(self) -> PatientDrugToxicityBundle:
        results: list[DrugHepatotoxicityAnalysis] = []
        for entry in self.drugs.entries:
            drug_name = entry.name.strip()
            if not drug_name:
                continue
            analysis = await self._process_drug(drug_name)
            results.append(analysis)
        return PatientDrugToxicityBundle(entries=results)

    # -------------------------------------------------------------------------
    async def _process_drug(self, drug_name: str) -> DrugHepatotoxicityAnalysis:
        source_text = await self._gather_livertox_text(drug_name)
        if not source_text:
            message = "No LiverTox monograph available for this drug."
            logger.warning("LiverTox data unavailable for drug '%s'", drug_name)
            return DrugHepatotoxicityAnalysis(
                drug_name=drug_name, source_text=None, analysis=None, error=message
            )

        prompt_text = self._prepare_prompt_text(source_text)
        user_prompt = HEPATOTOXICITY_ANALYSIS_USER_PROMPT.format(
            drug_name=drug_name,
            source_text=prompt_text,
        )

        try:
            findings = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=HEPATOTOXICITY_ANALYSIS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=DrugToxicityFindings,
                temperature=0.0,
                use_json_mode=True,
                max_repair_attempts=2,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "LLM hepatotoxicity analysis failed for '%s': %s", drug_name, exc
            )
            return DrugHepatotoxicityAnalysis(
                drug_name=drug_name,
                source_text=prompt_text,
                analysis=None,
                error=str(exc),
            )

        return DrugHepatotoxicityAnalysis(
            drug_name=drug_name,
            source_text=prompt_text,
            analysis=findings,
            error=None,
        )

    # -------------------------------------------------------------------------
    async def _gather_livertox_text(self, drug_name: str) -> str | None:
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            nbk_id = await self._search_livertox_id(client, drug_name)
            if not nbk_id:
                logger.info("No NBK identifier found for drug '%s'", drug_name)
                return None

            xml_payload = await self._fetch_livertox_entry(client, nbk_id)
            if not xml_payload:
                return None

        return self._extract_livertox_sections(xml_payload)

    # -------------------------------------------------------------------------
    async def _search_livertox_id(
        self,
        client: httpx.AsyncClient,
        drug_name: str,
    ) -> str | None:
        params = {
            "db": "books",
            "term": f"LiverTox[book] AND {drug_name}[title]",
            "retmode": "json",
        }

        try:
            response = await client.get(self._search_url, params=params)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to query LiverTox esearch for '%s': %s", drug_name, exc)
            return None

        id_list = payload.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return None
        return id_list[0]

    # -------------------------------------------------------------------------
    async def _fetch_livertox_entry(
        self,
        client: httpx.AsyncClient,
        nbk_id: str,
    ) -> str | None:
        params = {"db": "books", "id": nbk_id, "retmode": "xml"}

        try:
            response = await client.get(self._fetch_url, params=params)
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to fetch LiverTox entry '%s': %s", nbk_id, exc)
            return None

        return response.text

    # -------------------------------------------------------------------------
    def _extract_livertox_sections(self, xml_payload: str) -> str | None:
        try:
            root = ET.fromstring(xml_payload)
        except ET.ParseError as exc:
            logger.error("Failed to parse LiverTox XML: %s", exc)
            return None

        sections: list[str] = []
        excluded_titles = {"References"}

        for sect in root.findall(".//sect1"):
            title = (sect.findtext("title") or "").strip()
            if title in excluded_titles:
                continue

            paragraphs: list[str] = []
            for para in sect.findall("p"):
                text = "".join(para.itertext()).strip()
                if text:
                    paragraphs.append(text)

            if not paragraphs:
                continue

            if title:
                sections.append(f"{title}: {' '.join(paragraphs)}")
            else:
                sections.append(" ".join(paragraphs))

        if not sections:
            return None

        return "\n\n".join(sections)

    # -------------------------------------------------------------------------
    def _prepare_prompt_text(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if len(normalized) <= self.max_prompt_chars:
            return normalized
        return normalized[: self.max_prompt_chars]

