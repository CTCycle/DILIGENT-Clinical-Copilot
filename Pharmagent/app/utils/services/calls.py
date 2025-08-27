from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Tuple

import pandas as pd

from Pharmagent.app.api.models.prompts import DISEASE_EXTRACTION_PROMPT
from Pharmagent.app.api.models.providers import OllamaClient, OllamaError
from Pharmagent.app.api.schemas.patients import PatientData, PatientDiseases
from Pharmagent.app.constants import PARSER_MODEL
from Pharmagent.app.logger import logger

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


###############################################################################
class PatientCase:
    def __init__(self):
        self.HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+(.+?)\s*$", re.MULTILINE)
        self.expected_tags = ("ANAMNESIS", "BLOOD TESTS", "ADDITIONAL TESTS", "DRUGS")
        self.response = {
            "name": "Unknown",
            "sections": {},
            "unknown_headers": [],
            "missing_tags": list(self.expected_tags),
            "all_tags_present": False,
        }

    # -------------------------------------------------------------------------
    def clean_patient_info(self, text: str) -> str:
        # Normalize unicode width/compatibility (e.g., μ → μ, fancy quotes → ASCII where possible)
        processed_text = unicodedata.normalize("NFKC", text)
        # Normalize newlines
        processed_text = processed_text.replace("\r\n", "\n").replace("\r", "\n")
        # Strip trailing spaces on each line
        processed_text = "\n".join(line.rstrip() for line in processed_text.split("\n"))
        # Collapse 3+ blank lines to max 2, and leading/trailing blank lines
        processed_text = re.sub(r"\n{3,}", "\n\n", processed_text).strip()

        return processed_text

    # -------------------------------------------------------------------------
    def split_text_by_tags(self, text: str, name: str | None = None) -> Dict[str, Any]:
        hits = [
            (m.group(1).strip(), m.start(), m.end())
            for m in self.HEADER_RE.finditer(text)
        ]
        if not hits:
            return self.response

        sections = {
            title.replace(" ", "_").lower(): text[
                end : (hits[i + 1][1] if i + 1 < len(hits) else len(text))
            ].strip()
            for i, (title, _start, end) in enumerate(hits)
        }

        exp_lower = {e.lower() for e in self.expected_tags}
        found_map = {k.lower(): k for k in sections}
        missing = [e for e in self.expected_tags if e.lower() not in found_map]
        unknown = [
            orig
            for low, orig in ((k.lower(), k) for k in sections)
            if low not in exp_lower
        ]

        self.response["name"] = name or "Unknown"
        self.response["sections"] = sections
        self.response["unknown_headers"] = unknown
        self.response["missing_tags"] = missing
        self.response["all_tags_present"] = not missing

        return sections

    # -------------------------------------------------------------------------
    def extract_sections_from_text(
        self, payload: PatientData
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        full_text = self.clean_patient_info(payload.info)
        sections = self.split_text_by_tags(full_text, payload.name)

        patient_table = pd.DataFrame.from_dict([sections], orient="columns")
        patient_table["name"] = self.response["name"]

        return sections, patient_table


###############################################################################
class DiseasesParsing:
    def __init__(self, timeout_s: float = 300.0, temperature: float = 0.0) -> None:
        self.temperature = float(temperature)
        self.client = OllamaClient(base_url=None, timeout_s=timeout_s)
        self.JSON_schema = {"diseases": List[str], "hepatic_diseases": List[str]}
        self.model = PARSER_MODEL

    # -------------------------------------------------------------------------
    def normalize_unique(self, lst: List[str]):
        seen = set()
        result = []
        for x in lst:
            norm = x.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)

        return result

    # -------------------------------------------------------------------------
    async def extract_diseases(self, text: str) -> Dict[str, Any]:
        if not text:
            return

        # LLM messages: system prompt + user content
        messages = [
            {"role": "system", "content": DISEASE_EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ]

        try:
            llm_response = await self.client.chat(
                model=self.model, messages=messages, format="json"
            )

        except OllamaError as e:
            # Customize exception handling/logging as needed
            raise RuntimeError(f"Failed to extract diseases: {e}") from e

        data = None
        # 1. Parse LLM response to dict if necessary
        if isinstance(llm_response, dict):
            data = llm_response
        elif isinstance(llm_response, str):
            try:
                data = json.loads(llm_response)
            except Exception:
                logger.error(f"Could not parse LLM response as JSON: {llm_response}")
                return

        data = self.validate_json_schema(data)

        return data

    # uses lanchain as wrapper to perform persing and validation to patient diseases model
    # -------------------------------------------------------------------------
    async def extract_diseases(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"diseases": [], "hepatic_diseases": []}
        try:
            parsed: PatientDiseases = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=DISEASE_EXTRACTION_PROMPT,
                user_prompt=text,
                schema=PatientDiseases,
                temperature=self.temperature,
                use_json_mode=True,
                max_repair_attempts=2,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to extract diseases (structured): {e}") from e

        diseases = self.normalize_unique(parsed.diseases)
        hepatic = [
            h for h in self.normalize_unique(parsed.hepatic_diseases) if h in diseases
        ]

        return {"diseases": diseases, "hepatic_diseases": hepatic}

    # -------------------------------------------------------------------------
    def validate_json_schema(self, output: dict) -> dict:
        for key in ["diseases", "hepatic_diseases"]:
            if key not in output or not isinstance(output[key], list):
                raise ValueError(f"Missing or invalid field: '{key}'. Must be a list.")
            if not all(isinstance(x, str) for x in output[key]):
                raise ValueError(f"All entries in '{key}' must be strings.")

        diseases = self.normalize_unique(output["diseases"])
        hepatic_diseases = self.normalize_unique(output["hepatic_diseases"])

        # Subset validation
        if not set(hepatic_diseases).issubset(set(diseases)):
            missing = set(hepatic_diseases) - set(diseases)
            raise ValueError("hepatic diseases were not validated")

        return {"diseases": diseases, "hepatic_diseases": hepatic_diseases}


###############################################################################
class DiseasesParsing:
    def __init__(self, timeout_s: float = 300.0, temperature: float = 0.0) -> None:
        self.temperature = float(temperature)
        self.client = OllamaClient(base_url=None, timeout_s=timeout_s)
        self.JSON_schema = {"diseases": List[str], "hepatic_diseases": List[str]}
        self.model = PARSER_MODEL

    # -------------------------------------------------------------------------
    def normalize_unique(self, lst: List[str]):
        seen = set()
        result = []
        for x in lst:
            norm = x.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)

        return result

    # -------------------------------------------------------------------------
    async def extract_diseases(self, text: str) -> Dict[str, Any]:
        if not text:
            return

        # LLM messages: system prompt + user content
        messages = [
            {"role": "system", "content": DISEASE_EXTRACTION_PROMPT},
            {"role": "user", "content": text},
        ]

        try:
            llm_response = await self.client.chat(
                model=self.model, messages=messages, format="json"
            )

        except OllamaError as e:
            # Customize exception handling/logging as needed
            raise RuntimeError(f"Failed to extract diseases: {e}") from e

        data = None
        # 1. Parse LLM response to dict if necessary
        if isinstance(llm_response, dict):
            data = llm_response
        elif isinstance(llm_response, str):
            try:
                data = json.loads(llm_response)
            except Exception:
                logger.error(f"Could not parse LLM response as JSON: {llm_response}")
                return

        data = self.validate_json_schema(data)

        return data

    # uses lanchain as wrapper to perform persing and validation to patient diseases model
    # -------------------------------------------------------------------------
    async def extract_diseases(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"diseases": [], "hepatic_diseases": []}
        try:
            parsed: PatientDiseases = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=DISEASE_EXTRACTION_PROMPT,
                user_prompt=text,
                schema=PatientDiseases,
                temperature=self.temperature,
                use_json_mode=True,
                max_repair_attempts=2,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to extract diseases (structured): {e}") from e

        diseases = self.normalize_unique(parsed.diseases)
        hepatic = [
            h for h in self.normalize_unique(parsed.hepatic_diseases) if h in diseases
        ]

        return {"diseases": diseases, "hepatic_diseases": hepatic}

    # -------------------------------------------------------------------------
    def validate_json_schema(self, output: dict) -> dict:
        for key in ["diseases", "hepatic_diseases"]:
            if key not in output or not isinstance(output[key], list):
                raise ValueError(f"Missing or invalid field: '{key}'. Must be a list.")
            if not all(isinstance(x, str) for x in output[key]):
                raise ValueError(f"All entries in '{key}' must be strings.")

        diseases = self.normalize_unique(output["diseases"])
        hepatic_diseases = self.normalize_unique(output["hepatic_diseases"])

        # Subset validation
        if not set(hepatic_diseases).issubset(set(diseases)):
            missing = set(hepatic_diseases) - set(diseases)
            raise ValueError("hepatic diseases were not validated")

        return {"diseases": diseases, "hepatic_diseases": hepatic_diseases}
