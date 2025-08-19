from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd

from Pharmagent.app.api.models.server import OllamaClient
from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.api.schemas.clinical import PatientData, PatientOutputReport

from Pharmagent.app.api.models.prompts import DISEASE_EXTRACTION_PROMPT
from Pharmagent.app.constants import DATA_PATH
from Pharmagent.app.logger import logger

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")



###############################################################################
class DiseasesExtraction:
    """
    Thin service that:
      1) Builds messages for disease extraction
      2) Uses OllamaClient to chat and to normalize the response

    All parsing/validation lives in OllamaClient.
    """

    def __init__(self, model: str = "llama3.1:8b", base_url: Optional[str] = None,
        timeout_s: float = 60.0, temperature: float = 0.0) -> None:        
        self.temperature = float(temperature)
        self.client = OllamaClient(base_url=base_url, timeout_s=timeout_s)
        self.model = model

    #--------------------------------------------------------------------------
    def get_selected_model(self, model_name: Optional[str] = None) -> None:
        self.client.pull(model_name or self.model)

    #--------------------------------------------------------------------------
    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Returns: {"diseases": [...]}  (empty list on failure)
        """
        if not isinstance(text, str) or not text.strip():
            return {"diseases": []}

        messages = [
            {
                "role": "system",
                "content": (
                    "You extract ONLY disease/condition names about the patient from clinical text. "
                    "No medications, labs, procedures, or pure symptoms unless they are diseases. "
                    "Return STRICT JSON ONLY in this shape: "
                    '{"diseases":[{"name":"<disease>"}]} . If none, return {"diseases":[]}.'
                ),
            },
            {
                "role": "user",
                "content": f"Clinical text:\n-----\n{text}\n-----\nReturn JSON ONLY.",
            },
        ]

        raw = self.client.chat(
            model=self.model,
            messages=messages,
            format="json",
            options={"temperature": self.temperature},
        )
        return self.client.normalize_diseases(raw)
    

###############################################################################
class PatientInfoExtraction:

    def __init__(self, patient_payload: PatientData):
        self.HEADER_RE = re.compile(r'^[ \t]*#{1,6}[ \t]+(.+?)\s*$', re.MULTILINE)
        self.expected_tags = ("ANAMNESIS", "BLOOD TESTS", "ADDITIONAL TESTS", "DRUGS")
        self.patient_payload = patient_payload

        self.response = {
            "name": patient_payload.name or "Unknown",
            "sections": {}, 
            "unknown_headers": [], 
            "missing_tags": list(self.expected_tags), 
            "all_tags_present": False}        

    #--------------------------------------------------------------------------
    def clean_patient_info(self, text: str) -> str:        
        # Normalize unicode width/compatibility (e.g., μ → μ, fancy quotes → ASCII where possible)
        processed_text = unicodedata.normalize("NFKC", text)
        # Normalize newlines
        processed_text = processed_text.replace("\r\n", "\n").replace("\r", "\n")
        # Strip trailing spaces on each line
        processed_text = "\n".join(line.rstrip() for line in processed_text.split("\n"))
        # Collapse 3+ blank lines to max 2, and leading/trailing blank lines
        processed_text = re.sub(r'\n{3,}', '\n\n', processed_text).strip()

        return processed_text 
    
    #--------------------------------------------------------------------------
    def split_text_by_tags(self, text : str) -> Dict[str, Any]:        
        hits = [(m.group(1).strip(), m.start(), m.end()) for m in self.HEADER_RE.finditer(text)]
        if not hits:
            return self.response

        sections = {
            title.replace(' ', '_').lower(): text[end:(hits[i+1][1] if i+1 < len(hits) else len(text))].strip()
            for i, (title, _start, end) in enumerate(hits)}

        exp_lower = {e.lower() for e in self.expected_tags}
        found_map = {k.lower(): k for k in sections}
        missing = [e for e in self.expected_tags if e.lower() not in found_map]
        unknown = [orig for low, orig in ((k.lower(), k) for k in sections) if low not in exp_lower]

        self.response["sections"] = sections
        self.response["unknown_headers"] = unknown
        self.response["missing_tags"] = missing 
        self.response["all_tags_present"] = not missing

        data = pd.DataFrame.from_dict([sections], orient="columns")
        data['name'] = self.patient_payload.name or "Unknown"
       
        return data
    
    #--------------------------------------------------------------------------
    def extract_textual_sections(self) -> Dict[str, Any]:     
        text = self.clean_patient_info(self.patient_payload.info) 
        patient_info = self.split_text_by_tags(text)        
        
        return patient_info

                
        