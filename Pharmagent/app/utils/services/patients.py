from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, Tuple

import pandas as pd

from Pharmagent.app.api.schemas.clinical import PatientData
from Pharmagent.app.constants import DATA_PATH
from Pharmagent.app.logger import logger

    

###############################################################################
class PatientCase:

    def __init__(self):
        self.HEADER_RE = re.compile(r'^[ \t]*#{1,6}[ \t]+(.+?)\s*$', re.MULTILINE)
        self.expected_tags = ("ANAMNESIS", "BLOOD TESTS", "ADDITIONAL TESTS", "DRUGS")
        self.response = {
            "name": "Unknown",
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
    def split_text_by_tags(self, text : str, name : str | None = None) -> Dict[str, Any]:        
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

        self.response['name'] = name or "Unknown"
        self.response["sections"] = sections
        self.response["unknown_headers"] = unknown
        self.response["missing_tags"] = missing 
        self.response["all_tags_present"] = not missing
       
        return sections
    
    #--------------------------------------------------------------------------
    def extract_sections_from_text(self, payload : PatientData) -> Tuple[Dict[str, Any], pd.DataFrame]:  
        full_text = self.clean_patient_info(payload.info)        
        sections = self.split_text_by_tags(full_text, payload.name)  

        patient_table = pd.DataFrame.from_dict([sections], orient="columns")
        patient_table['name'] = self.response['name']
        
        return sections, patient_table

                
        