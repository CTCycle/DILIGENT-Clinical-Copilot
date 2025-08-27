from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


from Pharmagent.app.api.models.providers import OllamaClient, OllamaError
from Pharmagent.app.api.schemas.diseases import PatientDiseases
from Pharmagent.app.api.models.prompts import DISEASE_EXTRACTION_PROMPT
from Pharmagent.app.constants import PARSER_MODEL
from Pharmagent.app.logger import logger

    
###############################################################################
class HepatoPatterns:
    
    def __init__(self, base_url: Optional[str] = None,
        timeout_s: float = 180.0, temperature: float = 0.0) -> None:        
        self.temperature = float(temperature)
        self.client = OllamaClient(base_url=base_url, timeout_s=timeout_s)
        self.model = PARSER_MODEL
        self.JSON_schema = {'diseases': List[str], 
                            'hepatic_diseases': List[str]}

    #-------------------------------------------------------------------------
    def get_selected_model(self, model_name: Optional[str] = None) -> None:
        self.client.pull(model_name or self.model)

    #-------------------------------------------------------------------------
    def normalize_unique(self, lst):
        seen = set()
        result = []
        for x in lst:
            norm = x.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                result.append(norm)
        return result

    #-------------------------------------------------------------------------
    async def extract_diseases(self, text: str) -> Dict[str, Any]:
        if not text:
            return
        
        # LLM messages: system prompt + user content
        messages = [
            {"role": "system", "content": DISEASE_EXTRACTION_PROMPT},
            {"role": "user", "content": text}]
        
        try:
            llm_response = await self.client.chat(
                model=self.model,
                messages=messages,
                format="json")
            
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
    #-------------------------------------------------------------------------
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
                max_repair_attempts=2)
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract diseases (structured): {e}") from e

        diseases = self.normalize_unique(parsed.diseases)
        hepatic = [h for h in self.normalize_unique(parsed.hepatic_diseases) if h in diseases]

        return {"diseases": diseases, "hepatic_diseases": hepatic}
    
    #-------------------------------------------------------------------------
    def validate_json_schema(self, output: dict) -> dict:    
        for key in ['diseases', 'hepatic_diseases']:
            if key not in output or not isinstance(output[key], list):
                raise ValueError(f"Missing or invalid field: '{key}'. Must be a list.")
            if not all(isinstance(x, str) for x in output[key]):
                raise ValueError(f"All entries in '{key}' must be strings.")       

        diseases = self.normalize_unique(output['diseases'])
        hepatic_diseases = self.normalize_unique(output['hepatic_diseases'])

        # Subset validation
        if not set(hepatic_diseases).issubset(set(diseases)):
            missing = set(hepatic_diseases) - set(diseases)
            raise ValueError("hepatic diseases were not validated")

        return {
            'diseases': diseases,
            'hepatic_diseases': hepatic_diseases}
    





                
        

