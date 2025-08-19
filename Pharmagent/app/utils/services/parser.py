from __future__ import annotations

import os
import json
import unicodedata
from typing import Any, Dict, List, Optional

import pandas as pd

from Pharmagent.app.api.models.server import OllamaClient, OllamaError
from Pharmagent.app.utils.serializer import DataSerializer
from Pharmagent.app.api.models.prompts import DISEASE_EXTRACTION_PROMPT
from Pharmagent.app.constants import PARSER_MODEL
from Pharmagent.app.logger import logger

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")



###############################################################################
class DiseasesParsing:
    """

    Thin service that:
      1) Builds messages for disease extraction
      2) Uses OllamaClient to chat and to normalize the response

    
    """
    def __init__(self, base_url: Optional[str] = None,
        timeout_s: float = 180.0, temperature: float = 0.0) -> None:        
        self.temperature = float(temperature)
        self.client = OllamaClient(base_url=base_url, timeout_s=timeout_s)
        self.model = PARSER_MODEL

    #--------------------------------------------------------------------------
    def get_selected_model(self, model_name: Optional[str] = None) -> None:
        self.client.pull(model_name or self.model)

    #--------------------------------------------------------------------------
    async def extract_diseases_from_text(self, text: str) -> Dict[str, Any]:
        if text is None:
            return
        
        # LLM messages: system prompt + user content
        messages = [
            {"role": "system", "content": DISEASE_EXTRACTION_PROMPT},
            {"role": "user", "content": text}
            ]
        
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
            except Exception as e:
                logger.error(f"Could not parse LLM response as JSON: {llm_response}")
                return
            
        return data

 