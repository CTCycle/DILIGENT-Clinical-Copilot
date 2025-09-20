from __future__ import annotations

import json
from typing import Any, List, Optional

from Pharmagent.app.api.models.prompts import DISEASE_EXTRACTION_PROMPT
from Pharmagent.app.api.models.providers import OllamaClient, OllamaError
from Pharmagent.app.api.schemas.clinical import PatientDiseases
from Pharmagent.app.constants import PARSER_MODEL
from Pharmagent.app.logger import logger


###############################################################################
class HepatoPatterns:
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_s: float = 180.0,
        temperature: float = 0.0,
    ) -> None:
        self.temperature = float(temperature)
        self.client = OllamaClient(base_url=base_url, timeout_s=timeout_s)
        self.model = PARSER_MODEL
        self.JSON_schema = {"diseases": list[str], "hepatic_diseases": list[str]}
