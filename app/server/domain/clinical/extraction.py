from __future__ import annotations

from pydantic import BaseModel


class LlmClinicalSectionTextDraft(BaseModel):
    anamnesis: str = ""
    therapy: str = ""
    lab_analysis: str = ""
