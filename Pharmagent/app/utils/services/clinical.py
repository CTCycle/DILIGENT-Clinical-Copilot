from __future__ import annotations


from Pharmagent.app.api.schemas.clinical import PatientDrugs
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.api.models.providers import initialize_llm_client


###############################################################################
class DrugToxicityEssay:
    def __init__(self, drugs: PatientDrugs) -> None:
        for drug in drugs.entries:
            pass
