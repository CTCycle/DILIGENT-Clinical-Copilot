from __future__ import annotations


from Pharmagent.app.api.schemas.clinical import PatientDrugs
from Pharmagent.app.configurations import ClientRuntimeConfig
from Pharmagent.app.api.models.providers import initialize_llm_client


###############################################################################
class HepatoPatterns:
    def __init__(
        self,
        base_url: str | None = None,
        timeout_s: float = 180.0,
        temperature: float = 0.0,
    ) -> None:
        self.temperature = float(temperature)
        client_kwargs: dict[str, float | str | None] = {"timeout_s": timeout_s}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = initialize_llm_client(purpose="parser", **client_kwargs)
        self.model = ClientRuntimeConfig.get_parsing_model()
        self.JSON_schema = {"diseases": list[str], "hepatic_diseases": list[str]}


###############################################################################
class DrugToxicityEssay:
    def __init__(
        self,
        drugs: PatientDrugs        
    ) -> None:
        for drug in drugs.entries:
            pass
