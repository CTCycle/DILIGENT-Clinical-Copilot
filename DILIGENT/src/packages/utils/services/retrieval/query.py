from __future__ import annotations

from collections import defaultdict

from DILIGENT.src.app.backend.models.prompts import DILI_RAG_QUERY_PROMPT
from DILIGENT.src.app.backend.schemas.clinical import PatientDrugs
from DILIGENT.src.packages.constants import (
    DEFAULT_DILI_CLASSIFICATION,
    NO_CLINICAL_CONTEXT_FALLBACK,
    UNKNOWN_R_SCORE_TOKEN,
)


###############################################################################
class DILIQueryBuilder:

    def __init__(self, drugs : PatientDrugs) -> None: 
        self.drug_names = [x.name for x in drugs.entries if x.name]

    # -------------------------------------------------------------------------
    def build_dili_queries(
        self,
        *,        
        clinical_context: str,
        pattern_classification: str | None,
        r_score: float | None,      
    ) -> dict[str, str]:
        queries = defaultdict(str)       
        classification = (pattern_classification or DEFAULT_DILI_CLASSIFICATION).strip()
        r_part = f"R={r_score:.2f}" if r_score is not None else UNKNOWN_R_SCORE_TOKEN
        clinical = clinical_context.strip() or NO_CLINICAL_CONTEXT_FALLBACK
        for name in self.drug_names:           
            queries[name] = DILI_RAG_QUERY_PROMPT.format(
                name=name,
                classification=classification,
                r_part=r_part,             
                clinical=clinical,
            )
          
        return queries  


    
       
