from __future__ import annotations

from collections import defaultdict
from typing import Any

from DILIGENT.app.api.schemas.clinical import PatientDrugs


###############################################################################
class DILIQueryBuilder:

    def __init__(self, drugs : PatientDrugs) -> None: 
        self.drug_names = [x.name for x in drugs.entries if x.name]
        self.dili_query_template = (
            "{name} drug induced liver injury (DILI) {classification} pattern "
            "Pattern of hepatotoxicity - {r_part} "
            "Focus: latency, pattern match vs observed pattern, severity, risk factors, "
            "case reports, rechallenge outcomes, likelihood grading, and management. "
            "Summarize evidence, contradictions, and strength of association. "
            "Clinical context: {clinical}"
        )  

    # -------------------------------------------------------------------------
    def build_dili_queries(
        self,
        *,        
        clinical_context: str,
        pattern_classification: str | None,
        r_score: float | None,      
    ) -> dict[str, str]:
        queries = defaultdict(str)       
        classification = (pattern_classification or "indeterminate").strip()
        r_part = f"R={r_score:.2f}" if r_score is not None else "R=NA"  
        for name in self.drug_names:           
            clinical = clinical_context.strip() or "No additional clinical context provided."
            queries[name] = self.dili_query_template.format(
                name=name,
                classification=classification,
                r_part=r_part,             
                clinical=clinical,
            )
          
        return queries  


    
       