from domain.clinical.entities import DiseaseContextEntry, DrugEntry
from services.clinical.disease import DiseaseExtractor
from services.clinical.parser import DrugsParser

###############################################################################
def test_drug_llm_post_processing_downgrades_ungrounded_evidence() -> None:
    parser = DrugsParser()
    result = parser.post_process_llm_entry(
        DrugEntry(name="ImaginaryDrug", evidence="not in source"),
        raw_line="Patient denies medication use.",
        source="anamnesis",
        historical_flag=True,
    )

    assert result is not None
    assert result.confidence == "low"
    assert result.attribution == "unclear"

###############################################################################
def test_disease_evidence_validation_sets_span_and_attribution() -> None:
    extractor = DiseaseExtractor()
    entry = DiseaseContextEntry(name="diabetes", evidence="diabetes mellitus")

    result = extractor.validate_entry_evidence(
        entry,
        "History of diabetes mellitus in the patient.",
    )

    assert result.source_span == [11, 28]
    assert result.confidence == "high"
    assert result.attribution == "patient"
