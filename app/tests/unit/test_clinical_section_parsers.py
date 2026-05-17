from __future__ import annotations

from services.session.clinical_section_parsers import (
    extract_sections_from_markers,
    find_section_markers,
    validate_sections_against_source,
)


def _extract(text: str):
    return extract_sections_from_markers(text, find_section_markers(text))


def test_markdown_headings_parse() -> None:
    sections = _extract("# Anamnesis\nA\n## Therapy\nT\n### Laboratory Analysis\nL")
    assert sections is not None
    assert sections["anamnesis"] == "A"


def test_numbered_headings_parse() -> None:
    sections = _extract("1. Anamnesis\nA\n2) Therapy\nT\n3: Lab analysis\nL")
    assert sections is not None


def test_roman_headings_parse() -> None:
    sections = _extract("I. Anamnesis\nA\nII. Drugs\nT\nIII. Labs\nL")
    assert sections is not None


def test_plain_labeled_parse() -> None:
    sections = _extract("Anamnesis:\nA\nCurrent Drugs:\nT\nLaboratory Analysis:\nL")
    assert sections is not None


def test_realistic_report_aliases_parse() -> None:
    sections = _extract(
        "Anamnesis:\nA\nDrug exposure information:\nT\nLaboratory data:\nL"
    )
    assert sections is not None
    assert sections["drugs"] == "T"
    assert sections["laboratory_analysis"] == "L"


def test_plain_multilingual_titles_parse() -> None:
    sections = _extract("Anamnesi rilevante\nA\nTerapia farmacologica\nT\nLaboratorio\nL")
    assert sections is not None


def test_emphasized_titles_parse() -> None:
    sections = _extract("**Anamnesis**\nA\n**Laboratory analysis**\nL\n**Drugs**\nT")
    assert sections is not None


def test_near_typo_titles_parse() -> None:
    sections = _extract("Anamnesiss\nA\nLaboratory analyses\nL\nDrugz\nT")
    assert sections is not None


def test_unrelated_titles_fail() -> None:
    sections = _extract("Overview\nA\nMeasurements bucket\nL\nTreatment basket\nT")
    assert sections is None


def test_missing_sections_fail() -> None:
    sections = _extract("Anamnesis:\nA\nDrugs:\nT")
    assert sections is None


def test_no_heading_cue_based_parse() -> None:
    text = (
        "Paziente di 83 anni con storia clinica complessa e ricovero recente.\n\n"
        "Labor 31.01.2024: ALAT 255 U/L, ASAT 427 U/L, ALP 339 U/L, GGT 779 U/L.\n\n"
        "Liquemin 5000-0-0-5000 per os dal 31.01.2025, Candesartan 16 mg 2-0-0-0."
    )
    sections = _extract(text)
    assert sections is not None
    assert "ALAT" in sections["laboratory_analysis"]
    assert "Liquemin" in sections["drugs"]


def test_missing_labs_title_cue_based_parse() -> None:
    text = (
        "Anamnesis\n"
        "Paziente di 83 anni noto per adenocarcinoma prostatico e comorbidità.\n\n"
        "Labor 31.01.2024: ALAT 255 U/L, ASAT 427 U/L, ALP 339 U/L.\n\n"
        "Drugs\n"
        "Liquemin 5000-0-0-5000 per os; Candesartan 16 mg 2-0-0-0."
    )
    sections = _extract(text)
    assert sections is not None
    assert "ALAT" in sections["laboratory_analysis"]


def test_source_validation_rejects_fabrication() -> None:
    source = "Anamnesis: A\nTherapy: T\nLab analysis: L"
    assert validate_sections_against_source(
        source,
        {"anamnesis": "A", "drugs": "invented", "laboratory_analysis": "L"},
    ) is False


def test_source_validation_accepts_canonical_ocr_variants() -> None:
    source = (
        "Terapia farmacologica\n"
        "Mycostatin 100'000 IU/ml sosp orale 24ml 1-1-1-0 per os\n"
        "Anamnesi\n"
        "EpatopatiamistaG4 incorsodi accertamenti\n"
        "Laboratorio\n"
        "20.09 ALAT 902 U/L, ASAT 644 U/L, GGT 928 U/L, ALP 264 U/L"
    )
    assert validate_sections_against_source(
        source,
        {
            "anamnesis": "EpatopatiamistaG4 incorsodi accertamenti",
            "drugs": "Mycostatin 100000 IU ml sosp orale 24ml 1 1 1 0 per os",
            "laboratory_analysis": "20.09 ALAT 902 U L ASAT 644 U L GGT 928 U L ALP 264 U L",
        },
    )


def test_titles_with_identifiers_still_parse() -> None:
    text = (
        "Anamnesi paziente 8309062\n"
        "A\n\n"
        "Terapia farmacologica\n"
        "T\n\n"
        "Laboratorio\n"
        "L"
    )
    sections = _extract(text)
    assert sections is not None
    assert sections["anamnesis"] == "A"


def test_long_descriptive_titles_parse() -> None:
    text = (
        "Anamnesi della paziente con indicazioni cliniche iniziali\n"
        "A\n\n"
        "Terapia farmacologica in corso e modifiche recenti\n"
        "T\n\n"
        "Laboratorio con trend e date\n"
        "L"
    )
    sections = _extract(text)
    assert sections is not None


def test_inline_single_line_sections_parse() -> None:
    text = "Anamnesi: A\nTerapia farmacologica: T\nLaboratorio: L"
    sections = _extract(text)
    assert sections is not None
    assert sections["anamnesis"] == "A"
    assert sections["drugs"] == "T"
    assert sections["laboratory_analysis"] == "L"


def test_subheading_with_generic_terapia_is_not_reclassified_as_main_drugs_title() -> None:
    text = (
        "Terapia farmacologica\n"
        "D1\n\n"
        "Anamnesi\n"
        "A1\n"
        "Terapia Oncologica specialistica:\n"
        "Dettaglio oncologico interno alla storia clinica.\n\n"
        "Laboratorio\n"
        "L1"
    )
    sections = _extract(text)
    assert sections is not None
    assert "Dettaglio oncologico" in sections["anamnesis"]
