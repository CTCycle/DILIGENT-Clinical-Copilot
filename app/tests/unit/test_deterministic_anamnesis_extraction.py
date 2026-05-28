from __future__ import annotations

from services.clinical.deterministic_extraction import extract_deterministic_diseases
from services.clinical.parser import DrugsParser


def test_deterministic_anamnesis_regimen_extraction_captures_oncology_history() -> None:
    parser = DrugsParser(client=object())
    text = (
        "Dal 17.03.2023 al 30.06.2023 Carboplatino e Paclitaxel, con aggiunta di Bevacizumab dal secondo ciclo.\n"
        "Dal 28.12.2023 al 17.05.2024: Chemioterapia di seconda linea con Carboplatino e Caelyx, eseguiti 6 cicli.\n"
        "Dal 06.07.2024 al 16.07.2024: Terapia con Olaparib, sospeso per PD in sede peritoneale.\n"
        "Dal 10.01.2025 Protocollo con Gemcitabina + Bevacizumab, ultima somministrazione il 27.02."
    )

    result = parser.extract_drugs_from_anamnesis_deterministic(text)
    names = [entry.name for entry in result.entries]

    assert "Carboplatino" in names
    assert "Paclitaxel" in names
    assert "Bevacizumab" in names
    assert "Caelyx" in names
    assert "Olaparib" in names
    assert "Gemcitabina" in names
    assert result.regimen_lines


def test_deterministic_disease_extraction_captures_hepatic_and_oncologic_context() -> None:
    text = (
        "High grade ovarian serous carcinoma con carcinosi peritoneale.\n"
        "Steatosi epatica cronica documentata.\n"
        "Sospetta polmonite recente.\n"
    )

    result = extract_deterministic_diseases(text)
    names = [entry.name for entry in result.context.entries]

    assert any("carcinoma" in name.lower() for name in names)
    assert "Carcinosi peritoneale" in names
    assert "Steatosi epatica" in names
    assert "Polmonite" in names
