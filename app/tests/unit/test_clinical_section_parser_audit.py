from services.session.text_section_parser import (
    build_section_extraction_from_initial_text,
    parse_initial_text_sections,
)


###############################################################################
def test_section_parser_preserves_exact_body_slice_and_metadata() -> None:
    source = (
        "# Anamnesis\n"
        "Patient reports fatigue.\n\n"
        "# Therapy\n"
        "- Amoxicillin 500 mg bid\n\n"
        "# Laboratory history\n"
        "ALT 450 U/L\n"
    )
    parsed = parse_initial_text_sections(source)
    extraction = build_section_extraction_from_initial_text(parsed, source)
    therapy_meta = extraction.metadata["sections"]["drugs"]

    assert (
        extraction.drugs
        == source[therapy_meta["char_span"][0] : therapy_meta["char_span"][1]]
    )
    assert therapy_meta["canonical_key"] == "therapy"
    assert therapy_meta["body_line_span"][0] >= therapy_meta["heading_line_span"][1]
    assert therapy_meta["verbatim_coherent"] is True
    assert "source_hash" in extraction.metadata
