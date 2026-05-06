from __future__ import annotations

from services.clinical.drug_blocks import isolate_drug_blocks


def test_bullet_list_blocks() -> None:
    text = "- Esomeprazolo 20 mg\n- Boswellia serrata 1 cps"
    blocks = isolate_drug_blocks(text)
    assert len(blocks) == 2


def test_wrapped_bullet_continuation_attached() -> None:
    text = "- Esomeprazolo 20 mg\n  al mattino\n- Bromelina 1 cps"
    blocks = isolate_drug_blocks(text)
    assert "al mattino" in blocks[0].text


def test_free_prose_returns_single_block() -> None:
    text = "Paziente in terapia cronica senza dettagli posologici specifici."
    blocks = isolate_drug_blocks(text)
    assert len(blocks) == 1
