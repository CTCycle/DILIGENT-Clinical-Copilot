from services.clinical.preparation import ClinicalKnowledgePreparation

###############################################################################
def test_livertox_match_audit_flags_missing_ambiguous_and_low_confidence() -> None:
    preparation = ClinicalKnowledgePreparation()
    issues = preparation.build_match_audit_issues(
        {
            "drug-a": {
                "raw_mentions": ["Drug A"],
                "missing_livertox": True,
                "match_status": "missing_match",
                "rxnav_validated": False,
                "rxnav_rxcui": None,
            },
            "drug-b": {
                "raw_mentions": ["Drug B"],
                "ambiguous_match": True,
                "match_confidence": 0.4,
                "rxnav_validated": True,
                "rxnav_rxcui": "123",
            },
        }
    )

    codes = {issue.code for issue in issues}
    assert "livertox_match_missing" in codes
    assert "livertox_match_ambiguous" in codes
    assert "livertox_match_low_confidence" in codes
    assert "rxnav_alias_not_validated" in codes
