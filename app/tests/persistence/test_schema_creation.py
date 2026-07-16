from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.engine import Engine

###############################################################################
def test_canonical_persistence_tables_exist(persistence_engine: Engine) -> None:
    tables = set(inspect(persistence_engine).get_table_names())
    assert {
        "clinical_sessions",
        "clinical_session_versions",
        "clinical_lab_observations",
        "clinical_drug_mentions",
        "drug_identifiers",
        "application_configuration",
        "reference_catalog_manifests",
    }.issubset(tables)

###############################################################################
def test_canonical_indexes_are_present(persistence_engine: Engine) -> None:
    indexes = {
        index["name"]
        for table in ("clinical_lab_observations", "clinical_drug_mentions")
        for index in inspect(persistence_engine).get_indexes(table)
    }
    assert "ix_clinical_lab_observations_session_marker" in indexes
    assert "ix_clinical_drug_mentions_normalized_name" in indexes
