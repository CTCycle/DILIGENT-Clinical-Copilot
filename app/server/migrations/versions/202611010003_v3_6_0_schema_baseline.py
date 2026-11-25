# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""v3.6.0 schema baseline.

The complete schema of the v3.6.0 release. Databases created by v3.6.0 are
already stamped at this revision; older databases are not supported.

Revision ID: 202611010003
Revises:
Create Date: 2026-11-01
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "202611010003"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


###############################################################################
def upgrade() -> None:
    op.create_table(
        "access_keys",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("encrypted_value", sa.Text(), nullable=False),
        sa.Column("encryption_key_version", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(), nullable=False),
        sa.Column(
            "is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("last_used_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "provider IN ('openai', 'gemini', 'deepseek', 'anthropic', 'opencode', 'brave')",
            name=op.f("ck_access_keys_ck_access_keys_provider"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_access_keys")),
    )
    with op.batch_alter_table("access_keys", schema=None) as batch_op:
        batch_op.create_index("ix_access_keys_provider", ["provider"], unique=False)
        batch_op.create_index(
            "uq_access_keys_active_provider",
            ["provider"],
            unique=True,
            sqlite_where=sa.text("is_active = 1"),
            postgresql_where=sa.text("is_active = true"),
        )
    op.create_table(
        "application_configuration",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "revision", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_application_configuration")),
    )
    op.create_table(
        "drugs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column("canonical_name_norm", sa.String(), nullable=False),
        sa.Column("livertox_nbk_id", sa.String(), nullable=True),
        sa.Column("rxnav_last_update", sa.String(), nullable=True),
        sa.Column(
            "concept_type",
            sa.String(),
            server_default=sa.text("'ingredient'"),
            nullable=False,
        ),
        sa.Column("preferred_inn", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "concept_type IN ('ingredient', 'precise_ingredient', 'multi_ingredient')",
            name=op.f("ck_drugs_concept_type"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drugs")),
        sa.UniqueConstraint("canonical_name_norm", name="uq_drugs_canonical_name_norm"),
    )
    with op.batch_alter_table("drugs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_drugs_livertox_nbk_id", ["livertox_nbk_id"], unique=False
        )
    op.create_table(
        "provider_model_catalog_cache",
        sa.Column("provider_id", sa.String(length=64), nullable=False),
        sa.Column("configuration_fingerprint", sa.String(length=128), nullable=False),
        sa.Column("models", sa.JSON(), nullable=False),
        sa.Column("last_success_at", sa.DateTime(), nullable=True),
        sa.Column(
            "last_attempt_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("last_attempt_status", sa.String(length=32), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "provider_id", name=op.f("pk_provider_model_catalog_cache")
        ),
    )
    op.create_table(
        "reference_catalog_entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("manifest", sa.String(), nullable=False),
        sa.Column("manifest_version", sa.Integer(), nullable=False),
        sa.Column("domain", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column(
            "locale", sa.String(), server_default=sa.text("'und'"), nullable=False
        ),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("normalized_value", sa.String(), nullable=False),
        sa.Column(
            "priority", sa.Integer(), server_default=sa.text("100"), nullable=False
        ),
        sa.Column(
            "match_mode", sa.String(), server_default=sa.text("'token'"), nullable=False
        ),
        sa.Column(
            "case_sensitive",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column(
            "active", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reference_catalog_entries")),
        sa.UniqueConstraint(
            "manifest",
            "domain",
            "category",
            "key",
            "locale",
            "normalized_value",
            name="uq_reference_catalog_entries_identity",
        ),
    )
    with op.batch_alter_table("reference_catalog_entries", schema=None) as batch_op:
        batch_op.create_index(
            "ix_reference_catalog_entries_active", ["active"], unique=False
        )
        batch_op.create_index(
            "ix_reference_catalog_entries_lookup",
            ["domain", "category", "key", "locale"],
            unique=False,
        )
        batch_op.create_index(
            "ix_reference_catalog_entries_manifest", ["manifest"], unique=False
        )
    op.create_table(
        "reference_catalog_manifests",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("manifest", sa.String(), nullable=False),
        sa.Column("installed_version", sa.Integer(), nullable=False),
        sa.Column("manifest_hash", sa.String(), nullable=False),
        sa.Column("source_path", sa.Text(), nullable=False),
        sa.Column("entry_count", sa.Integer(), nullable=False),
        sa.Column(
            "installed_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reference_catalog_manifests")),
        sa.UniqueConstraint("manifest", name="uq_reference_catalog_manifests_manifest"),
    )
    with op.batch_alter_table("reference_catalog_manifests", schema=None) as batch_op:
        batch_op.create_index(
            "ix_reference_catalog_manifests_hash", ["manifest_hash"], unique=False
        )
    op.create_table(
        "terminology_import_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("source_version", sa.String(), nullable=True),
        sa.Column("license", sa.Text(), nullable=True),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("stats_json", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "mode IN ('full', 'incremental')",
            name=op.f("ck_terminology_import_runs_mode"),
        ),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'cancelled', 'skipped')",
            name=op.f("ck_terminology_import_runs_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_terminology_import_runs")),
    )
    with op.batch_alter_table("terminology_import_runs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_terminology_import_runs_source_status",
            ["source", "status", "finished_at"],
            unique=False,
        )
    op.create_table(
        "dilirank_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=True),
        sa.Column("ltkb_id", sa.String(), nullable=False),
        sa.Column("compound_name", sa.Text(), nullable=False),
        sa.Column("compound_name_norm", sa.String(), nullable=False),
        sa.Column("severity_class", sa.Integer(), nullable=True),
        sa.Column("label_section", sa.Text(), nullable=True),
        sa.Column("dili_concern", sa.String(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("source_last_modified", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            name=op.f("fk_dilirank_records_drug_id_drugs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_dilirank_records")),
        sa.UniqueConstraint("ltkb_id", name="uq_dilirank_records_ltkb_id"),
    )
    with op.batch_alter_table("dilirank_records", schema=None) as batch_op:
        batch_op.create_index(
            "ix_dilirank_records_compound_name_norm",
            ["compound_name_norm"],
            unique=False,
        )
        batch_op.create_index(
            "ix_dilirank_records_dili_concern", ["dili_concern"], unique=False
        )
        batch_op.create_index("ix_dilirank_records_drug_id", ["drug_id"], unique=False)
    op.create_table(
        "drug_aliases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=False),
        sa.Column("alias", sa.Text(), nullable=False),
        sa.Column("alias_norm", sa.String(), nullable=False),
        sa.Column("alias_kind", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("term_type", sa.String(), nullable=True),
        sa.Column("language", sa.String(), nullable=True),
        sa.Column("country", sa.String(), nullable=True),
        sa.Column(
            "validation_status",
            sa.String(),
            server_default=sa.text("'verified'"),
            nullable=False,
        ),
        sa.Column("import_run_id", sa.Integer(), nullable=True),
        sa.Column("source_record_id", sa.String(), nullable=True),
        sa.CheckConstraint(
            "validation_status IN ('verified', 'confirmed', 'provisional', 'rejected')",
            name=op.f("ck_drug_aliases_validation_status"),
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"], ["drugs.id"], name=op.f("fk_drug_aliases_drug_id_drugs")
        ),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            ["terminology_import_runs.id"],
            name=op.f("fk_drug_aliases_import_run_id_terminology_import_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drug_aliases")),
        sa.UniqueConstraint(
            "drug_id",
            "alias_norm",
            "alias_kind",
            "source",
            name="uq_drug_aliases_identity",
        ),
    )
    with op.batch_alter_table("drug_aliases", schema=None) as batch_op:
        batch_op.create_index(
            "ix_drug_aliases_alias_norm_source", ["alias_norm", "source"], unique=False
        )
        batch_op.create_index("ix_drug_aliases_drug_id", ["drug_id"], unique=False)
    op.create_table(
        "drug_atc_codes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=False),
        sa.Column("atc_code", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("import_run_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            name=op.f("fk_drug_atc_codes_drug_id_drugs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            ["terminology_import_runs.id"],
            name=op.f("fk_drug_atc_codes_import_run_id_terminology_import_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drug_atc_codes")),
        sa.UniqueConstraint(
            "drug_id", "atc_code", "source", name="uq_drug_atc_codes_identity"
        ),
    )
    with op.batch_alter_table("drug_atc_codes", schema=None) as batch_op:
        batch_op.create_index("ix_drug_atc_codes_atc_code", ["atc_code"], unique=False)
    op.create_table(
        "drug_concept_relations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=False),
        sa.Column("related_drug_id", sa.Integer(), nullable=False),
        sa.Column("relation_type", sa.String(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("import_run_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "relation_type IN ('form_of', 'has_component')",
            name=op.f("ck_drug_concept_relations_relation_type"),
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            name=op.f("fk_drug_concept_relations_drug_id_drugs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            ["terminology_import_runs.id"],
            name=op.f(
                "fk_drug_concept_relations_import_run_id_terminology_import_runs"
            ),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["related_drug_id"],
            ["drugs.id"],
            name=op.f("fk_drug_concept_relations_related_drug_id_drugs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drug_concept_relations")),
        sa.UniqueConstraint(
            "drug_id",
            "related_drug_id",
            "relation_type",
            "source",
            name="uq_drug_concept_relations_identity",
        ),
    )
    with op.batch_alter_table("drug_concept_relations", schema=None) as batch_op:
        batch_op.create_index(
            "ix_drug_concept_relations_drug_id", ["drug_id"], unique=False
        )
        batch_op.create_index(
            "ix_drug_concept_relations_related_drug_id",
            ["related_drug_id"],
            unique=False,
        )
    op.create_table(
        "drug_identifiers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=False),
        sa.Column("identifier_system", sa.String(), nullable=False),
        sa.Column("identifier_value", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("import_run_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            name=op.f("fk_drug_identifiers_drug_id_drugs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            ["terminology_import_runs.id"],
            name=op.f("fk_drug_identifiers_import_run_id_terminology_import_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drug_identifiers")),
        sa.UniqueConstraint(
            "identifier_system",
            "identifier_value",
            name="uq_drug_identifiers_system_value",
        ),
    )
    with op.batch_alter_table("drug_identifiers", schema=None) as batch_op:
        batch_op.create_index("ix_drug_identifiers_drug_id", ["drug_id"], unique=False)
    op.create_table(
        "drug_rxnorm_codes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=False),
        sa.Column("rxcui", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["drug_id"], ["drugs.id"], name=op.f("fk_drug_rxnorm_codes_drug_id_drugs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drug_rxnorm_codes")),
        sa.UniqueConstraint("drug_id", "rxcui", name="uq_drug_rxnorm_codes_identity"),
        sa.UniqueConstraint("rxcui", name="uq_drug_rxnorm_codes_rxcui"),
    )
    with op.batch_alter_table("drug_rxnorm_codes", schema=None) as batch_op:
        batch_op.create_index("ix_drug_rxnorm_codes_drug_id", ["drug_id"], unique=False)
    op.create_table(
        "kb_match_cache",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("raw_drug_name", sa.Text(), nullable=False),
        sa.Column("raw_drug_name_norm", sa.String(), nullable=False),
        sa.Column("normalized_drug_key", sa.String(), nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=True),
        sa.Column("rxnorm_rxcui", sa.String(), nullable=True),
        sa.Column("livertox_monograph_key", sa.String(), nullable=True),
        sa.Column("livertox_nbk_id", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("evidence_json", sa.Text(), nullable=True),
        sa.Column("deterministic_evidence_version", sa.String(), nullable=True),
        sa.Column("invalidated_at", sa.DateTime(), nullable=True),
        sa.Column("invalidation_reason", sa.Text(), nullable=True),
        sa.Column("resolution_status", sa.String(), nullable=True),
        sa.Column("method", sa.String(), nullable=True),
        sa.Column("terminology_revision", sa.Integer(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source IN ('rxnav', 'livertox', 'rag', 'terminology')",
            name=op.f("ck_kb_match_cache_ck_kb_match_cache_source"),
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name=op.f("ck_kb_match_cache_ck_kb_match_cache_confidence"),
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"], ["drugs.id"], name=op.f("fk_kb_match_cache_drug_id_drugs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_kb_match_cache")),
        sa.UniqueConstraint(
            "normalized_drug_key", "source", name="uq_kb_match_cache_key_source"
        ),
    )
    with op.batch_alter_table("kb_match_cache", schema=None) as batch_op:
        batch_op.create_index("ix_kb_match_cache_drug_id", ["drug_id"], unique=False)
        batch_op.create_index(
            "ix_kb_match_cache_normalized_source",
            ["normalized_drug_key", "source"],
            unique=False,
        )
        batch_op.create_index(
            "ix_kb_match_cache_raw_drug_name_norm", ["raw_drug_name_norm"], unique=False
        )
        batch_op.create_index(
            "ix_kb_match_cache_valid", ["invalidated_at"], unique=False
        )
    op.create_table(
        "livertox_monographs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=False),
        sa.Column("monograph_key", sa.String(), nullable=False),
        sa.Column("drug_name_norm", sa.String(), nullable=False),
        sa.Column("nbk_id", sa.String(), nullable=True),
        sa.Column("excerpt", sa.Text(), nullable=True),
        sa.Column("likelihood_score", sa.String(), nullable=True),
        sa.Column("last_update", sa.String(), nullable=True),
        sa.Column("reference_count", sa.Integer(), nullable=True),
        sa.Column("year_approved", sa.Integer(), nullable=True),
        sa.Column("agent_classification", sa.String(), nullable=True),
        sa.Column("primary_classification", sa.String(), nullable=True),
        sa.Column("secondary_classification", sa.String(), nullable=True),
        sa.Column("include_in_livertox", sa.Boolean(), nullable=True),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("source_last_modified", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["drug_id"], ["drugs.id"], name=op.f("fk_livertox_monographs_drug_id_drugs")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_livertox_monographs")),
        sa.UniqueConstraint(
            "monograph_key", name="uq_livertox_monographs_monograph_key"
        ),
    )
    with op.batch_alter_table("livertox_monographs", schema=None) as batch_op:
        batch_op.create_index(
            "ix_livertox_monographs_drug_id", ["drug_id"], unique=False
        )
        batch_op.create_index(
            "ix_livertox_monographs_drug_name_norm", ["drug_name_norm"], unique=False
        )
        batch_op.create_index("ix_livertox_monographs_nbk_id", ["nbk_id"], unique=False)
    op.create_table(
        "medicinal_products",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("name_norm", sa.String(), nullable=False),
        sa.Column("brand_name_norm", sa.String(), nullable=True),
        sa.Column("country", sa.String(), nullable=True),
        sa.Column("dose_form", sa.String(), nullable=True),
        sa.Column("route", sa.String(), nullable=True),
        sa.Column("composition_status", sa.String(), nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("source_product_id", sa.String(), nullable=False),
        sa.Column("import_run_id", sa.Integer(), nullable=True),
        sa.CheckConstraint(
            "composition_status IN ('verified', 'unverified')",
            name=op.f("ck_medicinal_products_composition_status"),
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            name=op.f("fk_medicinal_products_drug_id_drugs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            ["terminology_import_runs.id"],
            name=op.f("fk_medicinal_products_import_run_id_terminology_import_runs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_medicinal_products")),
        sa.UniqueConstraint(
            "source", "source_product_id", name="uq_medicinal_products_source_record"
        ),
    )
    with op.batch_alter_table("medicinal_products", schema=None) as batch_op:
        batch_op.create_index(
            "ix_medicinal_products_brand_name_norm", ["brand_name_norm"], unique=False
        )
        batch_op.create_index(
            "ix_medicinal_products_drug_id", ["drug_id"], unique=False
        )
        batch_op.create_index(
            "ix_medicinal_products_name_norm", ["name_norm"], unique=False
        )
    op.create_table(
        "terminology_conflicts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("import_run_id", sa.Integer(), nullable=True),
        sa.Column("conflict_type", sa.String(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("existing_drug_id", sa.Integer(), nullable=True),
        sa.Column("proposed_drug_id", sa.Integer(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column(
            "status", sa.String(), server_default=sa.text("'open'"), nullable=False
        ),
        sa.Column(
            "detected_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "conflict_type IN ('identifier_mismatch', 'name_identity_mismatch', 'composition_mismatch')",
            name=op.f("ck_terminology_conflicts_conflict_type"),
        ),
        sa.CheckConstraint(
            "status IN ('open', 'resolved', 'dismissed')",
            name=op.f("ck_terminology_conflicts_status"),
        ),
        sa.ForeignKeyConstraint(
            ["existing_drug_id"],
            ["drugs.id"],
            name=op.f("fk_terminology_conflicts_existing_drug_id_drugs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["import_run_id"],
            ["terminology_import_runs.id"],
            name=op.f("fk_terminology_conflicts_import_run_id_terminology_import_runs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["proposed_drug_id"],
            ["drugs.id"],
            name=op.f("fk_terminology_conflicts_proposed_drug_id_drugs"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_terminology_conflicts")),
    )
    with op.batch_alter_table("terminology_conflicts", schema=None) as batch_op:
        batch_op.create_index(
            "ix_terminology_conflicts_source", ["source"], unique=False
        )
        batch_op.create_index(
            "ix_terminology_conflicts_status", ["status"], unique=False
        )
    op.create_table(
        "clinical_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("patient_name", sa.String(), nullable=True),
        sa.Column("visit_date", sa.Date(), nullable=True),
        sa.Column("anamnesis", sa.Text(), nullable=True),
        sa.Column("drugs_text", sa.Text(), nullable=True),
        sa.Column("laboratory_analysis", sa.Text(), nullable=True),
        sa.Column("patient_image_blob", sa.LargeBinary(), nullable=True),
        sa.Column("session_timestamp", sa.DateTime(), nullable=True),
        sa.Column("hepatic_pattern", sa.String(), nullable=True),
        sa.Column("text_extraction_model", sa.String(), nullable=True),
        sa.Column("clinical_model", sa.String(), nullable=True),
        sa.Column("total_duration", sa.Float(), nullable=True),
        sa.Column(
            "session_status",
            sa.String(),
            server_default=sa.text("'draft'"),
            nullable=False,
        ),
        sa.Column("session_kind", sa.String(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("clinical_input", sa.Text(), nullable=True),
        sa.Column("job_id", sa.String(length=36), nullable=True),
        sa.Column("run_request_json", sa.Text(), nullable=True),
        sa.Column("progress_stage", sa.String(), nullable=True),
        sa.Column("progress_percent", sa.Float(), nullable=True),
        sa.Column("progress_message", sa.Text(), nullable=True),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column(
            "state_version", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("queued_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "session_status IN ('draft', 'queued', 'running', 'completed', 'cancelled', 'failed')",
            name=op.f("ck_clinical_sessions_ck_clinical_sessions_lifecycle_status"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_sessions")),
    )
    with op.batch_alter_table("clinical_sessions", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clinical_sessions_patient_name", ["patient_name"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_sessions_status", ["session_status"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_sessions_status_queued_at",
            ["session_status", "queued_at"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_sessions_status_timestamp_id",
            ["session_status", "session_timestamp", "id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_sessions_timestamp", ["session_timestamp"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_sessions_timestamp_id",
            ["session_timestamp", "id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_sessions_updated_at_id", ["updated_at", "id"], unique=False
        )
    op.create_table(
        "drug_livertox_links",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=False),
        sa.Column("monograph_id", sa.Integer(), nullable=False),
        sa.Column("link_type", sa.String(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "link_type IN ('monograph_owner', 'verified_alias', 'component_set', 'form_of_parent', 'form_of_child')",
            name=op.f("ck_drug_livertox_links_link_type"),
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            name=op.f("fk_drug_livertox_links_drug_id_drugs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["monograph_id"],
            ["livertox_monographs.id"],
            name=op.f("fk_drug_livertox_links_monograph_id_livertox_monographs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drug_livertox_links")),
        sa.UniqueConstraint(
            "drug_id", "monograph_id", name="uq_drug_livertox_links_identity"
        ),
    )
    with op.batch_alter_table("drug_livertox_links", schema=None) as batch_op:
        batch_op.create_index(
            "ix_drug_livertox_links_monograph_id", ["monograph_id"], unique=False
        )
    op.create_table(
        "medicinal_product_ingredients",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("product_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=True),
        sa.Column("ingredient_name", sa.Text(), nullable=False),
        sa.Column("strength_value", sa.Float(), nullable=True),
        sa.Column("strength_unit", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            name=op.f("fk_medicinal_product_ingredients_drug_id_drugs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["product_id"],
            ["medicinal_products.id"],
            name=op.f("fk_medicinal_product_ingredients_product_id_medicinal_products"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_medicinal_product_ingredients")),
        sa.UniqueConstraint(
            "product_id", "position", name="uq_medicinal_product_ingredients_position"
        ),
    )
    with op.batch_alter_table("medicinal_product_ingredients", schema=None) as batch_op:
        batch_op.create_index(
            "ix_medicinal_product_ingredients_drug_id", ["drug_id"], unique=False
        )
    op.create_table(
        "clinical_drug_mentions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("mention_ordinal", sa.Integer(), nullable=False),
        sa.Column("raw_name", sa.Text(), nullable=False),
        sa.Column("normalized_name", sa.String(), nullable=False),
        sa.Column("drug_id", sa.Integer(), nullable=True),
        sa.Column("match_status", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("match_reason", sa.String(), nullable=True),
        sa.Column("evidence_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["drug_id"],
            ["drugs.id"],
            name=op.f("fk_clinical_drug_mentions_drug_id_drugs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_drug_mentions_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_drug_mentions")),
        sa.UniqueConstraint(
            "session_id",
            "mention_ordinal",
            name="uq_clinical_drug_mentions_session_ordinal",
        ),
    )
    with op.batch_alter_table("clinical_drug_mentions", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clinical_drug_mentions_normalized_name",
            ["normalized_name"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_drug_mentions_session_id", ["session_id"], unique=False
        )
    op.create_table(
        "clinical_lab_observations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("marker_code", sa.String(), nullable=False),
        sa.Column("observation_at", sa.DateTime(), nullable=True),
        sa.Column("value_numeric", sa.Float(), nullable=True),
        sa.Column("value_text", sa.String(), nullable=True),
        sa.Column("unit", sa.String(), nullable=True),
        sa.Column("upper_limit_numeric", sa.Float(), nullable=True),
        sa.Column("source_ordinal", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_lab_observations_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_lab_observations")),
    )
    with op.batch_alter_table("clinical_lab_observations", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clinical_lab_observations_observation_at",
            ["observation_at"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_lab_observations_session_marker",
            ["session_id", "marker_code"],
            unique=False,
        )
    op.create_table(
        "clinical_session_results",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_session_results_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_session_results")),
        sa.UniqueConstraint(
            "session_id", name="uq_clinical_session_results_session_id"
        ),
    )
    with op.batch_alter_table("clinical_session_results", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_results_session_id", ["session_id"], unique=False
        )
    op.create_table(
        "clinical_session_sections",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("section_kind", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_session_sections_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_session_sections")),
        sa.UniqueConstraint(
            "session_id", "section_kind", name="uq_clinical_session_sections_identity"
        ),
    )
    with op.batch_alter_table("clinical_session_sections", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_sections_session_id", ["session_id"], unique=False
        )
    op.create_table(
        "clinical_session_timelines",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("generated_at", sa.DateTime(), nullable=False),
        sa.Column("generation_status", sa.String(), nullable=False),
        sa.Column("generation_note", sa.Text(), nullable=True),
        sa.Column("source_model", sa.String(), nullable=True),
        sa.Column("source_kind", sa.String(), nullable=True),
        sa.Column("model_provider", sa.String(), nullable=True),
        sa.Column("source_text_hash", sa.String(length=64), nullable=True),
        sa.Column("timeline_payload_json", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "generation_status IN ('llm_generated', 'fallback')",
            name=op.f(
                "ck_clinical_session_timelines_ck_clinical_session_timelines_generation_status"
            ),
        ),
        sa.CheckConstraint(
            "source_kind IS NULL OR source_kind IN ('local', 'cloud')",
            name=op.f(
                "ck_clinical_session_timelines_ck_clinical_session_timelines_source_kind"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_session_timelines_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_session_timelines")),
    )
    with op.batch_alter_table("clinical_session_timelines", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_timelines_generated_at", ["generated_at"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_session_timelines_session_id", ["session_id"], unique=False
        )
    op.create_table(
        "clinical_session_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=True),
        sa.Column("root_session_id", sa.Integer(), nullable=False),
        sa.Column("source_version_id", sa.Integer(), nullable=True),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("version_status", sa.String(), nullable=False),
        sa.Column("revision_kind", sa.String(), nullable=False),
        sa.Column("llm_qa_status", sa.String(), nullable=False),
        sa.Column("clinical_review_status", sa.String(), nullable=False),
        sa.Column("pipeline_run_id", sa.String(), nullable=True),
        sa.Column("model_configuration_json", sa.Text(), nullable=True),
        sa.Column("report_text", sa.Text(), nullable=True),
        sa.Column("hepatic_pattern", sa.String(), nullable=True),
        sa.Column("total_duration", sa.Float(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.CheckConstraint(
            "clinical_review_status IN ('not_reviewed', 'under_review', 'approved_by_human', 'rejected_by_human')",
            name=op.f(
                "ck_clinical_session_versions_ck_clinical_session_versions_clinical_review_status"
            ),
        ),
        sa.CheckConstraint(
            "llm_qa_status IN ('not_run', 'pending', 'passed', 'passed_with_warnings', 'failed', 'requires_human_review')",
            name=op.f(
                "ck_clinical_session_versions_ck_clinical_session_versions_llm_qa_status"
            ),
        ),
        sa.CheckConstraint(
            "revision_kind IN ('original', 'manual_edit', 'llm_assisted_revision')",
            name=op.f(
                "ck_clinical_session_versions_ck_clinical_session_versions_revision_kind"
            ),
        ),
        sa.CheckConstraint(
            "version_status IN ('current', 'superseded', 'draft_revision', 'pending_qa', 'cancelled', 'qa_failed', 'requires_human_review', 'llm_qa_passed', 'human_approved', 'human_rejected')",
            name=op.f(
                "ck_clinical_session_versions_ck_clinical_session_versions_version_status"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["root_session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_session_versions_root_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_session_versions_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_version_id"],
            ["clinical_session_versions.id"],
            name=op.f(
                "fk_clinical_session_versions_source_version_id_clinical_session_versions"
            ),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_session_versions")),
        sa.UniqueConstraint(
            "root_session_id",
            "version_number",
            name="uq_clinical_session_versions_root_version_number",
        ),
        sa.UniqueConstraint(
            "session_id",
            "version_number",
            name="uq_clinical_session_versions_session_version_number",
        ),
    )
    with op.batch_alter_table("clinical_session_versions", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_versions_pipeline_run_id",
            ["pipeline_run_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_versions_root_session_id",
            ["root_session_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_versions_session_id", ["session_id"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_session_versions_source_version_id",
            ["source_version_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_versions_status", ["version_status"], unique=False
        )
    op.create_table(
        "clinical_session_attachments",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("version_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("media_type", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('image', 'json', 'document')",
            name=op.f(
                "ck_clinical_session_attachments_ck_clinical_session_attachments_kind"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_session_attachments_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["clinical_session_versions.id"],
            name=op.f(
                "fk_clinical_session_attachments_version_id_clinical_session_versions"
            ),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_session_attachments")),
        sa.UniqueConstraint(
            "session_id",
            "version_id",
            "sha256",
            name="uq_clinical_session_attachments_session_version_sha256",
        ),
    )
    with op.batch_alter_table("clinical_session_attachments", schema=None) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_attachments_session_id", ["session_id"], unique=False
        )
    op.create_table(
        "clinical_session_revision_reviews",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("revision_version_id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=True),
        sa.Column("clinical_review_status", sa.String(), nullable=False),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column("reviewed_by", sa.String(), nullable=True),
        sa.Column("actor_id", sa.String(), nullable=True),
        sa.Column("actor_display_name", sa.String(), nullable=True),
        sa.Column("actor_source", sa.String(), nullable=False),
        sa.Column("actor_confidence", sa.String(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column(
            "reviewed_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "actor_confidence IN ('verified', 'unverified', 'system')",
            name=op.f(
                "ck_clinical_session_revision_reviews_ck_clinical_session_revision_reviews_actor_confidence"
            ),
        ),
        sa.CheckConstraint(
            "actor_source IN ('authenticated_user', 'local_profile', 'manual_entry', 'system', 'unknown')",
            name=op.f(
                "ck_clinical_session_revision_reviews_ck_clinical_session_revision_reviews_actor_source"
            ),
        ),
        sa.CheckConstraint(
            "clinical_review_status IN ('under_review', 'approved_by_human', 'rejected_by_human')",
            name=op.f(
                "ck_clinical_session_revision_reviews_ck_clinical_session_revision_reviews_clinical_review_status"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["revision_version_id"],
            ["clinical_session_versions.id"],
            name=op.f(
                "fk_clinical_session_revision_reviews_revision_version_id_clinical_session_versions"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f(
                "fk_clinical_session_revision_reviews_session_id_clinical_sessions"
            ),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint(
            "id", name=op.f("pk_clinical_session_revision_reviews")
        ),
    )
    with op.batch_alter_table(
        "clinical_session_revision_reviews", schema=None
    ) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_revision_reviews_reviewed_at",
            ["reviewed_at"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_revision_reviews_revision_version_id",
            ["revision_version_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_revision_reviews_session_id",
            ["session_id"],
            unique=False,
        )
    op.create_table(
        "clinical_session_revision_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pipeline_run_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("root_session_id", sa.Integer(), nullable=False),
        sa.Column("source_version_id", sa.Integer(), nullable=False),
        sa.Column("target_revision_version_id", sa.Integer(), nullable=True),
        sa.Column("revision_mode", sa.String(), nullable=False),
        sa.Column("revision_kind", sa.String(), nullable=False),
        sa.Column("configuration_json", sa.Text(), nullable=True),
        sa.Column("reviewer_note", sa.Text(), nullable=True),
        sa.Column("initiated_by", sa.String(), nullable=True),
        sa.Column("actor_id", sa.String(), nullable=True),
        sa.Column("actor_display_name", sa.String(), nullable=True),
        sa.Column("actor_source", sa.String(), nullable=False),
        sa.Column("actor_confidence", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column("token_usage_json", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("cost_estimate", sa.Float(), nullable=True),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "actor_confidence IN ('verified', 'unverified', 'system')",
            name=op.f(
                "ck_clinical_session_revision_runs_ck_clinical_session_revision_runs_actor_confidence"
            ),
        ),
        sa.CheckConstraint(
            "actor_source IN ('authenticated_user', 'local_profile', 'manual_entry', 'system', 'unknown')",
            name=op.f(
                "ck_clinical_session_revision_runs_ck_clinical_session_revision_runs_actor_source"
            ),
        ),
        sa.CheckConstraint(
            "revision_kind IN ('original', 'manual_edit', 'llm_assisted_revision')",
            name=op.f(
                "ck_clinical_session_revision_runs_ck_clinical_session_revision_runs_revision_kind"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["root_session_id"],
            ["clinical_sessions.id"],
            name=op.f(
                "fk_clinical_session_revision_runs_root_session_id_clinical_sessions"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["clinical_sessions.id"],
            name=op.f("fk_clinical_session_revision_runs_session_id_clinical_sessions"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_version_id"],
            ["clinical_session_versions.id"],
            name=op.f(
                "fk_clinical_session_revision_runs_source_version_id_clinical_session_versions"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["target_revision_version_id"],
            ["clinical_session_versions.id"],
            name=op.f(
                "fk_clinical_session_revision_runs_target_revision_version_id_clinical_session_versions"
            ),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_session_revision_runs")),
        sa.UniqueConstraint(
            "pipeline_run_id", name="uq_clinical_session_revision_runs_pipeline_run_id"
        ),
    )
    with op.batch_alter_table(
        "clinical_session_revision_runs", schema=None
    ) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_revision_runs_root_session_id",
            ["root_session_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_revision_runs_session_id", ["session_id"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_session_revision_runs_source_version_id",
            ["source_version_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_revision_runs_started_at", ["started_at"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_session_revision_runs_status", ["status"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_session_revision_runs_target_revision_version_id",
            ["target_revision_version_id"],
            unique=False,
        )
    op.create_table(
        "clinical_session_revision_artifacts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("revision_run_id", sa.Integer(), nullable=True),
        sa.Column("revision_version_id", sa.Integer(), nullable=False),
        sa.Column("pipeline_run_id", sa.String(), nullable=False),
        sa.Column("artifact_kind", sa.String(), nullable=False),
        sa.Column("artifact_key", sa.String(), nullable=False),
        sa.Column("entity_type", sa.String(), nullable=True),
        sa.Column("entity_name", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("schema_version", sa.String(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "artifact_kind IN ('structured_case_entity', 'llm_qa_output', 'report_comparison', 'pipeline_artifact')",
            name=op.f(
                "ck_clinical_session_revision_artifacts_ck_clinical_session_revision_artifacts_artifact_kind"
            ),
        ),
        sa.ForeignKeyConstraint(
            ["revision_run_id"],
            ["clinical_session_revision_runs.id"],
            name=op.f(
                "fk_clinical_session_revision_artifacts_revision_run_id_clinical_session_revision_runs"
            ),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["revision_version_id"],
            ["clinical_session_versions.id"],
            name=op.f(
                "fk_clinical_session_revision_artifacts_revision_version_id_clinical_session_versions"
            ),
        ),
        sa.PrimaryKeyConstraint(
            "id", name=op.f("pk_clinical_session_revision_artifacts")
        ),
    )
    with op.batch_alter_table(
        "clinical_session_revision_artifacts", schema=None
    ) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_revision_artifacts_artifact_kind",
            ["artifact_kind"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_revision_artifacts_entity_type",
            ["entity_type"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_revision_artifacts_pipeline_run_id",
            ["pipeline_run_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_revision_artifacts_revision_version_id",
            ["revision_version_id"],
            unique=False,
        )
    op.create_table(
        "clinical_session_revision_steps",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("revision_run_id", sa.Integer(), nullable=True),
        sa.Column("pipeline_run_id", sa.String(), nullable=False),
        sa.Column("step_name", sa.String(), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("input_hash", sa.String(), nullable=True),
        sa.Column("output_hash", sa.String(), nullable=True),
        sa.Column("input_summary_json", sa.Text(), nullable=True),
        sa.Column("output_summary_json", sa.Text(), nullable=True),
        sa.Column("output_payload_json", sa.Text(), nullable=True),
        sa.Column("schema_name", sa.String(), nullable=True),
        sa.Column("schema_version", sa.String(), nullable=True),
        sa.Column("prompt_version", sa.String(), nullable=True),
        sa.Column("parser_version", sa.String(), nullable=True),
        sa.Column("model_provider", sa.String(), nullable=True),
        sa.Column("model_name", sa.String(), nullable=True),
        sa.Column("token_usage_json", sa.Text(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "retry_count", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("superseded_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["revision_run_id"],
            ["clinical_session_revision_runs.id"],
            name=op.f(
                "fk_clinical_session_revision_steps_revision_run_id_clinical_session_revision_runs"
            ),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_clinical_session_revision_steps")),
        sa.UniqueConstraint(
            "pipeline_run_id",
            "step_name",
            "attempt_number",
            name="uq_clinical_session_revision_steps_run_step_attempt",
        ),
    )
    with op.batch_alter_table(
        "clinical_session_revision_steps", schema=None
    ) as batch_op:
        batch_op.create_index(
            "ix_clinical_session_revision_steps_pipeline_run_id",
            ["pipeline_run_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_clinical_session_revision_steps_status", ["status"], unique=False
        )
        batch_op.create_index(
            "ix_clinical_session_revision_steps_step_name", ["step_name"], unique=False
        )


###############################################################################
def downgrade() -> None:
    with op.batch_alter_table(
        "clinical_session_revision_steps", schema=None
    ) as batch_op:
        batch_op.drop_index("ix_clinical_session_revision_steps_step_name")
        batch_op.drop_index("ix_clinical_session_revision_steps_status")
        batch_op.drop_index("ix_clinical_session_revision_steps_pipeline_run_id")
    op.drop_table("clinical_session_revision_steps")
    with op.batch_alter_table(
        "clinical_session_revision_artifacts", schema=None
    ) as batch_op:
        batch_op.drop_index(
            "ix_clinical_session_revision_artifacts_revision_version_id"
        )
        batch_op.drop_index("ix_clinical_session_revision_artifacts_pipeline_run_id")
        batch_op.drop_index("ix_clinical_session_revision_artifacts_entity_type")
        batch_op.drop_index("ix_clinical_session_revision_artifacts_artifact_kind")
    op.drop_table("clinical_session_revision_artifacts")
    with op.batch_alter_table(
        "clinical_session_revision_runs", schema=None
    ) as batch_op:
        batch_op.drop_index(
            "ix_clinical_session_revision_runs_target_revision_version_id"
        )
        batch_op.drop_index("ix_clinical_session_revision_runs_status")
        batch_op.drop_index("ix_clinical_session_revision_runs_started_at")
        batch_op.drop_index("ix_clinical_session_revision_runs_source_version_id")
        batch_op.drop_index("ix_clinical_session_revision_runs_session_id")
        batch_op.drop_index("ix_clinical_session_revision_runs_root_session_id")
    op.drop_table("clinical_session_revision_runs")
    with op.batch_alter_table(
        "clinical_session_revision_reviews", schema=None
    ) as batch_op:
        batch_op.drop_index("ix_clinical_session_revision_reviews_session_id")
        batch_op.drop_index("ix_clinical_session_revision_reviews_revision_version_id")
        batch_op.drop_index("ix_clinical_session_revision_reviews_reviewed_at")
    op.drop_table("clinical_session_revision_reviews")
    with op.batch_alter_table("clinical_session_attachments", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_session_attachments_session_id")
    op.drop_table("clinical_session_attachments")
    with op.batch_alter_table("clinical_session_versions", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_session_versions_status")
        batch_op.drop_index("ix_clinical_session_versions_source_version_id")
        batch_op.drop_index("ix_clinical_session_versions_session_id")
        batch_op.drop_index("ix_clinical_session_versions_root_session_id")
        batch_op.drop_index("ix_clinical_session_versions_pipeline_run_id")
    op.drop_table("clinical_session_versions")
    with op.batch_alter_table("clinical_session_timelines", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_session_timelines_session_id")
        batch_op.drop_index("ix_clinical_session_timelines_generated_at")
    op.drop_table("clinical_session_timelines")
    with op.batch_alter_table("clinical_session_sections", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_session_sections_session_id")
    op.drop_table("clinical_session_sections")
    with op.batch_alter_table("clinical_session_results", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_session_results_session_id")
    op.drop_table("clinical_session_results")
    with op.batch_alter_table("clinical_lab_observations", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_lab_observations_session_marker")
        batch_op.drop_index("ix_clinical_lab_observations_observation_at")
    op.drop_table("clinical_lab_observations")
    with op.batch_alter_table("clinical_drug_mentions", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_drug_mentions_session_id")
        batch_op.drop_index("ix_clinical_drug_mentions_normalized_name")
    op.drop_table("clinical_drug_mentions")
    with op.batch_alter_table("medicinal_product_ingredients", schema=None) as batch_op:
        batch_op.drop_index("ix_medicinal_product_ingredients_drug_id")
    op.drop_table("medicinal_product_ingredients")
    with op.batch_alter_table("drug_livertox_links", schema=None) as batch_op:
        batch_op.drop_index("ix_drug_livertox_links_monograph_id")
    op.drop_table("drug_livertox_links")
    with op.batch_alter_table("clinical_sessions", schema=None) as batch_op:
        batch_op.drop_index("ix_clinical_sessions_updated_at_id")
        batch_op.drop_index("ix_clinical_sessions_timestamp_id")
        batch_op.drop_index("ix_clinical_sessions_timestamp")
        batch_op.drop_index("ix_clinical_sessions_status_timestamp_id")
        batch_op.drop_index("ix_clinical_sessions_status_queued_at")
        batch_op.drop_index("ix_clinical_sessions_status")
        batch_op.drop_index("ix_clinical_sessions_patient_name")
    op.drop_table("clinical_sessions")
    with op.batch_alter_table("terminology_conflicts", schema=None) as batch_op:
        batch_op.drop_index("ix_terminology_conflicts_status")
        batch_op.drop_index("ix_terminology_conflicts_source")
    op.drop_table("terminology_conflicts")
    with op.batch_alter_table("medicinal_products", schema=None) as batch_op:
        batch_op.drop_index("ix_medicinal_products_name_norm")
        batch_op.drop_index("ix_medicinal_products_drug_id")
        batch_op.drop_index("ix_medicinal_products_brand_name_norm")
    op.drop_table("medicinal_products")
    with op.batch_alter_table("livertox_monographs", schema=None) as batch_op:
        batch_op.drop_index("ix_livertox_monographs_nbk_id")
        batch_op.drop_index("ix_livertox_monographs_drug_name_norm")
        batch_op.drop_index("ix_livertox_monographs_drug_id")
    op.drop_table("livertox_monographs")
    with op.batch_alter_table("kb_match_cache", schema=None) as batch_op:
        batch_op.drop_index("ix_kb_match_cache_valid")
        batch_op.drop_index("ix_kb_match_cache_raw_drug_name_norm")
        batch_op.drop_index("ix_kb_match_cache_normalized_source")
        batch_op.drop_index("ix_kb_match_cache_drug_id")
    op.drop_table("kb_match_cache")
    with op.batch_alter_table("drug_rxnorm_codes", schema=None) as batch_op:
        batch_op.drop_index("ix_drug_rxnorm_codes_drug_id")
    op.drop_table("drug_rxnorm_codes")
    with op.batch_alter_table("drug_identifiers", schema=None) as batch_op:
        batch_op.drop_index("ix_drug_identifiers_drug_id")
    op.drop_table("drug_identifiers")
    with op.batch_alter_table("drug_concept_relations", schema=None) as batch_op:
        batch_op.drop_index("ix_drug_concept_relations_related_drug_id")
        batch_op.drop_index("ix_drug_concept_relations_drug_id")
    op.drop_table("drug_concept_relations")
    with op.batch_alter_table("drug_atc_codes", schema=None) as batch_op:
        batch_op.drop_index("ix_drug_atc_codes_atc_code")
    op.drop_table("drug_atc_codes")
    with op.batch_alter_table("drug_aliases", schema=None) as batch_op:
        batch_op.drop_index("ix_drug_aliases_drug_id")
        batch_op.drop_index("ix_drug_aliases_alias_norm_source")
    op.drop_table("drug_aliases")
    with op.batch_alter_table("dilirank_records", schema=None) as batch_op:
        batch_op.drop_index("ix_dilirank_records_drug_id")
        batch_op.drop_index("ix_dilirank_records_dili_concern")
        batch_op.drop_index("ix_dilirank_records_compound_name_norm")
    op.drop_table("dilirank_records")
    with op.batch_alter_table("terminology_import_runs", schema=None) as batch_op:
        batch_op.drop_index("ix_terminology_import_runs_source_status")
    op.drop_table("terminology_import_runs")
    with op.batch_alter_table("reference_catalog_manifests", schema=None) as batch_op:
        batch_op.drop_index("ix_reference_catalog_manifests_hash")
    op.drop_table("reference_catalog_manifests")
    with op.batch_alter_table("reference_catalog_entries", schema=None) as batch_op:
        batch_op.drop_index("ix_reference_catalog_entries_manifest")
        batch_op.drop_index("ix_reference_catalog_entries_lookup")
        batch_op.drop_index("ix_reference_catalog_entries_active")
    op.drop_table("reference_catalog_entries")
    op.drop_table("provider_model_catalog_cache")
    with op.batch_alter_table("drugs", schema=None) as batch_op:
        batch_op.drop_index("ix_drugs_livertox_nbk_id")
    op.drop_table("drugs")
    op.drop_table("application_configuration")
    with op.batch_alter_table("access_keys", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_access_keys_active_provider",
            sqlite_where=sa.text("is_active = 1"),
            postgresql_where=sa.text("is_active = true"),
        )
        batch_op.drop_index("ix_access_keys_provider")
    op.drop_table("access_keys")
