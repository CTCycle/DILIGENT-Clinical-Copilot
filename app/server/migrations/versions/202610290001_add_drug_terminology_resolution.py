# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Add the drug terminology resolution schema.

Extends the existing drug concept, alias, identifier and match-cache tables and
adds provenance, concept relations, ATC codes, medicinal products, conflicts
and explicit LiverTox links.

Revision ID: 202610290001
Revises: 202610240001
Create Date: 2026-10-29
"""

from __future__ import annotations

import json
from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "202610290001"
down_revision: Union[str, Sequence[str], None] = "202610240001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

KB_SOURCE_CHECK = "ck_kb_match_cache_ck_kb_match_cache_source"
KB_SOURCE_PREVIOUS = "source IN ('rxnav', 'livertox', 'rag')"
KB_SOURCE_UPGRADED = "source IN ('rxnav', 'livertox', 'rag', 'terminology')"
OBSOLETE_MATCHER_SETTINGS = (
    "direct_confidence",
    "master_confidence",
    "synonym_confidence",
    "normalization_cache_limit",
    "match_cache_limit",
    "alias_cache_limit",
    "min_confidence",
    "token_min_length",
    "catalog_index_limit",
    "spelling_confidence",
    "spelling_min_query_length",
    "spelling_short_name_length",
    "spelling_short_max_distance",
    "spelling_long_max_distance",
)

###############################################################################
def _migrate_existing_rows() -> None:
    # Legacy session aliases only duplicated exact canonical matches.
    op.execute("DELETE FROM drug_aliases WHERE source = 'session'")
    op.execute(
        "UPDATE kb_match_cache SET invalidated_at = CURRENT_TIMESTAMP, "
        "invalidation_reason = 'terminology_upgrade' WHERE invalidated_at IS NULL"
    )
    connection = op.get_bind()
    rows = connection.execute(
        sa.text("SELECT id, payload FROM application_configuration")
    ).fetchall()
    for row_id, raw_payload in rows:
        payload: Any = (
            json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
        )
        if not isinstance(payload, dict):
            continue
        changed = False
        matcher = payload.get("drugs_matcher")
        if isinstance(matcher, dict):
            cleaned = {
                key: value
                for key, value in matcher.items()
                if key not in OBSOLETE_MATCHER_SETTINGS
            }
            changed = cleaned != matcher
            payload["drugs_matcher"] = cleaned
        runtime = payload.get("runtime")
        if isinstance(runtime, dict) and "rxnav_max_concurrency" in runtime:
            payload["runtime"] = {
                key: value for key, value in runtime.items() if key != "rxnav_max_concurrency"
            }
            changed = True
        if not changed:
            continue
        connection.execute(
            sa.text(
                "UPDATE application_configuration SET payload = :payload WHERE id = :id"
            ).bindparams(sa.bindparam("payload", type_=sa.JSON())),
            {"payload": payload, "id": row_id},
        )

###############################################################################
def upgrade() -> None:
    op.create_table('terminology_import_runs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('source_version', sa.String(), nullable=True),
    sa.Column('license', sa.Text(), nullable=True),
    sa.Column('mode', sa.String(), nullable=False),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('started_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.Column('finished_at', sa.DateTime(), nullable=True),
    sa.Column('stats_json', sa.JSON(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.CheckConstraint("mode IN ('full', 'incremental')", name=op.f('ck_terminology_import_runs_mode')),
    sa.CheckConstraint("status IN ('running', 'completed', 'failed', 'cancelled', 'skipped')", name=op.f('ck_terminology_import_runs_status')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_terminology_import_runs'))
    )
    with op.batch_alter_table('terminology_import_runs', schema=None) as batch_op:
        batch_op.create_index('ix_terminology_import_runs_source_status', ['source', 'status', 'finished_at'], unique=False)

    op.create_table('drug_atc_codes',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('drug_id', sa.Integer(), nullable=False),
    sa.Column('atc_code', sa.String(), nullable=False),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('import_run_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['drug_id'], ['drugs.id'], name=op.f('fk_drug_atc_codes_drug_id_drugs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['import_run_id'], ['terminology_import_runs.id'], name=op.f('fk_drug_atc_codes_import_run_id_terminology_import_runs'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_drug_atc_codes')),
    sa.UniqueConstraint('drug_id', 'atc_code', 'source', name='uq_drug_atc_codes_identity')
    )
    with op.batch_alter_table('drug_atc_codes', schema=None) as batch_op:
        batch_op.create_index('ix_drug_atc_codes_atc_code', ['atc_code'], unique=False)

    op.create_table('drug_concept_relations',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('drug_id', sa.Integer(), nullable=False),
    sa.Column('related_drug_id', sa.Integer(), nullable=False),
    sa.Column('relation_type', sa.String(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=True),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('import_run_id', sa.Integer(), nullable=True),
    sa.CheckConstraint("relation_type IN ('form_of', 'has_component')", name=op.f('ck_drug_concept_relations_relation_type')),
    sa.ForeignKeyConstraint(['drug_id'], ['drugs.id'], name=op.f('fk_drug_concept_relations_drug_id_drugs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['import_run_id'], ['terminology_import_runs.id'], name=op.f('fk_drug_concept_relations_import_run_id_terminology_import_runs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['related_drug_id'], ['drugs.id'], name=op.f('fk_drug_concept_relations_related_drug_id_drugs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_drug_concept_relations')),
    sa.UniqueConstraint('drug_id', 'related_drug_id', 'relation_type', 'source', name='uq_drug_concept_relations_identity')
    )
    with op.batch_alter_table('drug_concept_relations', schema=None) as batch_op:
        batch_op.create_index('ix_drug_concept_relations_drug_id', ['drug_id'], unique=False)
        batch_op.create_index('ix_drug_concept_relations_related_drug_id', ['related_drug_id'], unique=False)

    op.create_table('medicinal_products',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('name_norm', sa.String(), nullable=False),
    sa.Column('brand_name_norm', sa.String(), nullable=True),
    sa.Column('country', sa.String(), nullable=True),
    sa.Column('dose_form', sa.String(), nullable=True),
    sa.Column('route', sa.String(), nullable=True),
    sa.Column('composition_status', sa.String(), nullable=False),
    sa.Column('drug_id', sa.Integer(), nullable=True),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('source_product_id', sa.String(), nullable=False),
    sa.Column('import_run_id', sa.Integer(), nullable=True),
    sa.CheckConstraint("composition_status IN ('verified', 'unverified')", name=op.f('ck_medicinal_products_composition_status')),
    sa.ForeignKeyConstraint(['drug_id'], ['drugs.id'], name=op.f('fk_medicinal_products_drug_id_drugs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['import_run_id'], ['terminology_import_runs.id'], name=op.f('fk_medicinal_products_import_run_id_terminology_import_runs'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_medicinal_products')),
    sa.UniqueConstraint('source', 'source_product_id', name='uq_medicinal_products_source_record')
    )
    with op.batch_alter_table('medicinal_products', schema=None) as batch_op:
        batch_op.create_index('ix_medicinal_products_brand_name_norm', ['brand_name_norm'], unique=False)
        batch_op.create_index('ix_medicinal_products_drug_id', ['drug_id'], unique=False)
        batch_op.create_index('ix_medicinal_products_name_norm', ['name_norm'], unique=False)

    op.create_table('terminology_conflicts',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('import_run_id', sa.Integer(), nullable=True),
    sa.Column('conflict_type', sa.String(), nullable=False),
    sa.Column('subject', sa.Text(), nullable=False),
    sa.Column('existing_drug_id', sa.Integer(), nullable=True),
    sa.Column('proposed_drug_id', sa.Integer(), nullable=True),
    sa.Column('detail', sa.Text(), nullable=True),
    sa.Column('status', sa.String(), server_default=sa.text("'open'"), nullable=False),
    sa.Column('detected_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.Column('resolved_at', sa.DateTime(), nullable=True),
    sa.CheckConstraint("conflict_type IN ('identifier_mismatch', 'name_identity_mismatch', 'composition_mismatch')", name=op.f('ck_terminology_conflicts_conflict_type')),
    sa.CheckConstraint("status IN ('open', 'resolved', 'dismissed')", name=op.f('ck_terminology_conflicts_status')),
    sa.ForeignKeyConstraint(['existing_drug_id'], ['drugs.id'], name=op.f('fk_terminology_conflicts_existing_drug_id_drugs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['import_run_id'], ['terminology_import_runs.id'], name=op.f('fk_terminology_conflicts_import_run_id_terminology_import_runs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['proposed_drug_id'], ['drugs.id'], name=op.f('fk_terminology_conflicts_proposed_drug_id_drugs'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_terminology_conflicts'))
    )
    with op.batch_alter_table('terminology_conflicts', schema=None) as batch_op:
        batch_op.create_index('ix_terminology_conflicts_source', ['source'], unique=False)
        batch_op.create_index('ix_terminology_conflicts_status', ['status'], unique=False)

    op.create_table('drug_livertox_links',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('drug_id', sa.Integer(), nullable=False),
    sa.Column('monograph_id', sa.Integer(), nullable=False),
    sa.Column('link_type', sa.String(), nullable=False),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
    sa.CheckConstraint("link_type IN ('monograph_owner', 'verified_alias', 'component_set', 'form_of_parent', 'form_of_child')", name=op.f('ck_drug_livertox_links_link_type')),
    sa.ForeignKeyConstraint(['drug_id'], ['drugs.id'], name=op.f('fk_drug_livertox_links_drug_id_drugs'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['monograph_id'], ['livertox_monographs.id'], name=op.f('fk_drug_livertox_links_monograph_id_livertox_monographs'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_drug_livertox_links')),
    sa.UniqueConstraint('drug_id', 'monograph_id', name='uq_drug_livertox_links_identity')
    )
    with op.batch_alter_table('drug_livertox_links', schema=None) as batch_op:
        batch_op.create_index('ix_drug_livertox_links_monograph_id', ['monograph_id'], unique=False)

    op.create_table('medicinal_product_ingredients',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('product_id', sa.Integer(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('drug_id', sa.Integer(), nullable=True),
    sa.Column('ingredient_name', sa.Text(), nullable=False),
    sa.Column('strength_value', sa.Float(), nullable=True),
    sa.Column('strength_unit', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['drug_id'], ['drugs.id'], name=op.f('fk_medicinal_product_ingredients_drug_id_drugs'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['product_id'], ['medicinal_products.id'], name=op.f('fk_medicinal_product_ingredients_product_id_medicinal_products'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_medicinal_product_ingredients')),
    sa.UniqueConstraint('product_id', 'position', name='uq_medicinal_product_ingredients_position')
    )
    with op.batch_alter_table('medicinal_product_ingredients', schema=None) as batch_op:
        batch_op.create_index('ix_medicinal_product_ingredients_drug_id', ['drug_id'], unique=False)

    with op.batch_alter_table('drug_aliases', schema=None) as batch_op:
        batch_op.add_column(sa.Column('language', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('country', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('validation_status', sa.String(), server_default=sa.text("'verified'"), nullable=False))
        batch_op.add_column(sa.Column('import_run_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('source_record_id', sa.String(), nullable=True))
        batch_op.create_foreign_key(batch_op.f('fk_drug_aliases_import_run_id_terminology_import_runs'), 'terminology_import_runs', ['import_run_id'], ['id'], ondelete='SET NULL')
        batch_op.create_check_constraint(batch_op.f('ck_drug_aliases_validation_status'), "validation_status IN ('verified', 'confirmed', 'provisional', 'rejected')")

    with op.batch_alter_table('drug_identifiers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('source', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('import_run_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(batch_op.f('fk_drug_identifiers_import_run_id_terminology_import_runs'), 'terminology_import_runs', ['import_run_id'], ['id'], ondelete='SET NULL')

    with op.batch_alter_table('drugs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('concept_type', sa.String(), server_default=sa.text("'ingredient'"), nullable=False))
        batch_op.add_column(sa.Column('preferred_inn', sa.Text(), nullable=True))
        batch_op.create_check_constraint(batch_op.f('ck_drugs_concept_type'), "concept_type IN ('ingredient', 'precise_ingredient', 'multi_ingredient')")

    with op.batch_alter_table('kb_match_cache', schema=None) as batch_op:
        batch_op.add_column(sa.Column('resolution_status', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('method', sa.String(), nullable=True))
        batch_op.add_column(sa.Column('terminology_revision', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('expires_at', sa.DateTime(), nullable=True))
        batch_op.drop_constraint(batch_op.f(KB_SOURCE_CHECK), type_='check')
        batch_op.create_check_constraint(batch_op.f(KB_SOURCE_CHECK), KB_SOURCE_UPGRADED)

    _migrate_existing_rows()

###############################################################################
def downgrade() -> None:
    op.execute(
        "DELETE FROM kb_match_cache WHERE source = 'terminology'"
    )
    with op.batch_alter_table('kb_match_cache', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f(KB_SOURCE_CHECK), type_='check')
        batch_op.create_check_constraint(batch_op.f(KB_SOURCE_CHECK), KB_SOURCE_PREVIOUS)
        batch_op.drop_column('expires_at')
        batch_op.drop_column('terminology_revision')
        batch_op.drop_column('method')
        batch_op.drop_column('resolution_status')

    with op.batch_alter_table('drugs', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('ck_drugs_concept_type'), type_='check')
        batch_op.drop_column('preferred_inn')
        batch_op.drop_column('concept_type')

    with op.batch_alter_table('drug_identifiers', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('fk_drug_identifiers_import_run_id_terminology_import_runs'), type_='foreignkey')
        batch_op.drop_column('import_run_id')
        batch_op.drop_column('source')

    with op.batch_alter_table('drug_aliases', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('ck_drug_aliases_validation_status'), type_='check')
        batch_op.drop_constraint(batch_op.f('fk_drug_aliases_import_run_id_terminology_import_runs'), type_='foreignkey')
        batch_op.drop_column('source_record_id')
        batch_op.drop_column('import_run_id')
        batch_op.drop_column('validation_status')
        batch_op.drop_column('country')
        batch_op.drop_column('language')

    with op.batch_alter_table('medicinal_product_ingredients', schema=None) as batch_op:
        batch_op.drop_index('ix_medicinal_product_ingredients_drug_id')

    op.drop_table('medicinal_product_ingredients')
    with op.batch_alter_table('drug_livertox_links', schema=None) as batch_op:
        batch_op.drop_index('ix_drug_livertox_links_monograph_id')

    op.drop_table('drug_livertox_links')
    with op.batch_alter_table('terminology_conflicts', schema=None) as batch_op:
        batch_op.drop_index('ix_terminology_conflicts_status')
        batch_op.drop_index('ix_terminology_conflicts_source')

    op.drop_table('terminology_conflicts')
    with op.batch_alter_table('medicinal_products', schema=None) as batch_op:
        batch_op.drop_index('ix_medicinal_products_name_norm')
        batch_op.drop_index('ix_medicinal_products_drug_id')
        batch_op.drop_index('ix_medicinal_products_brand_name_norm')

    op.drop_table('medicinal_products')
    with op.batch_alter_table('drug_concept_relations', schema=None) as batch_op:
        batch_op.drop_index('ix_drug_concept_relations_related_drug_id')
        batch_op.drop_index('ix_drug_concept_relations_drug_id')

    op.drop_table('drug_concept_relations')
    with op.batch_alter_table('drug_atc_codes', schema=None) as batch_op:
        batch_op.drop_index('ix_drug_atc_codes_atc_code')

    op.drop_table('drug_atc_codes')
    with op.batch_alter_table('terminology_import_runs', schema=None) as batch_op:
        batch_op.drop_index('ix_terminology_import_runs_source_status')

    op.drop_table('terminology_import_runs')
