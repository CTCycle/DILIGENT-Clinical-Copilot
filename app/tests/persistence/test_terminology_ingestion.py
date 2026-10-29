# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from repositories.schemas.knowledge import Drug, DrugAlias, DrugIdentifier
from repositories.schemas.terminology import (
    DrugLiverToxLink,
    MedicinalProduct,
    TerminologyConflict,
    TerminologyImportRun,
)
from repositories.terminology_repository import TerminologyRepository
from services.updater.terminology.snapshot import (
    StagedConcept,
    TerminologySnapshot,
    TerminologySourceUnavailable,
)
from terminology_fixtures import (
    StaticSource,
    add_livertox_monographs,
    build_context,
    ingest,
    rxnorm_snapshot,
    seed_reference_terminology,
    swissmedic_snapshot,
)


# -----------------------------------------------------------------------------
def table_state(repository: TerminologyRepository) -> dict[str, object]:
    with repository.session_factory() as db_session:
        return {
            "drugs": db_session.execute(select(func.count()).select_from(Drug)).scalar_one(),
            "aliases": sorted(
                db_session.execute(
                    select(DrugAlias.drug_id, DrugAlias.alias_norm, DrugAlias.alias_kind, DrugAlias.source, DrugAlias.validation_status)
                ).all()
            ),
            "products": sorted(
                db_session.execute(
                    select(MedicinalProduct.source_product_id, MedicinalProduct.composition_status, MedicinalProduct.drug_id)
                ).all()
            ),
            "links": sorted(
                db_session.execute(
                    select(DrugLiverToxLink.drug_id, DrugLiverToxLink.monograph_id, DrugLiverToxLink.link_type)
                ).all()
            ),
        }


# -----------------------------------------------------------------------------
def drug_id(repository: TerminologyRepository, name: str) -> int:
    with repository.session_factory() as db_session:
        return int(db_session.execute(select(Drug.id).where(Drug.canonical_name == name)).scalar_one())


###############################################################################
def test_full_refresh_is_idempotent_and_unchanged_versions_skip(persistence_engine: Engine) -> None:
    repository = seed_reference_terminology(build_context(persistence_engine))
    before = table_state(repository)

    rerun = ingest(repository, StaticSource(rxnorm_snapshot()), StaticSource(swissmedic_snapshot()), force_full=True)
    assert {item["status"] for item in rerun["sources"].values()} == {"completed"}
    assert table_state(repository) == before

    unchanged = StaticSource(rxnorm_snapshot())
    skipped = ingest(repository, unchanged)
    assert skipped["sources"]["rxnorm"]["status"] == "skipped"
    assert unchanged.fetches == []


###############################################################################
def test_livertox_links_cover_owner_component_set_and_salt_forms(persistence_engine: Engine) -> None:
    repository = seed_reference_terminology(build_context(persistence_engine))
    with repository.session_factory() as db_session:
        links = set(
            db_session.execute(
                select(Drug.canonical_name, DrugLiverToxLink.link_type).join(
                    Drug, Drug.id == DrugLiverToxLink.drug_id
                )
            ).tuples()
        )

    assert ("acetaminophen", "monograph_owner") in links
    # "Trimethoprim-Sulfamethoxazole" is linked to the RxNorm combination concept.
    assert ("sulfamethoxazole / trimethoprim", "component_set") in links
    # A salt inherits its parent's monograph.
    assert ("metoprolol succinate", "form_of_parent") in links
    # Single components of a combination do not inherit the combination monograph.
    assert not {name for name, _kind in links} & {"trimethoprim", "sulfamethoxazole"}


###############################################################################
def test_failed_source_keeps_last_good_snapshot(persistence_engine: Engine) -> None:
    repository = seed_reference_terminology(build_context(persistence_engine))
    before = table_state(repository)

    class Unreachable(StaticSource):
        def fetch(self, progress, should_stop):  # type: ignore[no-untyped-def]
            raise TerminologySourceUnavailable("download truncated")

    result = ingest(
        repository,
        Unreachable(rxnorm_snapshot(version="rxnorm-test-2")),
        StaticSource(swissmedic_snapshot(version="swissmedic-test-2")),
    )

    assert result["failed_sources"] == ["rxnorm"]
    assert result["sources"]["swissmedic"]["status"] == "completed"
    assert table_state(repository) == before
    assert repository.last_completed_version("rxnorm") == "rxnorm-test-1"


###############################################################################
def test_persistence_error_rolls_back_the_whole_source(
    persistence_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = seed_reference_terminology(build_context(persistence_engine))
    before = table_state(repository)

    def explode(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("disk full")

    monkeypatch.setattr(repository, "_write_products", explode)
    result = ingest(repository, StaticSource(rxnorm_snapshot(version="rxnorm-test-2")))

    assert result["sources"]["rxnorm"]["status"] == "failed"
    assert table_state(repository) == before
    with repository.session_factory() as db_session:
        statuses = db_session.execute(
            select(TerminologyImportRun.status).where(TerminologyImportRun.source == "rxnorm")
        ).scalars().all()
    assert statuses == ["completed", "failed"]


###############################################################################
def test_contradicting_cross_walk_records_a_conflict_instead_of_merging(persistence_engine: Engine) -> None:
    context = build_context(persistence_engine)
    repository = TerminologyRepository(context)
    ingest(repository, StaticSource(rxnorm_snapshot()))
    add_livertox_monographs(context)
    atorvastatin = drug_id(repository, "atorvastatin")
    metoprolol = drug_id(repository, "metoprolol")

    snapshot = TerminologySnapshot(source="drugcentral", version="dc-1", license="CC BY-SA 4.0")
    # The cross-walk wrongly points "atorvastatin" at metoprolol's RxCUI.
    snapshot.concepts["dc-1"] = StagedConcept(
        key="dc-1", name="atorvastatin", rxcui="6918", identifiers={"unii": "A0JWA85V8F", "drugcentral": "dc-1"}
    )
    snapshot.add_alias("dc-1", "atorvastatin calcium", "synonym")
    snapshot.concepts["dc-2"] = StagedConcept(
        key="dc-2", name="metoprolol", identifiers={"unii": "GEB06NHM23", "drugcentral": "dc-2"}
    )
    ingest(repository, StaticSource(snapshot))

    with repository.session_factory() as db_session:
        conflicts = db_session.execute(
            select(TerminologyConflict.conflict_type, TerminologyConflict.existing_drug_id, TerminologyConflict.proposed_drug_id)
        ).all()
        synonym_target = db_session.execute(
            select(DrugAlias.drug_id).where(DrugAlias.alias_norm == "atorvastatin calcium")
        ).scalar_one()
        owners = dict(
            db_session.execute(select(DrugIdentifier.identifier_value, DrugIdentifier.drug_id)).all()
        )

    assert ("identifier_mismatch", metoprolol, atorvastatin) in conflicts
    assert synonym_target == atorvastatin
    assert owners["A0JWA85V8F"] == atorvastatin
    assert owners["GEB06NHM23"] == metoprolol
