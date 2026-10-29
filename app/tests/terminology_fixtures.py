# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Small, deterministic terminology fixture shared by resolver and clinical tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from repositories.context import RepositoryContext
from repositories.database.migrations import migrate_database
from repositories.schemas.knowledge import Drug, LiverToxMonograph
from repositories.terminology_repository import TerminologyRepository
from services.terminology.normalization import fold
from services.updater.terminology.ingestor import TerminologyIngestor
from services.updater.terminology.snapshot import (
    StagedConcept,
    StagedProduct,
    StagedProductIngredient,
    TerminologySnapshot,
)
from services.updater.terminology.sources.curated import CuratedAliasSource

# rxcui -> (name, concept type, parent rxcui, component rxcuis, extra aliases)
RXNORM_CONCEPTS: dict[str, tuple[str, str, str | None, tuple[str, ...], tuple[tuple[str, str], ...]]] = {
    "161": ("acetaminophen", "ingredient", None, (), (("Tylenol", "brand"), ("paracetamol", "synonym"))),
    "83367": ("atorvastatin", "ingredient", None, (), (("Lipitor", "brand"),)),
    "723": ("amoxicillin", "ingredient", None, (), ()),
    "48203": ("clavulanate", "ingredient", None, (), (("clavulanic acid", "synonym"),)),
    "19711": ("amoxicillin / clavulanate", "multi_ingredient", None, ("723", "48203"), (("Augmentin", "brand"),)),
    "10180": ("sulfamethoxazole", "ingredient", None, (), ()),
    "10829": ("trimethoprim", "ingredient", None, (), ()),
    "82122": ("sulfamethoxazole / trimethoprim", "multi_ingredient", None, ("10180", "10829"), (("Bactrim", "brand"),)),
    "6918": ("metoprolol", "ingredient", None, (), ()),
    "221124": ("metoprolol succinate", "precise_ingredient", "6918", (), ()),
    "5553": ("hydroxyzine", "ingredient", None, (), ()),
    "5470": ("hydralazine", "ingredient", None, (), ()),
    "6809": ("metformin", "ingredient", None, (), ()),
    "5487": ("hydrochlorothiazide", "ingredient", None, (), ()),
    "7646": ("omeprazole", "ingredient", None, (), ()),
    "283742": ("esomeprazole", "ingredient", None, (), ()),
    "6851": ("methotrexate", "ingredient", None, (), ()),
    "7052": ("morphine", "ingredient", None, (), (("MS", "abbreviation"),)),
    "6585": ("magnesium sulfate", "ingredient", None, (), (("MS", "abbreviation"),)),
}
LIVERTOX_MONOGRAPHS = (
    ("Acetaminophen", "NBK548162"),
    ("Atorvastatin", "NBK548236"),
    ("Amoxicillin-Clavulanate", "NBK548517"),
    ("Trimethoprim-Sulfamethoxazole", "NBK548221"),
    ("Metoprolol", "NBK548300"),
    ("Hydroxyzine", "NBK548230"),
    ("Hydralazine", "NBK548900"),
    ("Omeprazole", "NBK548195"),
    ("Esomeprazole", "NBK548196"),
    ("Methotrexate", "NBK548219"),
)

###############################################################################
@dataclass
class StaticSource:
    """Terminology source returning a prepared snapshot (no network)."""

    snapshot: TerminologySnapshot
    key: str = ""
    license: str = "test"
    attribution: str = "test fixture"
    precedence: int = 1
    fetches: list[int] = field(default_factory=list)

    # -------------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.key = self.key or self.snapshot.source

    # -------------------------------------------------------------------------
    def probe_version(self) -> str | None:
        return self.snapshot.version

    # -------------------------------------------------------------------------
    def fetch(self, progress, should_stop) -> TerminologySnapshot:  # type: ignore[no-untyped-def]
        del progress, should_stop
        self.fetches.append(1)
        return self.snapshot


# -----------------------------------------------------------------------------
def build_sqlite_engine(path: Path) -> Engine:
    engine = create_engine(
        f"sqlite+pysqlite:///{path}",
        future=True,
        connect_args={"timeout": 30.0},
    )

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    migrate_database(engine, database_was_empty=True)
    return engine


# -----------------------------------------------------------------------------
def build_context(engine: Engine) -> RepositoryContext:
    factory = sessionmaker(bind=engine, future=True, expire_on_commit=False)
    return RepositoryContext.create(engine=engine, session_factory=factory)


# -----------------------------------------------------------------------------
def rxnorm_snapshot(version: str = "rxnorm-test-1") -> TerminologySnapshot:
    snapshot = TerminologySnapshot(source="rxnorm", version=version, license="Public domain")
    for rxcui, (name, concept_type, parent, components, aliases) in RXNORM_CONCEPTS.items():
        snapshot.concepts[rxcui] = StagedConcept(
            key=rxcui,
            name=name,
            concept_type=concept_type,  # type: ignore[arg-type]
            rxcui=rxcui,
            parent_key=parent,
            component_keys=components,
        )
        snapshot.add_alias(rxcui, name, "canonical")
        for alias, kind in aliases:
            snapshot.add_alias(rxcui, alias, kind)
    snapshot.products.append(
        StagedProduct(
            product_id="861007",
            name="metformin hydrochloride 500 MG Oral Tablet",
            brand_name=None,
            ingredients=[
                StagedProductIngredient(
                    "metformin hydrochloride", concept_key="6809", strength_value=500.0, strength_unit="mg"
                )
            ],
        )
    )
    return snapshot


# -----------------------------------------------------------------------------
def swissmedic_snapshot(version: str = "swissmedic-test-1") -> TerminologySnapshot:
    snapshot = TerminologySnapshot(
        source="swissmedic", version=version, license="Open use", brand_families=True
    )
    snapshot.products.extend(
        [
            StagedProduct(
                product_id="CH-1",
                name="Dafalgan Tabl 500 mg",
                brand_name="Dafalgan",
                country="CH",
                ingredients=[StagedProductIngredient("paracetamolum", lookup_names=("paracetamol",))],
            ),
            StagedProduct(
                product_id="CH-2",
                name="Zurcal Filmtabl 40 mg",
                brand_name="Zurcal",
                country="CH",
                ingredients=[StagedProductIngredient("pantoprazolum natricum", lookup_names=("pantoprazole",))],
            ),
        ]
    )
    return snapshot


# -----------------------------------------------------------------------------
def add_livertox_monographs(context: RepositoryContext) -> None:
    """Mirror the LiverTox updater: one monograph per drug row found by name."""
    with context.session_factory() as db_session:
        for name, nbk_id in LIVERTOX_MONOGRAPHS:
            norm = fold(name)
            drug = db_session.execute(select(Drug).where(Drug.canonical_name_norm == norm)).scalar_one_or_none()
            if drug is None:
                drug = Drug(canonical_name=name, canonical_name_norm=norm)
                db_session.add(drug)
                db_session.flush()
            db_session.add(
                LiverToxMonograph(
                    drug_id=int(drug.id),
                    monograph_key=f"{norm}|{nbk_id}",
                    drug_name_norm=norm,
                    nbk_id=nbk_id,
                    excerpt=f"{name} LiverTox excerpt.",
                    likelihood_score="B",
                )
            )
        db_session.commit()


# -----------------------------------------------------------------------------
def ingest(repository: TerminologyRepository, *sources: object, force_full: bool = False) -> dict:
    return TerminologyIngestor(repository, sources).run(  # type: ignore[arg-type]
        force_full=force_full,
        progress=lambda *_args: None,
        should_stop=lambda: False,
    )


# -----------------------------------------------------------------------------
def seed_reference_terminology(context: RepositoryContext) -> TerminologyRepository:
    """RxNorm, LiverTox monographs, curated catalog aliases and Swiss products."""
    repository = TerminologyRepository(context)
    ingest(repository, StaticSource(rxnorm_snapshot()))
    add_livertox_monographs(context)
    ingest(repository, CuratedAliasSource(), StaticSource(swissmedic_snapshot()))
    return repository
