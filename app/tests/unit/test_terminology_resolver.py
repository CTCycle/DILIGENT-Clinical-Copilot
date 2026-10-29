# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

from __future__ import annotations

import threading
from collections.abc import Callable

import httpx
import pytest
from sqlalchemy import func, select

import services.terminology.external as external_module
from repositories.context import RepositoryContext
from repositories.schemas.knowledge import DrugAlias
from repositories.terminology_lookup_repository import TerminologyLookupRepository
from services.terminology.external import RxNavRuntimeClient
from services.terminology.index import TerminologyIndexCache
from services.terminology.resolver import ResolverSettings, TerminologyResolver
from terminology_fixtures import build_context, build_sqlite_engine, seed_reference_terminology


# -----------------------------------------------------------------------------
def _seeded_context(tmp_path_factory) -> RepositoryContext:  # type: ignore[no-untyped-def]
    context = build_context(build_sqlite_engine(tmp_path_factory.mktemp("terminology") / "t.db"))
    seed_reference_terminology(context)
    return context


###############################################################################
@pytest.fixture(scope="module")
def seeded_context(tmp_path_factory) -> RepositoryContext:  # type: ignore[no-untyped-def]
    return _seeded_context(tmp_path_factory)


# -----------------------------------------------------------------------------
def build_resolver(
    context: RepositoryContext,
    *,
    external: RxNavRuntimeClient | None = None,
    **settings: int,
) -> TerminologyResolver:
    return TerminologyResolver(
        TerminologyLookupRepository(context),
        TerminologyIndexCache(),
        settings=ResolverSettings(**settings),
        external=external,
    )


###############################################################################
@pytest.mark.parametrize(
    ("text", "preferred", "ingredients", "method"),
    [
        ("Lipitor", "atorvastatin", ["atorvastatin"], "exact_alias"),
        ("paracetamol", "acetaminophen", ["acetaminophen"], "exact_alias"),
        ("Dafalgan", "acetaminophen", ["acetaminophen"], "exact_alias"),
        ("HCTZ", "hydrochlorothiazide", ["hydrochlorothiazide"], "confirmed_local"),
        ("Augmentin", "amoxicillin / clavulanate", ["amoxicillin", "clavulanate"], "exact_alias"),
        ("amoxicillin/clavulanic acid", "amoxicillin / clavulanate", ["amoxicillin", "clavulanate"], "combination"),
        ("metformina", "metformin", ["metformin"], "language_variant"),
        ("esomeprazole", "esomeprazole", ["esomeprazole"], "exact_alias"),
    ],
)
def test_verified_local_resolution(
    seeded_context: RepositoryContext,
    text: str,
    preferred: str,
    ingredients: list[str],
    method: str,
) -> None:
    result = build_resolver(seeded_context).resolve(text)

    assert (result.status, result.accepted) == ("verified", True)
    assert result.preferred_name == preferred
    assert [item.name for item in result.ingredients] == ingredients
    assert result.method == method
    assert result.original == text


###############################################################################
def test_salt_stays_a_precise_ingredient_under_its_parent(seeded_context: RepositoryContext) -> None:
    result = build_resolver(seeded_context).resolve("Metoprolol succinate ER 50 mg")

    assert result.preferred_name == "metoprolol succinate"
    assert result.concept_type == "precise_ingredient"
    assert result.ingredients[0].parent_name == "metoprolol"
    assert result.qualifiers == ["er"]
    assert result.strengths == ["50 mg"]


###############################################################################
def test_strength_support_is_evaluated_per_input(seeded_context: RepositoryContext) -> None:
    resolver = build_resolver(seeded_context)

    supported = resolver.resolve("metformin 500mg")
    unsupported = resolver.resolve("metformin 850 mg")

    assert supported.strength_supported is True
    # Same concept, different strength: the cached result must not leak.
    assert unsupported.strength_supported is False
    assert unsupported.strengths == ["850 mg"]
    assert unsupported.drug_id == supported.drug_id


###############################################################################
def test_spelling_correction_is_probable_and_flagged(seeded_context: RepositoryContext) -> None:
    result = build_resolver(seeded_context).resolve("amoxcillin")

    assert (result.status, result.accepted, result.method) == ("probable", True, "fuzzy")
    assert result.preferred_name == "amoxicillin"
    assert "spelling_corrected_requires_review" in result.review_reasons
    assert result.candidates[0].edit_distance == 1


###############################################################################
@pytest.mark.parametrize(
    ("text", "reason", "candidates"),
    [
        # Look-alike/sound-alike pair inside the edit bound.
        ("hydroxazine", "lookalike_within_edit_bound", {"hydroxyzine", "hydralazine"}),
        ("eomeprazole", "near_equidistant_candidates", {"omeprazole", "esomeprazole"}),
        ("ketoprolol", "first_letter_differs", {"metoprolol"}),
        ("MS", "error_prone_abbreviation", {"morphine", "magnesium sulfate"}),
    ],
)
def test_unsafe_inputs_stay_ambiguous_with_candidates(
    seeded_context: RepositoryContext,
    text: str,
    reason: str,
    candidates: set[str],
) -> None:
    result = build_resolver(seeded_context).resolve(text)

    assert (result.status, result.accepted, result.drug_id) == ("ambiguous", False, None)
    assert reason in result.review_reasons
    assert {item.name for item in result.candidates} == candidates


###############################################################################
def test_unverified_product_and_unknown_input_are_not_accepted(seeded_context: RepositoryContext) -> None:
    resolver = build_resolver(seeded_context)

    product = resolver.resolve("Zurcal")
    unknown = resolver.resolve("zzyzx")

    assert (product.status, product.accepted, product.method) == ("probable", False, "product_unverified")
    assert "product_composition_unverified" in product.review_reasons
    assert (unknown.status, unknown.accepted) == ("unresolved", False)


###############################################################################
class FakeRxNav:
    """Minimal RxNav REST responder for httpx.MockTransport."""

    # -------------------------------------------------------------------------
    def __init__(self, names: dict[str, str], related: dict[str, list[dict[str, str]]]) -> None:
        self.names = names
        self.related = related
        self.paths: list[str] = []

    # -------------------------------------------------------------------------
    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/REST/")
        self.paths.append(path)
        if path == "rxcui.json":
            rxcui = self.names.get(request.url.params["name"].casefold())
            return httpx.Response(200, json={"idGroup": {"rxnormId": [rxcui] if rxcui else []}})
        rxcui = path.split("/")[1]
        if path.endswith("related.json"):
            group = {"conceptGroup": [{"conceptProperties": self.related.get(rxcui, [])}]}
            return httpx.Response(200, json={"relatedGroup": group})
        return httpx.Response(200, json={})


# -----------------------------------------------------------------------------
def rxnav_client(handler: Callable[[httpx.Request], httpx.Response]) -> RxNavRuntimeClient:
    return RxNavRuntimeClient(timeout=1.0, transport=httpx.MockTransport(handler))


###############################################################################
@pytest.fixture
def fresh_context(tmp_path_factory, monkeypatch: pytest.MonkeyPatch) -> RepositoryContext:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(external_module.time, "sleep", lambda _seconds: None)
    return _seeded_context(tmp_path_factory)


###############################################################################
def test_external_relation_is_persisted_as_verified_runtime_alias(fresh_context: RepositoryContext) -> None:
    rxnav = FakeRxNav(
        names={"panadol": "1000000"},
        related={"1000000": [{"rxcui": "161", "name": "acetaminophen", "tty": "IN"}]},
    )
    result = build_resolver(fresh_context, external=rxnav_client(rxnav)).resolve("Panadol")

    assert (result.status, result.method, result.preferred_name) == ("verified", "external_rxnav", "acetaminophen")
    assert result.external_calls == 2
    with fresh_context.session_factory() as db_session:
        alias = db_session.execute(select(DrugAlias).where(DrugAlias.alias_norm == "panadol")).scalar_one()
    assert (alias.source, alias.validation_status, alias.source_record_id) == ("rxnav_runtime", "verified", "1000000")

    # The persisted alias now resolves locally, without RxNav.
    local = build_resolver(fresh_context).resolve("Panadol")
    assert (local.method, local.external_calls) == ("exact_alias", 0)


###############################################################################
def test_external_miss_is_negatively_cached(fresh_context: RepositoryContext) -> None:
    rxnav = FakeRxNav(names={}, related={})

    first = build_resolver(fresh_context, external=rxnav_client(rxnav)).resolve("Unknownium")
    calls_after_first = len(rxnav.paths)
    second = build_resolver(fresh_context, external=rxnav_client(rxnav)).resolve("Unknownium")

    assert first.status == second.status == "unresolved"
    assert calls_after_first == 1
    assert len(rxnav.paths) == calls_after_first


###############################################################################
def test_external_budget_and_outage_degrade_without_caching(fresh_context: RepositoryContext) -> None:
    rxnav = FakeRxNav(
        names={"panadol": "1000000"},
        related={"1000000": [{"rxcui": "161", "name": "acetaminophen", "tty": "IN"}]},
    )
    budgeted = build_resolver(
        fresh_context, external=rxnav_client(rxnav), external_max_requests_per_name=1
    ).resolve("Panadol")
    assert (budgeted.status, budgeted.accepted) == ("unresolved", False)
    assert rxnav.paths == ["rxcui.json"]

    def outage(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("RxNav unreachable")

    degraded = build_resolver(fresh_context, external=rxnav_client(outage)).resolve("Panadol")
    assert (degraded.status, degraded.accepted) == ("unresolved", False)
    # Neither a budget stop nor an outage is remembered as a negative result.
    recovered = build_resolver(fresh_context, external=rxnav_client(rxnav)).resolve("Panadol")
    assert recovered.method == "external_rxnav"


###############################################################################
def test_concurrent_alias_recording_creates_one_row(fresh_context: RepositoryContext) -> None:
    lookup = TerminologyLookupRepository(fresh_context)
    drug_id = build_resolver(fresh_context).resolve("acetaminophen").drug_id
    assert drug_id is not None
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def record() -> None:
        try:
            barrier.wait()
            lookup.record_alias(
                drug_id=drug_id,
                alias="Panadol",
                alias_norm="panadol",
                alias_kind="synonym",
                source="rxnav_runtime",
                validation_status="verified",
            )
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=record) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    with fresh_context.session_factory() as db_session:
        count = db_session.execute(
            select(func.count()).select_from(DrugAlias).where(DrugAlias.alias_norm == "panadol")
        ).scalar_one()
    assert count == 1
