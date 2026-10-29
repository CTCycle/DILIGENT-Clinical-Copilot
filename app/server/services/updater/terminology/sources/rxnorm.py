# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""RxNorm Current Prescribable Content (public domain, no UMLS licence).

Concepts are RxNorm ingredients (IN), precise ingredients (PIN, e.g. salts)
and multiple-ingredient concepts (MIN). Brand names (BN) become aliases of the
ingredient or MIN concept matching their branded products' composition.
RXNREL rows read ``RXCUI2 <RELA> RXCUI1``.
"""

from __future__ import annotations

import io
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from services.updater.terminology.download import USER_AGENT, download_file, request_json
from services.updater.terminology.snapshot import (
    ConceptType,
    ProgressFn,
    StagedConcept,
    StagedProduct,
    StagedProductIngredient,
    StopFn,
    TerminologySnapshot,
    TerminologySourceUnavailable,
)

RELEASES_URL = "https://uts-ws.nlm.nih.gov/releases"
RELEASE_TYPE = "rxnorm-prescribable-content-monthly-release"
LICENSE = "Public domain (NLM RxNorm Current Prescribable Content; no licence required)"
ATTRIBUTION = "RxNorm, U.S. National Library of Medicine"
CONCEPT_TYPES: dict[str, ConceptType] = {
    "IN": "ingredient",
    "PIN": "precise_ingredient",
    "MIN": "multi_ingredient",
}
SYNONYM_TYPES = {"SY", "TMSY"}
PRODUCT_TYPES = {"SCD", "SBD"}
_BRAND_RE = re.compile(r"\s*\[([^\]]+)\]\s*$")
_COMPONENT_RE = re.compile(
    r"^(?P<name>.+?) (?P<value>\d+(?:\.\d+)?) (?P<unit>[A-Z%]+(?:/[A-Z]+)?)(?: (?P<rest>.+))?$"
)
_QUANTITY_PREFIX_RE = re.compile(r"^\d+(?:\.\d+)? [A-Z]+ ")

###############################################################################
@dataclass
class _ConceptTable:
    concepts: dict[str, str] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)
    brands: dict[str, str] = field(default_factory=dict)
    products: dict[str, str] = field(default_factory=dict)
    synonyms: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

###############################################################################
@dataclass
class _RelationTable:
    parents: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    components: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    brand_ingredients: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

###############################################################################
class RxNormPrescribableSource:
    key = "rxnorm"
    license = LICENSE
    attribution = ATTRIBUTION
    precedence = 1

    # -------------------------------------------------------------------------
    def __init__(self, archive_dir: Path, *, timeout: float = 300.0) -> None:
        self.archive_dir = archive_dir
        self.timeout = timeout
        self._release: dict[str, str] | None = None

    # -------------------------------------------------------------------------
    def _current_release(self) -> dict[str, str]:
        if self._release is None:
            with httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}) as client:
                payload = request_json(
                    client,
                    RELEASES_URL,
                    params={"releaseType": RELEASE_TYPE, "current": "true"},
                )
            if not isinstance(payload, list) or not payload:
                raise TerminologySourceUnavailable("RxNorm release listing is empty")
            release = payload[0]
            if not release.get("downloadUrl") or not release.get("fileName"):
                raise TerminologySourceUnavailable("RxNorm release listing is incomplete")
            self._release = {
                "url": str(release["downloadUrl"]),
                "file_name": str(release["fileName"]),
                "version": str(release.get("releaseVersion") or release["fileName"]),
            }
        return self._release

    # -------------------------------------------------------------------------
    def probe_version(self) -> str | None:
        return self._current_release()["version"]

    # -------------------------------------------------------------------------
    def fetch(self, progress: ProgressFn, should_stop: StopFn) -> TerminologySnapshot:
        release = self._current_release()
        progress(5.0, f"Downloading RxNorm {release['version']}")
        archive, _sha = download_file(
            release["url"],
            self.archive_dir / "rxnorm_prescribable.zip",
            timeout=self.timeout,
            should_stop=should_stop,
        )
        progress(40.0, "Parsing RxNorm concepts")
        return self.parse_archive(archive, version=release["version"], should_stop=should_stop)

    # -------------------------------------------------------------------------
    def parse_archive(
        self,
        archive: Path,
        *,
        version: str | None,
        should_stop: StopFn,
    ) -> TerminologySnapshot:
        try:
            with zipfile.ZipFile(archive) as bundle:
                conso_name = self._member(bundle, "RXNCONSO.RRF")
                rel_name = self._member(bundle, "RXNREL.RRF")
                with bundle.open(conso_name) as handle:
                    conso = self._read_conso(io.TextIOWrapper(handle, encoding="utf-8"))
                if should_stop():
                    raise TerminologySourceUnavailable("RxNorm parse cancelled")
                with bundle.open(rel_name) as handle:
                    relations = self._read_relations(
                        io.TextIOWrapper(handle, encoding="utf-8"), conso
                    )
        except zipfile.BadZipFile as exc:
            raise TerminologySourceUnavailable(f"RxNorm archive is corrupt: {exc}") from exc
        return self._build_snapshot(conso, relations, version)

    # -------------------------------------------------------------------------
    @staticmethod
    def _member(bundle: zipfile.ZipFile, suffix: str) -> str:
        for name in bundle.namelist():
            if name.upper().endswith(suffix):
                return name
        raise TerminologySourceUnavailable(f"RxNorm archive lacks {suffix}")

    # -------------------------------------------------------------------------
    @staticmethod
    def _read_conso(lines: io.TextIOWrapper) -> _ConceptTable:
        table = _ConceptTable()
        for line in lines:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 17 or fields[11] != "RXNORM" or fields[16] not in {"N", ""}:
                continue
            rxcui, tty, text = fields[0], fields[12], fields[14]
            if tty in CONCEPT_TYPES:
                table.concepts[rxcui] = text
                table.types[rxcui] = tty
            elif tty == "BN":
                table.brands[rxcui] = text
            elif tty in SYNONYM_TYPES:
                table.synonyms[rxcui].append(text)
            elif tty in PRODUCT_TYPES:
                table.products[rxcui] = text
        return table

    # -------------------------------------------------------------------------
    @staticmethod
    def _read_relations(lines: io.TextIOWrapper, conso: _ConceptTable) -> _RelationTable:
        types = conso.types
        table = _RelationTable()
        for line in lines:
            fields = line.rstrip("\n").split("|")
            if len(fields) < 11 or fields[10] != "RXNORM":
                continue
            target, source, rela = fields[0], fields[4], fields[7]
            if rela == "form_of" and types.get(source) == "PIN" and types.get(target) == "IN":
                table.parents[source].add(target)
            elif rela == "part_of" and types.get(source) == "IN" and types.get(target) == "MIN":
                table.components[target].add(source)
            elif rela == "has_tradename" and types.get(source) == "IN" and target in conso.brands:
                table.brand_ingredients[target].add(source)
        return table

    # -------------------------------------------------------------------------
    def _build_snapshot(
        self,
        conso: _ConceptTable,
        relations: _RelationTable,
        version: str | None,
    ) -> TerminologySnapshot:
        snapshot = TerminologySnapshot(source=self.key, version=version, license=self.license)
        names = conso.concepts
        types = conso.types
        for rxcui, name in names.items():
            parents = relations.parents.get(rxcui, set())
            snapshot.concepts[rxcui] = StagedConcept(
                key=rxcui,
                name=name,
                concept_type=CONCEPT_TYPES[types[rxcui]],
                rxcui=rxcui,
                parent_key=next(iter(parents)) if len(parents) == 1 else None,
                component_keys=tuple(sorted(relations.components.get(rxcui, ()))),
            )
            snapshot.add_alias(rxcui, name, "canonical", term_type=types[rxcui])
        for rxcui, values in conso.synonyms.items():
            for value in values:
                snapshot.add_alias(rxcui, value, "synonym", term_type="SY")

        by_name = {name.casefold(): rxcui for rxcui, name in names.items()}
        min_by_components = {
            concept.component_keys: concept.key
            for concept in snapshot.concepts.values()
            if concept.concept_type == "multi_ingredient" and concept.component_keys
        }
        brand_sets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        products = conso.products
        brand_names = conso.brands
        brand_by_name = {name: rxcui for rxcui, name in brand_names.items()}
        for rxcui, text in products.items():
            product = self._parse_product(rxcui, text, by_name)
            if product is None:
                snapshot.stats["unparsed_products"] = snapshot.stats.get("unparsed_products", 0) + 1
                continue
            snapshot.products.append(product)
            if product.brand_name and product.brand_name in brand_by_name:
                ingredient_set = self._ingredient_set(product, snapshot)
                if ingredient_set:
                    brand_sets[brand_by_name[product.brand_name]].add(ingredient_set)

        for brand_rxcui, brand_name in brand_names.items():
            sets = brand_sets.get(brand_rxcui)
            if not sets:
                fallback = tuple(sorted(relations.brand_ingredients.get(brand_rxcui, ())))
                sets = {fallback} if fallback else set()
            for ingredient_set in sets:
                target = (
                    ingredient_set[0]
                    if len(ingredient_set) == 1
                    else min_by_components.get(ingredient_set)
                )
                if target is None:
                    snapshot.stats["brands_without_concept"] = (
                        snapshot.stats.get("brands_without_concept", 0) + 1
                    )
                    continue
                snapshot.add_alias(
                    target,
                    brand_name,
                    "brand",
                    term_type="BN",
                    country="US",
                    source_record_id=brand_rxcui,
                )
        snapshot.stats.update(
            concepts=len(snapshot.concepts),
            aliases=len(snapshot.aliases),
            products=len(snapshot.products),
        )
        return snapshot

    # -------------------------------------------------------------------------
    @staticmethod
    def _ingredient_set(product: StagedProduct, snapshot: TerminologySnapshot) -> tuple[str, ...]:
        keys: set[str] = set()
        for ingredient in product.ingredients:
            if ingredient.concept_key is None:
                return ()
            concept = snapshot.concepts.get(ingredient.concept_key)
            if concept is None:
                return ()
            keys.add(concept.parent_key or concept.key)
        return tuple(sorted(keys))

    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_product(
        rxcui: str,
        text: str,
        by_name: dict[str, str],
    ) -> StagedProduct | None:
        brand_match = _BRAND_RE.search(text)
        brand = brand_match.group(1).strip() if brand_match else None
        body = _BRAND_RE.sub("", text).strip()
        body = _QUANTITY_PREFIX_RE.sub("", body)
        ingredients: list[StagedProductIngredient] = []
        dose_form: str | None = None
        parts = body.split(" / ")
        for index, part in enumerate(parts):
            match = _COMPONENT_RE.match(part.strip())
            if match is None:
                return None
            name = match.group("name").strip()
            concept_key = by_name.get(name.casefold())
            if concept_key is None:
                return None
            ingredients.append(
                StagedProductIngredient(
                    ingredient_name=name,
                    concept_key=concept_key,
                    strength_value=float(match.group("value")),
                    strength_unit=match.group("unit").casefold(),
                )
            )
            if index == len(parts) - 1:
                dose_form = (match.group("rest") or "").strip() or None
        return StagedProduct(
            product_id=rxcui,
            name=text,
            brand_name=brand,
            ingredients=ingredients,
            country="US",
            dose_form=dose_form,
        )
