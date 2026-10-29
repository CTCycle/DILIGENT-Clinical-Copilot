# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Deterministic drug-name normalization shared by ingestion and resolution.

Superficial differences (Unicode, case, punctuation, whitespace, strength and
unit formatting) are folded into a lookup key. Clinically meaningful details
are preserved separately: strengths and dose-form, route or release qualifiers
are returned alongside the key, and salt words stay inside the key because a
salt can denote a distinct precise ingredient.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from common.utils.text_utils import normalize_drug_name
from services.text.vocabulary import get_text_normalization_snapshot

# A strength may carry a rate or concentration denominator (mcg/h, mg/ml).
_STRENGTH_RE = re.compile(
    r"(?<![a-z0-9])(\d+(?:[.,]\d+)?)\s*"
    r"(mg|mcg|µg|μg|ug|g|kg|ml|l|iu|ui|u|mmol|meq|%)(?![a-z])"
    r"(?:\s*/\s*(?:\d+\s*)?(ml|hr|h|kg|g|l|dose)(?![a-z]))?",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_SCHEDULE_RE = re.compile(r"^\d+(?:[.,]\d+)?(?:-\d+(?:[.,]\d+)?){1,3}$")
_COMBINATION_SPLIT_RE = re.compile(r"\s*(?:/|\+|&|\band\b|\bwith\b|\bund\b|\bet\b|\be\b|\by\b)\s*")
_HYPHEN_RE = re.compile(r"\s*-\s*")
_CONNECTOR_WORDS = frozenset({"and", "or", "with", "without"})
_UNIT_ALIASES = {
    "µg": "mcg",
    "μg": "mcg",
    "ug": "mcg",
    "ui": "iu",
    "u": "iu",
    "h": "hr",
}
_QUALIFIER_WORDS = frozenset(
    {
        # dose forms
        "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
        "film", "coated", "filmtabletten", "filmtablette", "tabletten", "tablette",
        "kapseln", "kapsel", "comprimes", "comprime", "compresse", "compressa",
        "cpr", "solution", "suspension", "syrup", "sirup", "drops", "gtt",
        "injection", "inj", "infusion", "cream", "ointment", "gel", "patch",
        "suppository", "inhaler", "spray", "powder", "granules", "sachet",
        "tabl", "filmtabl", "kaps", "supp", "amp", "susp", "lsg",
        # routes
        "oral", "po", "iv", "im", "sc", "sl", "topical", "rectal", "inhaled",
        "transdermal", "ophthalmic", "nasal",
        # release characteristics
        "er", "xr", "xl", "sr", "cr", "ir", "dr", "la", "retard", "mups", "zok",
        "extended", "release", "delayed", "modified", "prolonged", "depot",
        # administration frequency
        "daily", "weekly", "monthly", "bid", "tid", "qid", "qd", "qhs", "prn", "od",
        "bd", "tds", "nocte", "mane", "once", "twice", "hourly", "stat",
    }
)
_SKELETON_RULES = (("ph", "f"), ("th", "t"), ("ch", "c"), ("ae", "e"), ("oe", "e"), ("y", "i"), ("k", "c"))
_RELEASE_WORDS = frozenset(
    {"er", "xr", "xl", "sr", "cr", "dr", "la", "retard", "zok", "extended", "delayed",
     "modified", "prolonged", "depot"}
)

# Latin pharmacopoeial salt and hydrate words (Swissmedic composition style).
_LATIN_CATIONS = {
    "natricum": "sodium",
    "natrii": "sodium",
    "natrium": "sodium",
    "dinatricum": "disodium",
    "kalicum": "potassium",
    "kalii": "potassium",
    "kalium": "potassium",
    "calcicum": "calcium",
    "calcii": "calcium",
    "magnesicum": "magnesium",
    "magnesii": "magnesium",
}
_LATIN_ANIONS = {
    "hydrochloridum": "hydrochloride",
    "hydrobromidum": "hydrobromide",
    "dihydrochloridum": "dihydrochloride",
}
_LATIN_HYDRATES = {
    "trihydricum": "trihydrate",
    "dihydricum": "dihydrate",
    "monohydricum": "monohydrate",
    "hemihydricum": "hemihydrate",
    "sesquihydricum": "sesquihydrate",
    "anhydricum": "anhydrous",
}
_LATIN_QUALIFIERS = ("ut", "corresp", "correspondens")


# -----------------------------------------------------------------------------
def _latin_word(token: str, table: dict[str, str]) -> str | None:
    """Match a Latin salt word regardless of adjective ending (-um, -us, -a)."""
    if token in table:
        return table[token]
    for suffix in ("us", "a"):
        if token.endswith(suffix):
            candidate = token[: -len(suffix)] + "um"
            if candidate in table:
                return table[candidate]
    return None


###############################################################################
@dataclass(frozen=True)
class Strength:
    value: float
    unit: str

    # -------------------------------------------------------------------------
    def label(self) -> str:
        value = int(self.value) if self.value.is_integer() else self.value
        return f"{value} {self.unit}"


###############################################################################
@dataclass(frozen=True)
class NormalizedDrugInput:
    original: str
    raw_key: str
    key: str
    full_key: str
    strengths: tuple[Strength, ...]
    qualifiers: tuple[str, ...]
    component_keys: tuple[str, ...]

    # -------------------------------------------------------------------------
    @property
    def release_qualifiers(self) -> tuple[str, ...]:
        return tuple(token for token in self.qualifiers if token in _RELEASE_WORDS)


# -----------------------------------------------------------------------------
def fold(value: str | None) -> str:
    """Fold Unicode, case and punctuation into the canonical alias key."""
    if not value:
        return ""
    return normalize_drug_name(unicodedata.normalize("NFKC", value))


# -----------------------------------------------------------------------------
def parse_strengths(value: str) -> tuple[Strength, ...]:
    strengths: list[Strength] = []
    for number, unit, per in _STRENGTH_RE.findall(value):
        unit_key = unit.casefold()
        label = _UNIT_ALIASES.get(unit_key) or unit_key
        if per:
            per_key = per.casefold()
            label = f"{label}/{_UNIT_ALIASES.get(per_key, per_key)}"
        strengths.append(Strength(float(number.replace(",", ".")), label))
    return tuple(strengths)


# -----------------------------------------------------------------------------
def _ignored_tokens() -> frozenset[str]:
    snapshot = get_text_normalization_snapshot()
    return (
        (snapshot.formulation_stopwords - _CONNECTOR_WORDS)
        | snapshot.trailing_temporal_tokens
        | snapshot.manufacturer_tokens
    )


# -----------------------------------------------------------------------------
def _key_tokens(value: str, ignored: frozenset[str]) -> tuple[list[str], list[str], list[str]]:
    """Return (key tokens, qualifier tokens, key tokens with qualifiers in place)."""
    kept: list[str] = []
    qualifiers: list[str] = []
    full: list[str] = []
    for token in fold(value).split():
        if _NUMERIC_RE.fullmatch(token) or _SCHEDULE_RE.fullmatch(token):
            continue
        if token in _UNIT_ALIASES or token in {"mg", "mcg", "g", "kg", "ml", "l", "iu", "mmol", "meq"}:
            continue
        if token in _QUALIFIER_WORDS:
            qualifiers.append(token)
            full.append(token)
            continue
        if token in ignored:
            continue
        kept.append(token)
        full.append(token)
    return kept, qualifiers, full


# -----------------------------------------------------------------------------
def normalize_drug_input(value: str) -> NormalizedDrugInput:
    """Build the lookup key while keeping strengths and qualifiers separate."""
    original = value
    text = unicodedata.normalize("NFKC", value or "")
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text)
    strengths = parse_strengths(text)
    without_strengths = _STRENGTH_RE.sub(" ", text)
    ignored = _ignored_tokens()
    kept, qualifiers, full = _key_tokens(without_strengths, ignored)
    components: list[str] = []
    parts = [part for part in _COMBINATION_SPLIT_RE.split(without_strengths.casefold()) if part.strip()]
    if len(parts) == 1:
        hyphen_parts = [part for part in _HYPHEN_RE.split(parts[0]) if part.strip()]
        if len(hyphen_parts) > 1 and all(len(fold(part)) >= 4 for part in hyphen_parts):
            parts = hyphen_parts
    if len(parts) > 1:
        for part in parts:
            part_tokens, _, _ = _key_tokens(part, ignored)
            if part_tokens:
                components.append(" ".join(part_tokens))
    return NormalizedDrugInput(
        original=original,
        raw_key=fold(text),
        key=" ".join(kept),
        full_key=" ".join(full),
        strengths=strengths,
        qualifiers=tuple(qualifiers),
        component_keys=tuple(components) if len(components) > 1 else (),
    )


# -----------------------------------------------------------------------------
def orthographic_skeleton(key: str) -> str:
    """Collapse spelling conventions that differ across languages (ph/f, th/t,
    y/i, k/c, doubled letters). Used only to look up exact aliases."""
    skeleton = key
    for old, new in _SKELETON_RULES:
        skeleton = skeleton.replace(old, new)
    collapsed: list[str] = []
    for char in skeleton:
        if collapsed and collapsed[-1] == char and char.isalpha():
            continue
        collapsed.append(char)
    return "".join(collapsed)


# -----------------------------------------------------------------------------
def _latin_stem_options(token: str) -> list[str]:
    if token.endswith("um") and len(token) > 4:
        return [token[:-2], token[:-2] + "e"]
    if token.endswith("i") and len(token) > 4:
        return [token[:-1], token[:-1] + "e"]
    return [token]


# -----------------------------------------------------------------------------
def latin_to_english_variants(key: str) -> list[str]:
    """Translate a folded Latin pharmacopoeial name into English candidates.

    Candidates are ordered from most to least specific (salt and hydrate kept
    first, bare moiety last). Each one is only a lookup key: acceptance still
    requires an exact match against verified terminology.
    """
    tokens = key.split()
    for marker in _LATIN_QUALIFIERS:
        if marker in tokens:
            tokens = tokens[: tokens.index(marker)]
    if not tokens:
        return []
    if tokens[0] == "acidum" and len(tokens) >= 2:
        stems = [token[:-2] if token.endswith("um") else token for token in tokens[1:]]
        return [" ".join(stems) + " acid"]
    cations: list[str] = []
    anions: list[str] = []
    hydrates: list[str] = []
    bases: list[str] = []
    for token in tokens:
        cation = _latin_word(token, _LATIN_CATIONS)
        hydrate = _latin_word(token, _LATIN_HYDRATES)
        anion = _latin_word(token, _LATIN_ANIONS)
        if cation:
            cations.append(cation)
        elif hydrate:
            hydrates.append(hydrate)
        elif anion:
            anions.append(anion)
        elif token.endswith("as") and len(token) > 4:
            anions.append(token[:-2] + "ate")
        else:
            bases.append(token)
    variants: list[str] = []
    if not bases:
        if len(anions) != 1:
            return []
        anion = anions[0]
        for cation in cations:
            variants.extend([f"{anion} {cation}", f"{cation} {anion}"])
        variants.append(anion)
        return list(dict.fromkeys(variants))
    salts = [*anions, *cations]
    for combination in _product([_latin_stem_options(token) for token in bases]):
        base = " ".join(combination)
        if salts and hydrates:
            variants.append(" ".join([base, *salts, *hydrates]))
        if salts:
            variants.append(" ".join([base, *salts]))
        if hydrates and not salts:
            variants.append(" ".join([base, *hydrates]))
    for combination in _product([_latin_stem_options(token) for token in bases]):
        variants.append(" ".join(combination))
    return list(dict.fromkeys(variant for variant in variants if variant and variant != key))


# -----------------------------------------------------------------------------
def language_variants(key: str) -> list[str]:
    """Curated orthographic variants for common European spelling conventions."""
    variants: list[str] = list(latin_to_english_variants(key))
    tokens = key.split()
    if not tokens:
        return variants
    acid_rules = (
        ("acido", "ico"),
        ("acide", "ique"),
    )
    for prefix, suffix in acid_rules:
        if len(tokens) == 2 and tokens[0] == prefix and tokens[1].endswith(suffix):
            variants.append(tokens[1][: -len(suffix)] + "ic acid")
    if len(tokens) == 1:
        word = tokens[0]
        for suffix, replacement in (("insaure", "ic acid"), ("saure", "ic acid")):
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                variants.append(word[: -len(suffix)] + replacement)
                break
    last = tokens[-1]
    prefix = " ".join(tokens[:-1])
    endings: list[str] = []
    if len(last) >= 6:
        if last.endswith("e"):
            endings.append(last[:-1])
        if last.endswith(("a", "o")):
            endings.extend([last[:-1], last[:-1] + "e"])
        if last[-1] not in "aeo":
            endings.append(last + "e")
    for ending in endings:
        variants.append(f"{prefix} {ending}".strip())
    return list(dict.fromkeys(variant for variant in variants if variant and variant != key))


# -----------------------------------------------------------------------------
def _product(options: list[list[str]]) -> list[tuple[str, ...]]:
    combinations: list[tuple[str, ...]] = [()]
    for choices in options:
        combinations = [(*prefix, choice) for prefix in combinations for choice in choices]
    return combinations
