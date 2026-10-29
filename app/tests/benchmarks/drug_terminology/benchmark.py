# Copyright © 2023–2025 Thomas Virdis
# Licensed under the GNU General Public License, version 3 or later.

"""Labeled drug-terminology benchmark: case loading, scoring and metrics.

Labels are author-curated from standard nomenclature and still require
clinical review; results measure this benchmark only and are not a clinical
validation claim.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent
CASES_PATH = BENCHMARK_DIR / "cases.jsonl"
EQUIVALENTS_PATH = BENCHMARK_DIR / "equivalents.json"
CALIBRATION_BUCKETS = 6
COMBINATION_SPLIT = re.compile(r"\s*(?:/|\+|,|\band\b|\bwith\b)\s*")
HYPHEN_SPLIT = re.compile(r"\s*-\s*")
NON_WORD = re.compile(r"[^a-z0-9]+")


###############################################################################
@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    input: str
    expect: str
    ingredients: tuple[str, ...]
    alias_type: str
    language: str
    error_category: str

    # -------------------------------------------------------------------------
    @property
    def split(self) -> str:
        digest = hashlib.sha256(self.case_id.encode("utf-8")).digest()
        return "calibration" if digest[0] % 10 < CALIBRATION_BUCKETS else "heldout"


###############################################################################
@dataclass
class Prediction:
    accepted: bool
    status: str
    ingredients: list[str]
    candidates: list[list[str]] = field(default_factory=list)
    latency_ms: float = 0.0
    external_calls: int = 0
    method: str | None = None


ResolveFn = Callable[[str], Prediction]


###############################################################################
class IngredientCanonicalizer:

    # -------------------------------------------------------------------------
    def __init__(self, equivalents_path: Path = EQUIVALENTS_PATH) -> None:
        payload = json.loads(equivalents_path.read_text(encoding="utf-8"))
        self.salt_tokens = {self.fold(token) for token in payload["salt_tokens"]}
        self.preferred: dict[str, str] = {}
        for group in payload["groups"]:
            folded = [self.fold(name) for name in group]
            for name in folded:
                self.preferred[name] = folded[0]

    # -------------------------------------------------------------------------
    @staticmethod
    def fold(value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value.casefold())
        ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        return " ".join(NON_WORD.sub(" ", ascii_text).split())

    # -------------------------------------------------------------------------
    def canonical(self, name: str) -> str:
        folded = self.fold(name)
        if folded in self.preferred:
            return self.preferred[folded]
        tokens = folded.split()
        stripped = [token for token in tokens if token not in self.salt_tokens]
        if stripped and len(stripped) < len(tokens):
            folded = " ".join(stripped)
        return self.preferred.get(folded, folded)

    # -------------------------------------------------------------------------
    def ingredient_set(self, names: Iterable[str]) -> frozenset[str]:
        result: set[str] = set()
        for name in names:
            for part in self.split_combination(name):
                canonical = self.canonical(part)
                if canonical:
                    result.add(canonical)
        return frozenset(result)

    # -------------------------------------------------------------------------
    def split_combination(self, name: str) -> list[str]:
        parts = [part for part in COMBINATION_SPLIT.split(name) if part.strip()]
        expanded: list[str] = []
        for part in parts:
            pieces = [piece for piece in HYPHEN_SPLIT.split(part) if piece.strip()]
            if len(pieces) > 1 and all(len(self.fold(piece)) >= 4 for piece in pieces):
                expanded.extend(pieces)
            else:
                expanded.append(part)
        return expanded


# -----------------------------------------------------------------------------
def load_cases(path: Path = CASES_PATH) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(
            BenchmarkCase(
                case_id=row["id"],
                input=row["input"],
                expect=row["expect"],
                ingredients=tuple(row["ingredients"]),
                alias_type=row["alias_type"],
                language=row["language"],
                error_category=row["error_category"],
            )
        )
    return cases


# -----------------------------------------------------------------------------
def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


###############################################################################
@dataclass
class CaseOutcome:
    case: BenchmarkCase
    prediction: Prediction
    top1_correct: bool
    in_top5: bool
    auto_correct: bool
    auto_wrong: bool

    # -------------------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.case.case_id,
            "input": self.case.input,
            "expect": self.case.expect,
            "expected": list(self.case.ingredients),
            "split": self.case.split,
            "status": self.prediction.status,
            "method": self.prediction.method,
            "accepted": self.prediction.accepted,
            "predicted": self.prediction.ingredients,
            "candidates": self.prediction.candidates[:5],
            "top1_correct": self.top1_correct,
            "auto_wrong": self.auto_wrong,
            "latency_ms": round(self.prediction.latency_ms, 3),
            "external_calls": self.prediction.external_calls,
        }


# -----------------------------------------------------------------------------
def score_case(
    case: BenchmarkCase,
    prediction: Prediction,
    canonicalizer: IngredientCanonicalizer,
) -> CaseOutcome:
    expected = canonicalizer.ingredient_set(case.ingredients)
    predicted = canonicalizer.ingredient_set(prediction.ingredients)
    candidate_sets = [
        canonicalizer.ingredient_set(candidate)
        for candidate in prediction.candidates[:5]
    ]
    if predicted and predicted not in candidate_sets:
        candidate_sets.insert(0, predicted)
    should_accept = case.expect == "accept"
    top1_correct = should_accept and bool(predicted) and predicted == expected
    in_top5 = should_accept and expected in candidate_sets[:5]
    auto_correct = prediction.accepted and top1_correct
    auto_wrong = prediction.accepted and not top1_correct
    return CaseOutcome(case, prediction, top1_correct, in_top5, auto_correct, auto_wrong)


# -----------------------------------------------------------------------------
def summarize(outcomes: list[CaseOutcome]) -> dict[str, Any]:
    accept_cases = [item for item in outcomes if item.case.expect == "accept"]
    abstain_cases = [item for item in outcomes if item.case.expect == "abstain"]
    auto = [item for item in outcomes if item.prediction.accepted]
    latencies = [item.prediction.latency_ms for item in outcomes]

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    return {
        "cases": len(outcomes),
        "accept_cases": len(accept_cases),
        "abstain_cases": len(abstain_cases),
        "top1_accuracy": ratio(sum(i.top1_correct for i in accept_cases), len(accept_cases)),
        "recall_at_5": ratio(sum(i.in_top5 for i in accept_cases), len(accept_cases)),
        "automatic_matches": len(auto),
        "automatic_match_precision": ratio(sum(i.auto_correct for i in auto), len(auto)),
        "false_automatic_matches": sum(i.auto_wrong for i in outcomes),
        "false_automatic_match_rate": ratio(sum(i.auto_wrong for i in outcomes), len(outcomes)),
        "abstention_recall": ratio(
            sum(not i.prediction.accepted for i in abstain_cases), len(abstain_cases)
        ),
        "automatic_coverage": ratio(sum(i.auto_correct for i in accept_cases), len(accept_cases)),
        "latency_ms_median": round(statistics.median(latencies), 3) if latencies else None,
        "latency_ms_p95": round(percentile(latencies, 0.95), 3),
        "external_calls_total": sum(i.prediction.external_calls for i in outcomes),
        "external_calls_per_resolution": ratio(
            sum(i.prediction.external_calls for i in outcomes), len(outcomes)
        ),
    }


# -----------------------------------------------------------------------------
def breakdown(outcomes: list[CaseOutcome], attribute: str) -> dict[str, Any]:
    groups: dict[str, list[CaseOutcome]] = {}
    for item in outcomes:
        groups.setdefault(str(getattr(item.case, attribute)), []).append(item)
    return {
        key: {
            "cases": len(items),
            "top1": sum(i.top1_correct for i in items),
            "auto_correct": sum(i.auto_correct for i in items),
            "auto_wrong": sum(i.auto_wrong for i in items),
            "not_accepted": sum(not i.prediction.accepted for i in items),
        }
        for key, items in sorted(groups.items())
    }


# -----------------------------------------------------------------------------
def run_benchmark(
    resolve: ResolveFn,
    *,
    cases: list[BenchmarkCase] | None = None,
    split: str = "all",
) -> dict[str, Any]:
    canonicalizer = IngredientCanonicalizer()
    selected = [
        case
        for case in (cases or load_cases())
        if split == "all" or case.split == split
    ]
    outcomes = [score_case(case, resolve(case.input), canonicalizer) for case in selected]
    report: dict[str, Any] = {"split": split, "overall": summarize(outcomes)}
    for name, subset in (
        ("calibration", [o for o in outcomes if o.case.split == "calibration"]),
        ("heldout", [o for o in outcomes if o.case.split == "heldout"]),
    ):
        if subset and split == "all":
            report[name] = summarize(subset)
    report["by_alias_type"] = breakdown(outcomes, "alias_type")
    report["by_language"] = breakdown(outcomes, "language")
    report["by_error_category"] = breakdown(outcomes, "error_category")
    report["false_automatic_matches"] = [
        outcome.to_json() for outcome in outcomes if outcome.auto_wrong
    ]
    report["cases"] = [outcome.to_json() for outcome in outcomes]
    return report
