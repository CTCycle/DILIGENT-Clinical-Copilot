from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from services.catalogs.runtime import get_reference_catalog_snapshot


@dataclass(frozen=True)
class DrugBlock:
    text: str
    start: int
    end: int


BULLET_RE = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+")
UPPER_TOKEN_RE = re.compile(r"^[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'/-]+")


@lru_cache(maxsize=1)
def _metadata_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    terms = list(snapshot.values("clinical_extraction", "drug_metadata_labels"))
    terms.extend(snapshot.values("clinical_extraction", "drug_dosage_units"))
    terms.extend(snapshot.values("clinical_extraction", "drug_frequency_terms"))
    terms.extend(snapshot.values("clinical_extraction", "drug_route_terms"))
    escaped = [re.escape(term) for term in terms if term.strip()]
    if not escaped:
        return re.compile(r"$^")
    return re.compile(
        r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])(?:" + "|".join(escaped) + r")\b",
        re.IGNORECASE,
    )


@lru_cache(maxsize=1)
def _continuation_prefix_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    terms = list(snapshot.values("clinical_extraction", "drug_continuation_markers"))
    escaped = [re.escape(term) + r"\b" for term in terms if term.strip()]
    prefix_body = "|".join(escaped) if escaped else r"$^"
    return re.compile(
        r"^(?:" + prefix_body + r"|peso\b|\d+(?:[.,]\d+)?\s*kg\b)",
        re.IGNORECASE,
    )


@lru_cache(maxsize=1)
def _regimen_split_re() -> re.Pattern[str]:
    snapshot = get_reference_catalog_snapshot()
    separators = [
        value.strip().lower()
        for value in snapshot.values("clinical_extraction", "drug_regimen_separators")
        if value
    ]
    separator_map = {
        "semicolon": r";",
        "plus": r"\+|\s+plus\s+",
        "slash": r"/",
    }
    escaped = [separator_map[value] for value in separators if value in separator_map]
    if not escaped:
        return re.compile(r"$^")
    return re.compile(r"(?:%s)" % "|".join(escaped), re.IGNORECASE)


def _likely_drug_start(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if _continuation_prefix_re().search(text):
        return False
    if not UPPER_TOKEN_RE.search(text):
        return False
    return bool(_metadata_re().search(text))


def isolate_drug_blocks(text: str) -> list[DrugBlock]:
    source = text or ""
    if not source.strip():
        return []
    bullet_matches = list(BULLET_RE.finditer(source))
    if bullet_matches:
        blocks: list[DrugBlock] = []
        for index, match in enumerate(bullet_matches):
            start = match.start()
            end = bullet_matches[index + 1].start() if index + 1 < len(bullet_matches) else len(source)
            raw = source[start:end].strip()
            if raw:
                blocks.append(DrugBlock(text=raw, start=start, end=end))
        return blocks or [DrugBlock(text=source.strip(), start=0, end=len(source))]

    lines = [line.strip() for line in source.splitlines() if line.strip()]
    likely_start_count = sum(1 for line in lines if _likely_drug_start(line))
    if len(lines) > 1 and likely_start_count >= 2:
        blocks: list[DrugBlock] = []
        current_lines: list[str] = []
        current_start: int | None = None
        current_end: int | None = None
        cursor = 0
        for line in lines:
            pos = source.find(line, cursor)
            if pos < 0:
                continue
            line_end = pos + len(line)
            if _likely_drug_start(line):
                if current_lines and current_start is not None and current_end is not None:
                    blocks.append(
                        DrugBlock(
                            text="\n".join(current_lines),
                            start=current_start,
                            end=current_end,
                        )
                    )
                current_lines = [line]
                current_start = pos
                current_end = line_end
            elif current_lines:
                current_lines.append(line)
                current_end = line_end
            cursor = line_end
        if current_lines and current_start is not None and current_end is not None:
            blocks.append(
                DrugBlock(
                    text="\n".join(current_lines),
                    start=current_start,
                    end=current_end,
                )
            )
        if blocks:
            return blocks

    regimen_split = _regimen_split_re()
    if regimen_split.pattern != r"$^":
        parts = [part.strip() for part in regimen_split.split(source)]
        if len(parts) > 1 and all(_likely_drug_start(part) for part in parts if part):
            blocks: list[DrugBlock] = []
            cursor = 0
            for part in parts:
                if not part:
                    continue
                pos = source.find(part, cursor)
                if pos < 0:
                    continue
                blocks.append(DrugBlock(text=part, start=pos, end=pos + len(part)))
                cursor = pos + len(part)
            if blocks:
                return blocks

    period_split = re.split(r"(?<=\.)\s+(?=[A-ZÀ-ÖØ-Þ])", source.strip())
    likely_period_parts = [part for part in period_split if _likely_drug_start(part)]
    if len(likely_period_parts) >= 2:
        blocks = []
        cursor = 0
        for part in period_split:
            if not _likely_drug_start(part):
                continue
            pos = source.find(part, cursor)
            if pos < 0:
                continue
            blocks.append(DrugBlock(text=part.strip(), start=pos, end=pos + len(part)))
            cursor = pos + len(part)
        if blocks:
            return blocks

    return [DrugBlock(text=source.strip(), start=0, end=len(source))]
