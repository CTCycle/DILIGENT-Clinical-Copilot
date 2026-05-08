from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class DrugBlock:
    text: str
    start: int
    end: int


BULLET_RE = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+")
UPPER_TOKEN_RE = re.compile(r"^[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'/-]+")
METADATA_RE = re.compile(
    r"\b(?:mg|mcg|g|ui|iu|po|ev|iv|im|sc|bid|tid|die|volta\/die|sospes[oa]|continu[ae]|cronic[ao]|iniziat[oa])\b",
    re.IGNORECASE,
)


def _likely_drug_start(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if not UPPER_TOKEN_RE.search(text):
        return False
    return bool(METADATA_RE.search(text))


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
    if len(lines) > 1 and all(_likely_drug_start(line) for line in lines):
        blocks: list[DrugBlock] = []
        cursor = 0
        for line in lines:
            pos = source.find(line, cursor)
            if pos < 0:
                continue
            blocks.append(DrugBlock(text=line, start=pos, end=pos + len(line)))
            cursor = pos + len(line)
        if blocks:
            return blocks

    if ";" in source:
        parts = [part.strip() for part in source.split(";")]
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
    if len(period_split) > 1 and all(_likely_drug_start(part) for part in period_split if part):
        blocks = []
        cursor = 0
        for part in period_split:
            pos = source.find(part, cursor)
            if pos < 0:
                continue
            blocks.append(DrugBlock(text=part.strip(), start=pos, end=pos + len(part)))
            cursor = pos + len(part)
        if blocks:
            return blocks

    return [DrugBlock(text=source.strip(), start=0, end=len(source))]
