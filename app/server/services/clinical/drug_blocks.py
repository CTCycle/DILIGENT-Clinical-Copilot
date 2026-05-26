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
    r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])(?:mg|mcg|g|ml|ui|iu|po|ev|iv|im|sc|bid|tid|die|volta\/die|sospes[oa]|continu[ae]|cronic[ao]|iniziat[oa]|started|stopped)\b",
    re.IGNORECASE,
)
CONTINUATION_PREFIX_RE = re.compile(
    r"^(?:dal\b|da\b|dall['’]|se\b|in\s+riserva\b|peso\b|\d+(?:[.,]\d+)?\s*kg\b)",
    re.IGNORECASE,
)


def _likely_drug_start(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    if CONTINUATION_PREFIX_RE.search(text):
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
