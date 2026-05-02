from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from typing import Any

from configurations.llm_configs import LLMRuntimeConfig
from configurations.startup import server_settings
from domain.clinical.entities import ClinicalSectionExtractionResult
from services.llm.client_runtime import ensure_runtime_client
from services.llm.prompts import CLINICAL_SECTION_EXTRACTION_PROMPT
from services.llm.providers import select_llm_provider


###############################################################################
class ClinicalInputExtractionError(RuntimeError):
    pass


###############################################################################
class ClinicalInputExtractor:
    SECTION_ALIASES: dict[str, tuple[str, ...]] = {
        "anamnesis": (
            "anamnesis",
            "anamnesi",
            "anamnese",
            "antecedentes",
            "historia clinica",
            "clinical history",
            "history",
            "storia clinica",
            "storia",
            "anamnesis and history",
        ),
        "drugs": (
            "therapy",
            "current drugs",
            "drugs",
            "medicamentos",
            "medicamento",
            "medikamente",
            "medications",
            "medication",
            "farmaci",
            "farmaci attuali",
            "farmaci in uso",
            "terapia",
            "terapie",
        ),
        "laboratory_analysis": (
            "laboratory analysis",
            "laboratory",
            "lab results",
            "lab",
            "labs",
            "esami",
            "esami di laboratorio",
            "esami ematici",
            "laboratorio",
            "analisi laboratorio",
            "analisi di laboratorio",
        ),
    }
    MARKDOWN_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s*(?P<label>[^#\n]+?)\s*$")
    INDEXED_HEADER_RE = re.compile(
        r"^\s*(?:\d+|[ivxlcdm]+|[abc])[\)\.\-]\s*(?P<label>[^:\n]+)\s*:?\s*$",
        re.IGNORECASE,
    )
    INLINE_HEADER_RE = re.compile(
        r"^\s*(?P<label>[A-Za-zÀ-ÿ ]{3,40})\s*:\s*(?P<tail>.*)$"
    )
    WHITESPACE_RE = re.compile(r"\s+")

    def __init__(
        self,
        *,
        client: Any | None = None,
        timeout_s: float = server_settings.external_data.default_llm_timeout,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.client: Any | None = client
        self.model: str = ""
        self.client_lock = asyncio.Lock()
        self.client_loop_id: int | None = None
        self.forced_provider: str | None = None
        self.forced_model: str | None = None
        if client is None:
            self.client_provider: str | None = None
            self.runtime_revision = -1
        else:
            self.client_provider = "injected"
            self.runtime_revision = LLMRuntimeConfig.get_revision()

    # -------------------------------------------------------------------------
    async def ensure_client(self) -> None:
        revision = LLMRuntimeConfig.get_revision()
        resolved_provider, resolved_model = LLMRuntimeConfig.resolve_provider_and_model(
            "parser"
        )
        provider = self.forced_provider or resolved_provider
        model = self.forced_model or resolved_model
        await ensure_runtime_client(
            self,
            provider=provider,
            model=model,
            revision=revision,
            client_factory=lambda selected_provider, selected_model: (
                select_llm_provider(
                    provider=selected_provider,
                    default_model=selected_model,
                    timeout_s=self.timeout_s,
                )
            ),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _build_prompt(clinical_input: str) -> str:
        return (
            "Extract the three text blocks anamnesis, drugs, laboratory_analysis.\n"
            "Return valid JSON matching the provided schema, with plain text fields and an overall confidence score between 0 and 1.\n"
            "Do not include span coordinates.\n"
            "Use this exact source text:\n\n"
            f"{clinical_input}"
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _normalize_label(cls, value: str) -> str:
        normalized = re.sub(r"[\*\_`~\[\]\(\)]", " ", value or "")
        normalized = re.sub(r"[^A-Za-zÀ-ÿ0-9 ]+", " ", normalized)
        normalized = cls.WHITESPACE_RE.sub(" ", normalized).strip().lower()
        return normalized

    # -------------------------------------------------------------------------
    @classmethod
    def _match_section_key(cls, label: str) -> str | None:
        normalized = cls._normalize_label(label)
        if not normalized:
            return None
        for section_key, aliases in cls.SECTION_ALIASES.items():
            if normalized in aliases:
                return section_key
        return None

    # -------------------------------------------------------------------------
    @staticmethod
    def _split_blocks(text: str) -> list[str]:
        blocks = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        return blocks

    # -------------------------------------------------------------------------
    @classmethod
    def _score_block(cls, block: str) -> dict[str, int]:
        lowered = block.lower()
        scores = {"anamnesis": 0, "drugs": 0, "laboratory_analysis": 0}
        if any(
            token in lowered
            for token in (
                "alt",
                "ast",
                "alp",
                "ggt",
                "bilirubin",
                "bilirubina",
                "inr",
                "uln",
                "u/l",
                "umol/l",
                "prelievo",
                "follow-up",
            )
        ):
            scores["laboratory_analysis"] += 3
        if any(
            token in lowered
            for token in (
                " mg ",
                " ev",
                " iv",
                " po",
                " bid",
                "tid",
                "q8h",
                "farmac",
                "terap",
                "drug",
                "medication",
            )
        ):
            scores["drugs"] += 3
        if any(
            token in lowered
            for token in (
                "paziente",
                "patient",
                "anamnes",
                "history",
                "storia clinica",
                "symptom",
                "nausea",
                "ittero",
                "prurito",
            )
        ):
            scores["anamnesis"] += 2
        return scores

    # -------------------------------------------------------------------------
    @classmethod
    def _build_result(
        cls,
        *,
        source_text: str,
        sections: dict[str, list[str]],
        confidence: float,
    ) -> ClinicalSectionExtractionResult:
        return ClinicalSectionExtractionResult(
            source_text=source_text,
            anamnesis=cls._join_section_lines(sections, "anamnesis"),
            drugs=cls._join_section_lines(sections, "drugs"),
            laboratory_analysis=cls._join_section_lines(sections, "laboratory_analysis"),
            confidence=max(0.0, min(1.0, float(confidence))),
        )

    # -------------------------------------------------------------------------
    @staticmethod
    def _join_section_lines(sections: dict[str, list[str]], section_key: str) -> str | None:
        lines = [line.strip() for line in sections.get(section_key, []) if line.strip()]
        if not lines:
            return None
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    @classmethod
    def _deterministic_extract(
        cls,
        clinical_input: str,
    ) -> ClinicalSectionExtractionResult | None:
        source_text = clinical_input
        sections: dict[str, list[str]] = {
            "anamnesis": [],
            "drugs": [],
            "laboratory_analysis": [],
        }
        current_section: str | None = None
        explicit_markers = 0
        indexed_markers = 0
        inline_markers = 0
        found_any = False
        orphan_lines: list[str] = []

        for raw_line in clinical_input.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if not line:
                if current_section is not None and sections[current_section]:
                    sections[current_section].append("")
                continue
            header_match = cls.MARKDOWN_HEADER_RE.match(line)
            if header_match:
                maybe_section = cls._match_section_key(header_match.group("label"))
                if maybe_section is not None:
                    current_section = maybe_section
                    explicit_markers += 1
                    found_any = True
                    continue
            indexed_match = cls.INDEXED_HEADER_RE.match(line)
            if indexed_match:
                maybe_section = cls._match_section_key(indexed_match.group("label"))
                if maybe_section is not None:
                    current_section = maybe_section
                    indexed_markers += 1
                    found_any = True
                    continue
            inline_match = cls.INLINE_HEADER_RE.match(line)
            if inline_match:
                maybe_section = cls._match_section_key(inline_match.group("label"))
                if maybe_section is not None:
                    current_section = maybe_section
                    inline_markers += 1
                    found_any = True
                    tail = (inline_match.group("tail") or "").strip()
                    if tail:
                        sections[current_section].append(tail)
                    continue
            if current_section is not None:
                sections[current_section].append(line)
            else:
                orphan_lines.append(line)

        if orphan_lines:
            for line in orphan_lines:
                scores = cls._score_block(line)
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                if not ranked or ranked[0][1] <= 0:
                    continue
                sections[ranked[0][0]].append(line)

        if found_any:
            confidence = 0.92
            if explicit_markers == 0:
                confidence = 0.85
            if explicit_markers == 0 and indexed_markers == 0:
                confidence = 0.8
            return cls._build_result(
                source_text=source_text,
                sections=sections,
                confidence=confidence,
            )

        # Fallback deterministic split: classify high-level blocks by keywords.
        blocks = cls._split_blocks(clinical_input)
        if not blocks:
            return None
        assigned = False
        if len(blocks) == 1:
            for line in (
                part.strip()
                for part in clinical_input.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            ):
                if not line:
                    continue
                scores = cls._score_block(line)
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                if not ranked or ranked[0][1] <= 0:
                    continue
                sections[ranked[0][0]].append(line)
                assigned = True
            if assigned:
                return cls._build_result(
                    source_text=source_text,
                    sections=sections,
                    confidence=0.68,
                )
        for block in blocks:
            scores = cls._score_block(block)
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if not ranked or ranked[0][1] <= 0:
                continue
            sections[ranked[0][0]].append(block.strip())
            assigned = True
        if not assigned:
            return None
        return cls._build_result(
            source_text=source_text,
            sections=sections,
            confidence=0.65,
        )

    # -------------------------------------------------------------------------
    @classmethod
    def _is_source_text_compatible(cls, expected_text: str, received_text: str) -> bool:
        if received_text == expected_text:
            return True
        expected = expected_text.replace("\r\n", "\n").replace("\r", "\n")
        received = received_text.replace("\r\n", "\n").replace("\r", "\n")
        if received == expected:
            return True
        if received.rstrip() == expected.rstrip():
            return True
        expected_compact = cls.WHITESPACE_RE.sub("", expected)
        received_compact = cls.WHITESPACE_RE.sub("", received)
        if expected_compact != received_compact:
            return False
        # Allow only tiny formatting drift (trailing newline, extra spaces, etc.).
        return abs(len(expected) - len(received)) <= 8

    # -------------------------------------------------------------------------
    async def extract(
        self,
        *,
        clinical_input: str,
        progress_callback: Callable[[float], None] | None = None,
    ) -> ClinicalSectionExtractionResult:
        if not clinical_input.strip():
            raise ClinicalInputExtractionError("clinical_input is empty")

        if progress_callback is not None:
            progress_callback(0.0)

        deterministic = self._deterministic_extract(clinical_input)
        if deterministic is not None and (
            deterministic.anamnesis is not None
            or deterministic.drugs is not None
            or deterministic.laboratory_analysis is not None
        ):
            if progress_callback is not None:
                progress_callback(1.0)
            return deterministic

        await self.ensure_client()
        if self.client is None:
            raise ClinicalInputExtractionError(
                "LLM client is not initialized for input extraction"
            )

        try:
            extraction = await self.client.llm_structured_call(
                model=self.model,
                system_prompt=CLINICAL_SECTION_EXTRACTION_PROMPT,
                user_prompt=self._build_prompt(clinical_input),
                schema=ClinicalSectionExtractionResult,
                temperature=0.0,
                use_json_mode=True,
                max_repair_attempts=2,
            )
        except Exception as exc:  # noqa: BLE001
            raise ClinicalInputExtractionError(
                f"Clinical input extraction failed: {exc}"
            ) from exc
        if not self._is_source_text_compatible(clinical_input, extraction.source_text):
            raise ClinicalInputExtractionError(
                "Clinical input extraction returned mismatched source_text."
            )
        if progress_callback is not None:
            progress_callback(1.0)
        return extraction
