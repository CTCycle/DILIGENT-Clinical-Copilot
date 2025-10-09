from __future__ import annotations

import asyncio
import inspect
from typing import Any

from DILIGENT.app.api.models.prompts import (
    CLINICAL_ENHANCER_SYSTEM_PROMPT,
    CLINICAL_ENHANCER_USER_PROMPT,
)
from DILIGENT.app.api.models.providers import initialize_llm_client
from DILIGENT.app.api.schemas.clinical import PatientData
from DILIGENT.app.configurations import ClientRuntimeConfig
from DILIGENT.app.constants import DEFAULT_LLM_TIMEOUT_SECONDS
from DILIGENT.app.logger import logger


###############################################################################
class ClinicalTextEnhancer:
    SECTION_TEMPLATES: dict[str, dict[str, str]] = {
        "anamnesis": {
            "title": "Anamnesis",
            "instruction": (
                "Polish the anamnesis while keeping every fact unchanged. "
                "Normalize punctuation, spacing, and terminology so the prose reads "
                "naturally in English. Maintain any list or multiline formatting from the "
                "input."
            ),
        },
        "exams": {
            "title": "Exams",
            "instruction": (
                "Rewrite the exam findings in fluent English, fixing typographical errors "
                "and spacing. Preserve numeric values, dates, and the existing multiline "
                "structure."
            ),
        },
        "drugs": {
            "title": "Drugs",
            "instruction": (
                "Clean the medication list so it is consistent, readable, and in English. "
                "Do not alter drug names, dosages, schedules, or suspension notes. Keep "
                "the multiline layout and bulleting exactly as provided."
            ),
        },
    }

    def __init__(self, *, timeout_s: float = DEFAULT_LLM_TIMEOUT_SECONDS) -> None:
        self.timeout_s = float(timeout_s)
        self.client = initialize_llm_client(purpose="enhancer", timeout_s=self.timeout_s)
        provider, model_candidate = ClientRuntimeConfig.resolve_provider_and_model(
            "enhancer"
        )
        self.using_ollama = provider == "ollama"
        enhancer_model = ClientRuntimeConfig.get_enhancer_model()
        self.model = (
            model_candidate
            or enhancer_model
            or ClientRuntimeConfig.get_clinical_model()
        )
        self.temperature = 0.2
        self.keep_alive = "5m" if self.using_ollama else None
        try:
            chat_signature = inspect.signature(self.client.chat)
        except (TypeError, ValueError):
            chat_signature = None
        parameters = chat_signature.parameters if chat_signature else {}
        self._chat_supports_temperature = "temperature" in parameters
        self._chat_supports_think = "think" in parameters
        self._chat_supports_options = "options" in parameters
        self._chat_supports_keep_alive = "keep_alive" in parameters

    # -------------------------------------------------------------------------
    async def enhance(self, payload: PatientData) -> PatientData:
        sections: list[tuple[str, dict[str, str], str]] = []
        for field, config in self.SECTION_TEMPLATES.items():
            original = getattr(payload, field, None)
            if original:
                sections.append((field, config, original))

        if not sections:
            return payload

        updates: dict[str, str] = {}
        if self.using_ollama:
            for field, config, original in sections:
                try:
                    rewritten = await self._rewrite_section(
                        section_name=config["title"],
                        text=original,
                        instruction=config["instruction"],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Clinical text enhancement failed for %s: %s", config["title"], exc
                    )
                    continue
                cleaned = rewritten.strip()
                if cleaned:
                    updates[field] = cleaned
        else:
            tasks = [
                self._rewrite_section(
                    section_name=config["title"],
                    text=original,
                    instruction=config["instruction"],
                )
                for _, config, original in sections
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for (field, config, _), outcome in zip(sections, results):
                if isinstance(outcome, Exception):
                    logger.warning(
                        "Clinical text enhancement failed for %s: %s", config["title"], outcome
                    )
                    continue
                cleaned = outcome.strip()
                if cleaned:
                    updates[field] = cleaned

        if not updates:
            return payload
        return payload.model_copy(update=updates)

    # -------------------------------------------------------------------------
    async def _rewrite_section(
        self, *, section_name: str, text: str, instruction: str
    ) -> str:
        messages = [
            {"role": "system", "content": CLINICAL_ENHANCER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CLINICAL_ENHANCER_USER_PROMPT.format(
                    section_name=section_name,
                    instruction=instruction,
                    text=text,
                ),
            },
        ]
        chat_kwargs = self._build_chat_kwargs(messages)
        raw = await self.client.chat(**chat_kwargs)
        return self._coerce_chat_text(raw) or text

    # -------------------------------------------------------------------------
    @staticmethod
    def _coerce_chat_text(raw_response: Any) -> str:
        if isinstance(raw_response, str):
            return raw_response.strip()
        if isinstance(raw_response, dict):
            for key in ("content", "text", "response", "message"):
                value = raw_response.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            choices = raw_response.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message") or {}
                    if isinstance(message, dict):
                        content = message.get("content")
                        if isinstance(content, str) and content.strip():
                            return content.strip()
        return ""

    # -------------------------------------------------------------------------
    def _build_chat_kwargs(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        chat_kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        options: dict[str, Any] = {}

        if self._chat_supports_temperature:
            chat_kwargs["temperature"] = self.temperature
        else:
            options["temperature"] = self.temperature

        if self._chat_supports_think:
            chat_kwargs["think"] = False
        else:
            options["think"] = False

        if self._chat_supports_keep_alive and self.keep_alive:
            chat_kwargs["keep_alive"] = self.keep_alive

        if options and self._chat_supports_options:
            chat_kwargs["options"] = options
        elif options and not self._chat_supports_options:
            if not self._chat_supports_temperature and "temperature" in options:
                chat_kwargs["temperature"] = options["temperature"]

        return chat_kwargs
