from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from configurations.llm_configs import LLMRuntimeConfig
from domain.clinical.entities import ClinicalSessionRequest
from services.runtime.jobs import get_job_manager
from services.session import preflight as preflight_module
from services.session.factory import build_clinical_session_service

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    from PyPDF2 import PdfReader  # type: ignore[no-redef]


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text:
            chunks.append(text)
    return "\n\n".join(chunks).strip()


@pytest.mark.skipif(
    os.environ.get("RUN_PDF_CORPUS_REGRESSION") != "1",
    reason="Set RUN_PDF_CORPUS_REGRESSION=1 to run corpus-scale PDF preflight regression.",
)
def test_pdf_corpus_preflight_and_preprocess_have_no_blockers() -> None:
    corpus_dir = Path(
        os.environ.get("PDF_CORPUS_DIR", r"C:\Users\Thomas V\Desktop\DILI_analysis\DILI")
    )
    assert corpus_dir.exists(), f"PDF corpus directory not found: {corpus_dir}"
    pdf_files = sorted(corpus_dir.glob("*.pdf"))
    assert pdf_files, f"No PDF files found in: {corpus_dir}"

    service = build_clinical_session_service(get_job_manager())
    runtime_provider = (LLMRuntimeConfig.get_llm_provider() or "").strip()
    selected_model_providers = [runtime_provider] if runtime_provider else []

    failures: list[dict[str, object]] = []
    for index, pdf_path in enumerate(pdf_files, start=1):
        clinical_input = _extract_pdf_text(pdf_path)
        if not clinical_input:
            failures.append(
                {
                    "document": pdf_path.name,
                    "stage": "extract_text",
                    "error": "empty_text",
                }
            )
            continue

        request = ClinicalSessionRequest(
            name=f"PDF-{index}",
            visit_date=None,
            clinical_input=clinical_input,
            selected_model_providers=selected_model_providers,
        )
        preflight = preflight_module.validate_clinical_input_preflight(service, request)
        if preflight.blocking_issues:
            failures.append(
                {
                    "document": pdf_path.name,
                    "stage": "preflight",
                    "blocking_codes": [issue.code for issue in preflight.blocking_issues],
                }
            )

        try:
            asyncio.run(service.preprocess_unified_input(request))
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "document": pdf_path.name,
                    "stage": "preprocess_unified_input",
                    "error": str(exc),
                }
            )

    assert not failures, f"PDF corpus regression failures: {failures}"

