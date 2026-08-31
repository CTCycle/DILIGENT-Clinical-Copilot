from __future__ import annotations

from common.prompts.clinical_assessment import (
    LIVERTOX_CLINICAL_USER_PROMPT,
    LIVERTOX_CONCLUSION_USER_PROMPT,
    LIVERTOX_REVISION_CLINICAL_USER_PROMPT,
    LIVERTOX_REVISION_CONCLUSION_USER_PROMPT,
)


def build_livertox_drug_assessment_user_prompt(
    *,
    revision: bool,
    drug_name: str,
    report_language: str,
    canonical_name: str,
    origins: str,
    extraction_metadata: str,
    livertox_status: str,
    excerpt: str,
    retrieved_documents_block: str,
    clinical_context: str,
    visit_date_anchor: str,
    therapy_start_details: str,
    suspension_details: str,
    timeline_note: str,
    pattern_summary: str,
    rucam_block: str,
    knowledge_prompt: str,
    metadata_block: str,
    livertox_score: str,
) -> str:
    template = (
        LIVERTOX_REVISION_CLINICAL_USER_PROMPT
        if revision
        else LIVERTOX_CLINICAL_USER_PROMPT
    )
    return template.format(
        drug_name=drug_name,
        report_language=report_language,
        canonical_name=canonical_name,
        origins=origins,
        extraction_metadata=extraction_metadata,
        livertox_status=livertox_status,
        excerpt=excerpt,
        retrieved_documents_block=retrieved_documents_block,
        clinical_context=clinical_context,
        visit_date_anchor=visit_date_anchor,
        therapy_start_details=therapy_start_details,
        suspension_details=suspension_details,
        timeline_note=timeline_note,
        pattern_summary=pattern_summary,
        rucam_block=rucam_block,
        knowledge_prompt=knowledge_prompt,
        metadata_block=metadata_block,
        livertox_score=livertox_score,
    )


def build_livertox_conclusion_user_prompt(
    *,
    revision: bool,
    report_language: str,
    clinical_context: str,
    multi_drug_report: str,
) -> str:
    template = (
        LIVERTOX_REVISION_CONCLUSION_USER_PROMPT
        if revision
        else LIVERTOX_CONCLUSION_USER_PROMPT
    )
    return template.format(
        report_language=report_language,
        clinical_context=clinical_context,
        multi_drug_report=multi_drug_report,
    )
