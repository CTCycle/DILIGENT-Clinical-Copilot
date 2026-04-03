from __future__ import annotations

from DILIGENT.server.domain.clinical import (
    ClinicalPipelineValidationError,
    DrugEntry,
    PatientData,
    PatientDrugs,
    PipelineIssue,
)
from DILIGENT.server.domain.validation import ValidationMessageBundle


def build_validation_bundle(report_language: str) -> ValidationMessageBundle:
    if report_language.startswith("it"):
        return ValidationMessageBundle(
            missing_anamnesis="Fornire l’anamnesi.",
            missing_visit_date="Fornire la data della visita.",
            missing_timed_drug="Fornire almeno un farmaco con data di inizio, sospensione o altra informazione temporale.",
            insufficient_labs="Fornire dati laboratoristici sufficienti per determinare il pattern di epatotossicità, idealmente ALT o AST datati, ALP e bilirubina.",
        )
    if report_language.startswith("de"):
        return ValidationMessageBundle(
            missing_anamnesis="Bitte Anamnese angeben.",
            missing_visit_date="Bitte Besuchsdatum angeben.",
            missing_timed_drug="Bitte mindestens ein Arzneimittel mit Start-, Stopp- oder anderen Zeitangaben angeben.",
            insufficient_labs="Bitte ausreichend Laborwerte zur Bestimmung des Hepatotoxizitätsmusters angeben, idealerweise datiertes ALT oder AST, ALP und Bilirubin.",
        )
    if report_language.startswith("fr"):
        return ValidationMessageBundle(
            missing_anamnesis="Veuillez fournir l’anamnèse.",
            missing_visit_date="Veuillez fournir la date de visite.",
            missing_timed_drug="Veuillez fournir au moins un médicament avec une date de début, d’arrêt ou une autre information temporelle.",
            insufficient_labs="Veuillez fournir des données biologiques suffisantes pour déterminer le profil d’hépatotoxicité, idéalement ALT ou AST datés, PAL et bilirubine.",
        )
    if report_language.startswith("es"):
        return ValidationMessageBundle(
            missing_anamnesis="Proporcione la anamnesis.",
            missing_visit_date="Proporcione la fecha de la visita.",
            missing_timed_drug="Proporcione al menos un fármaco con fecha de inicio, suspensión u otra información temporal.",
            insufficient_labs="Proporcione datos de laboratorio suficientes para determinar el patrón de hepatotoxicidad, idealmente ALT o AST con fecha, FA y bilirrubina.",
        )
    return ValidationMessageBundle(
        missing_anamnesis="Provide the anamnesis.",
        missing_visit_date="Provide the visit date.",
        missing_timed_drug="Provide at least one drug with start, stop, or other timing information.",
        insufficient_labs="Provide laboratory data sufficient to determine hepatotoxicity pattern, ideally dated ALT or AST, ALP, and bilirubin.",
    )


def ensure_required_sections(
    payload: PatientData,
    *,
    bundle: ValidationMessageBundle,
) -> None:
    issues: list[PipelineIssue] = []
    if not (payload.anamnesis or "").strip():
        issues.append(
            PipelineIssue(
                severity="error",
                code="missing_anamnesis",
                message=bundle.missing_anamnesis,
                field="anamnesis",
            )
        )
    if payload.visit_date is None:
        issues.append(
            PipelineIssue(
                severity="error",
                code="missing_visit_date",
                message=bundle.missing_visit_date,
                field="visit_date",
            )
        )
    if issues:
        raise ClinicalPipelineValidationError(issues=issues, message=issues[0].message)


def has_timing_information(entry: DrugEntry) -> bool:
    return bool(
        entry.therapy_start_date
        or entry.suspension_date
        or entry.temporal_classification == "temporal_known"
        or entry.therapy_start_status is True
        or entry.suspension_status is True
    )


def ensure_timed_therapy_drug(
    therapy_drugs: PatientDrugs,
    *,
    bundle: ValidationMessageBundle,
) -> None:
    if any(has_timing_information(entry) for entry in therapy_drugs.entries):
        return
    issue = PipelineIssue(
        severity="error",
        code="missing_timed_drug",
        message=bundle.missing_timed_drug,
        field="drugs",
    )
    raise ClinicalPipelineValidationError(issues=[issue], message=issue.message)
