from __future__ import annotations

from configurations.management import build_settings_payload_from_json
from domain.clinical.entities import PatientData
from domain.settings.environment import EnvironmentSnapshot
from services.session.session_shared import build_failed_session_payload

###############################################################################
def test_local_only_deployment_mode_is_explicit_in_runtime_payload() -> None:
    payload = build_settings_payload_from_json(
        {},
        EnvironmentSnapshot(
            ollama_url=None,
            ollama_host=None,
            ollama_port=None,
        ),
    )

    assert payload["deployment"]["mode"] == "local_single_user"

###############################################################################
def test_failed_clinical_job_payload_omits_phi_by_default() -> None:
    failed_payload = build_failed_session_payload(
        payload=PatientData(
            anamnesis="Patient Jane Doe has sensitive history.",
            drugs="Sensitive medication text",
            laboratory_analysis="ALT 1234 with identifiable note",
        ),
        patient_image_base64="base64-sensitive-image",
        issues=[],
        error_message="raw failure with patient Jane Doe",
        elapsed_seconds=1.2,
    )

    assert failed_payload["anamnesis"] is None
    assert failed_payload["drugs"] is None
    assert failed_payload["laboratory_analysis"] is None
    assert failed_payload["patient_image_base64"] is None
    result_payload = failed_payload["session_result_payload"]
    assert result_payload["error"] == "Clinical analysis failed before completion."
    assert result_payload["section_extraction"] is None
    assert result_payload["failure_metadata"]["has_patient_image"] is True
    counts = result_payload["failure_metadata"]["input_character_counts"]
    assert counts["anamnesis"] > 0
    assert counts["drugs"] > 0
    assert counts["laboratory_analysis"] > 0
