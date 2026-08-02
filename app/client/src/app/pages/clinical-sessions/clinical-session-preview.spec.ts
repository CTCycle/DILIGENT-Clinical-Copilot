import { ClinicalSessionDetail } from '../../core/models/types';
import {
  previewDetectedDiseases,
  previewHepatotoxicityPattern,
  previewLaboratorySummary,
  previewLabTimeline,
  previewReport,
} from './clinical-session-preview';

describe('clinical-session-preview', () => {
  const detail = (overrides: Partial<ClinicalSessionDetail> = {}): ClinicalSessionDetail => ({
    session_id: 1,
    patient_name: null,
    visit_date: null,
    session_timestamp: null,
    version: 1,
    original_session_id: null,
    status: 'successful',
    text_extraction_model: null,
    clinical_model: null,
    metadata: {},
    sections: {},
    session_text: '',
    source_clinical_text: '',
    result_payload: {},
    report: null,
    official_report_text: null,
    manual_edit_history: [],
    ...overrides,
  });

  it('keeps official report precedence and falls back to payload reports', () => {
    expect(previewReport(detail({
      official_report_text: ' official report ',
      report: 'legacy report',
      result_payload: { report: 'payload report' },
    }))).toBe('official report');

    expect(previewReport(detail({
      report: null,
      result_payload: { report: ' payload report ' },
    }))).toBe('payload report');

    expect(previewReport(detail())).toBe('No AI report preview is available for this session.');
  });

  it('uses direct disease payloads before source-text fallback and ignores unknown payload shapes', () => {
    expect(previewDetectedDiseases(detail({
      result_payload: {
        detected_diseases: ['Direct diagnosis'],
        anamnesis_diseases: ['Older diagnosis'],
      },
      sections: { anamnesis: 'Past history includes diabetes.' },
    }))).toEqual(['Direct diagnosis']);

    expect(previewDetectedDiseases(detail({
      result_payload: {
        detected_diseases: { unexpected: true },
        structured_case: ['unexpected'],
        section_extraction: [],
      },
      report: null,
      session_text: '',
    }))).toEqual([]);
  });

  it('extracts laboratory and hepatotoxicity previews without requiring a fixed payload shape', () => {
    const session = detail({
      result_payload: {
        labs: { ALT: 123 },
        hepatotoxicity: { classification: 'cholestatic' },
        lab_timeline: [{ marker_name: 'ALT', value: 123, relative_time: 'day 1' }],
      },
    });

    expect(previewLaboratorySummary(session)).toContainEqual({ label: 'ALT', value: '123' });
    expect(previewHepatotoxicityPattern(session)).toBe('cholestatic');
    expect(previewLabTimeline(session)).toEqual([expect.objectContaining({
      marker: 'ALT',
      value: '123',
      timing: 'day 1',
    })]);
  });
});
