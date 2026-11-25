// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import {
  ClinicalSessionDetail,
  TerminologyCandidate,
  TerminologyResolution,
  TerminologyResolutionStatus,
} from '../../core/models/inspection-types';
import { isRecord } from '../../core/utils';

const TERMINOLOGY_STATUSES = new Set<TerminologyResolutionStatus>(['verified', 'probable', 'ambiguous', 'unresolved']);
const ACCEPTED_DECISIONS = new Set(['accepted_verified', 'accepted_probable']);

export type DetectedDrugEvidence = {
  name: string;
  terminology: TerminologyResolution | null;
  liverTox: boolean;
  rxNav: boolean;
  inAnamnesis: boolean;
  inTherapy: boolean;
  temporalReference: string;
  bibliographyLabel: string;
};

export type LabTimelineRow = {
  marker: string;
  value: string;
  unit: string;
  upperLimit: string;
  timing: string;
  source: string;
  evidence: string;
};

export function previewReport(detail: ClinicalSessionDetail): string {
  return detail.official_report_text?.trim() || 'No AI report preview is available for this session.';
}

export function buildPersistedDrugEvidence(detail: ClinicalSessionDetail): DetectedDrugEvidence[] {
  const rows = new Map<string, DetectedDrugEvidence>();
  const sections = sectionTextMap(detail);
  const structuredCase = recordValue(detail.result_payload['structured_case']);
  const therapyDrugs = recordArray(structuredCase?.['therapy_drugs']);
  const ensureRow = (name: string): DetectedDrugEvidence => {
    const key = normalizeDrugName(name) || name.trim().toLowerCase();
    const existing = rows.get(key);
    if (existing) return existing;
    const next: DetectedDrugEvidence = {
      name,
      terminology: null,
      liverTox: false,
      rxNav: false,
      inAnamnesis: textContainsDrug(sections.anamnesis, name),
      inTherapy: textContainsDrug(sections.therapy, name),
      temporalReference: drugTemporalReference(name, therapyDrugs),
      bibliographyLabel: 'No backend match',
    };
    rows.set(key, next);
    return next;
  };

  for (const record of recordArray(detail.result_payload['matched_drugs'])) {
    const name = stringValue(record['raw_drug_name']);
    if (!name) continue;
    const row = ensureRow(name);
    row.terminology = row.terminology ?? parseTerminology(record['terminology']);
    row.liverTox = row.liverTox || hasLiverToxEvidence(record);
    row.rxNav = row.rxNav || hasRxNavEvidence(record);
    row.bibliographyLabel = resolveDrugBibliographyLabel(row);
    row.inTherapy = row.inTherapy || originsContain(record, 'therapy') || rawMentionsContain(record, sections.therapy);
    row.inAnamnesis = row.inAnamnesis || originsContain(record, 'anamnesis') || rawMentionsContain(record, sections.anamnesis);
  }
  for (const record of therapyDrugs) {
    const name = stringValue(record['name']);
    if (name) ensureRow(name).inTherapy = true;
  }
  for (const record of recordArray(structuredCase?.['anamnesis_drugs'])) {
    const name = stringValue(record['name']);
    if (name) ensureRow(name).inAnamnesis = true;
  }

  return [...rows.values()];
}

function resolveDrugBibliographyLabel(row: Pick<DetectedDrugEvidence, 'liverTox' | 'rxNav'>): string {
  const labels = [
    row.liverTox ? 'LiverTox' : null,
    row.rxNav ? 'RxNav' : null,
  ].filter((label): label is string => Boolean(label));
  return labels.length ? labels.join(' + ') : 'No backend match';
}

function drugTemporalReference(name: string, therapyDrugs: Record<string, unknown>[]): string {
  for (const record of therapyDrugs) {
    const drugName = stringValue(record['name']);
    if (!drugName || normalizeDrugName(drugName) !== normalizeDrugName(name)) continue;
    const startDate = stringValue(record['therapy_start_date']);
    if (startDate) return startDate;
    const temporal = stringValue(record['temporal_classification']);
    if (temporal) return temporal.replace(/_/g, ' ');
  }
  return 'time not specified';
}

function hasLiverToxEvidence(record: Record<string, unknown>): boolean {
  if (stringValue(record['nbk_id'])) return true;
  const status = stringValue(record['match_status'])?.toLowerCase();
  if (status && ACCEPTED_DECISIONS.has(status)) return true;
  if (record['missing_livertox'] === false) return true;
  return recordArray(record['livertox_candidates']).some((candidate) => candidate['has_excerpt'] === true);
}

function hasRxNavEvidence(record: Record<string, unknown>): boolean {
  return Boolean(stringValue(record['rxnorm_rxcui']));
}

/** Typed view of the persisted terminology identity; null when the drug has none. */
export function parseTerminology(value: unknown): TerminologyResolution | null {
  const record = recordValue(value);
  const status = stringValue(record?.['status']) as TerminologyResolutionStatus | null;
  if (!record || !status || !TERMINOLOGY_STATUSES.has(status)) return null;
  const names = (items: unknown) => arrayValue(items)
    .map((item) => recordValue(item))
    .filter((item): item is Record<string, unknown> => !!item && typeof item['drug_id'] === 'number' && !!stringValue(item['name']));
  return {
    ...(record as unknown as TerminologyResolution),
    status,
    accepted: record['accepted'] === true,
    ingredients: names(record['ingredients']) as unknown as TerminologyResolution['ingredients'],
    precise_forms: names(record['precise_forms']) as unknown as TerminologyResolution['precise_forms'],
    candidates: names(record['candidates']) as unknown as TerminologyCandidate[],
    review_reasons: arrayValue(record['review_reasons']).filter((item): item is string => typeof item === 'string'),
    strengths: arrayValue(record['strengths']).filter((item): item is string => typeof item === 'string'),
    qualifiers: arrayValue(record['qualifiers']).filter((item): item is string => typeof item === 'string'),
  };
}

export function terminologyResolvedLabel(terminology: TerminologyResolution): string {
  if (!terminology.accepted) return '';
  const ingredients = terminology.ingredients.map((item) => item.name);
  return ingredients.length > 1
    ? `${terminology.preferred_name ?? ingredients.join(' + ')} (${ingredients.join(' + ')})`
    : terminology.preferred_name ?? ingredients[0] ?? '';
}

function originsContain(record: Record<string, unknown>, origin: 'therapy' | 'anamnesis'): boolean {
  return arrayValue(record['origins']).some((value) => stringValue(value)?.toLowerCase().includes(origin));
}

function rawMentionsContain(record: Record<string, unknown>, text: string): boolean {
  return arrayValue(record['raw_mentions']).some((value) => {
    const mention = stringValue(value);
    return mention ? textContainsDrug(text, mention) : false;
  });
}

export function normalizeDrugName(value: string): string {
  return value.toLowerCase().replace(/\([^)]*\)/g, '').replace(/[^a-z0-9]+/g, ' ').trim();
}

function sectionTextMap(detail: ClinicalSessionDetail): { anamnesis: string; therapy: string } {
  return {
    anamnesis: detail.sections['anamnesis'] ?? '',
    therapy: detail.sections['drugs'] ?? '',
  };
}

function textContainsDrug(text: string, drug: string): boolean {
  if (!text.trim() || !drug.trim()) return false;
  const normalizedText = normalizeDrugName(text);
  const normalizedDrug = normalizeDrugName(drug);
  if (!normalizedDrug) return false;
  if (normalizedText.includes(normalizedDrug)) return true;
  const firstToken = normalizedDrug.split(' ')[0] || '';
  return firstToken.length > 3 && normalizedText.includes(firstToken);
}

export function previewDetectedDiseases(detail: ClinicalSessionDetail): string[] {
  const structuredCase = recordValue(detail.result_payload['structured_case']);
  const names = recordArray(structuredCase?.['anamnesis_diseases'])
    .filter((record) => {
      const attribution = stringValue(record['attribution'])?.toLowerCase();
      const diagnosisStatus = stringValue(record['diagnosis_status'])?.toLowerCase();
      return attribution !== 'negated' && diagnosisStatus !== 'ruled-out' && diagnosisStatus !== 'ruled_out';
    })
    .map((record) => stringValue(record['name']))
    .filter((name): name is string => Boolean(name));
  return [...new Set(names)];
}

export function diseaseTemporalLabel(disease: string): string {
  const normalized = disease.toLowerCase();
  if (normalized.includes('liver injury') || normalized.includes('hepatocellular')) return 'current concern';
  return 'past history';
}

function labSourceLabel(source: string | null): string {
  if (source === 'merged') return 'laboratory_analysis + anamnesis';
  return source || 'N/A';
}

export function previewLabTimeline(detail: ClinicalSessionDetail): LabTimelineRow[] {
  return recordArray(detail.result_payload['lab_timeline']).map((item) => ({
    marker: stringValue(item['marker_name']) || 'Lab',
    value: stringValue(item['value']) || stringValue(item['value_text']) || 'N/A',
    unit: stringValue(item['unit']) || '',
    upperLimit: stringValue(item['upper_limit_normal']) || stringValue(item['upper_limit_text']) || 'N/A',
    timing: stringValue(item['sample_date']) || stringValue(item['relative_time']) || 'Unknown',
    source: labSourceLabel(stringValue(item['source'])),
    evidence: stringValue(item['evidence']) || '',
  }));
}

export function previewHepatotoxicityPattern(detail: ClinicalSessionDetail): string {
  const metadata = recordValue(detail.result_payload['extraction_metadata']);
  return stringValue(recordValue(metadata?.['hepatic_pattern'])?.['final_value']) ?? 'N/A';
}

function stringValue(value: unknown): string | null {
  if (typeof value === 'string') return value.trim() || null;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return null;
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return isRecord(value) ? value : null;
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function recordArray(value: unknown): Record<string, unknown>[] {
  return arrayValue(value).filter((item): item is Record<string, unknown> => isRecord(item));
}
