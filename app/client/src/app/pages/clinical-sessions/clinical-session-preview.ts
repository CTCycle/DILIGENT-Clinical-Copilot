import { ClinicalSessionDetail } from '../../core/models/inspection-types';
import { isRecord } from '../../core/utils';

export type DetectedDrugEvidence = {
  name: string;
  liverTox: boolean;
  rxNav: boolean;
  inAnamnesis: boolean;
  inTherapy: boolean;
  temporalReference: string;
  extractionFallback: boolean;
  bibliographyLabel: string;
  bibliographyFallback: boolean;
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

export type DrugEvidenceDraft = DetectedDrugEvidence & {
  hasPersistedMatch: boolean;
};

export function previewReport(detail: ClinicalSessionDetail): string {
  const report = detail.official_report_text || detail.report || detail.result_payload?.['report'];
  return typeof report === 'string' && report.trim()
    ? report.trim()
    : 'No AI report preview is available for this session.';
}

export function previewDetectedDrugs(detail: ClinicalSessionDetail): string[] {
  const detected = detail.result_payload?.['detected_drugs'];
  return Array.isArray(detected)
    ? detected.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
    : [];
}

export function buildPersistedDrugEvidence(detail: ClinicalSessionDetail): DrugEvidenceDraft[] {
  const rows = new Map<string, DrugEvidenceDraft>();
  const sections = sectionTextMap(detail);
  const ensureRow = (name: string, options: { fallback?: boolean } = {}): DrugEvidenceDraft => {
    const normalized = normalizeDrugName(name);
    const key = normalized || name.trim().toLowerCase();
    const existing = rows.get(key);
    if (existing) {
      existing.extractionFallback = existing.extractionFallback && Boolean(options.fallback);
      return existing;
    }
    const next: DrugEvidenceDraft = {
      name,
      liverTox: false,
      rxNav: false,
      inAnamnesis: textContainsDrug(sections.anamnesis, name),
      inTherapy: textContainsDrug(sections.therapy, name),
      temporalReference: drugTemporalReference(name, detail),
      extractionFallback: Boolean(options.fallback),
      bibliographyLabel: 'No backend match',
      bibliographyFallback: false,
      hasPersistedMatch: false,
    };
    rows.set(key, next);
    return next;
  };

  for (const name of previewDetectedDrugs(detail)) {
    for (const candidate of expandDrugCandidates(name, detail)) {
      ensureRow(candidate, { fallback: candidate !== name });
    }
  }

  const structuredCase = recordValue(detail.result_payload?.['structured_case']);
  const therapyDrugs = arrayValue(structuredCase?.['therapy_drugs']);
  const anamnesisDrugs = [
    ...arrayValue(structuredCase?.['anamnesis_drugs']),
    ...arrayValue(detail.result_payload?.['anamnesis_drugs']),
  ];
  for (const item of therapyDrugs) {
    const name = drugNameFromUnknown(item);
    if (!name) continue;
    for (const candidate of expandDrugCandidates(name, detail)) {
      ensureRow(candidate, { fallback: candidate !== name }).inTherapy = true;
    }
  }
  for (const item of anamnesisDrugs) {
    const name = drugNameFromUnknown(item);
    if (!name) continue;
    ensureRow(name).inAnamnesis = true;
  }

  for (const item of arrayValue(detail.result_payload?.['matched_drugs'])) {
    const record = recordValue(item);
    if (!record) continue;
    const name = stringValue(record['raw_drug_name'])
      || stringValue(record['drug_name'])
      || stringValue(record['matched_drug_name']);
    if (!name) continue;
    const expandedNames = expandDrugCandidates(name, detail);
    for (const expandedName of expandedNames) {
      ensureRow(expandedName, { fallback: expandedName !== name }).hasPersistedMatch = true;
    }
    if (looksLikeSentenceFragment(name) && expandedNames.length) continue;
    const row = ensureRow(name);
    const matchedName = stringValue(record['matched_drug_name']);
    row.hasPersistedMatch = true;
    row.name = row.name || matchedName || name;
    row.liverTox = row.liverTox || hasLiverToxEvidence(record);
    row.rxNav = row.rxNav || hasRxNavEvidence(record);
    row.bibliographyLabel = resolveDrugBibliographyLabel(row);
    row.inTherapy = row.inTherapy || originsContain(record, 'therapy') || rawMentionsContain(record, sections.therapy);
    row.inAnamnesis = row.inAnamnesis || originsContain(record, 'anamnesis') || rawMentionsContain(record, sections.anamnesis);
    row.temporalReference = drugTemporalReference(row.name, detail);
  }

  for (const row of rows.values()) {
    if (!row.liverTox && hasLiverToxReportEvidence(detail, row.name)) {
      row.liverTox = true;
      row.bibliographyLabel = resolveDrugBibliographyLabel(row);
    }
  }

  return [...rows.values()];
}

export function resolveDrugBibliographyLabel(
  row: Pick<DetectedDrugEvidence, 'liverTox' | 'rxNav'>,
  fallback?: Partial<Pick<DetectedDrugEvidence, 'liverTox' | 'rxNav'>>,
): string {
  const backendLabels = [
    row.liverTox ? 'LiverTox' : null,
    row.rxNav ? 'RxNav' : null,
  ].filter((label): label is string => Boolean(label));
  if (backendLabels.length) return backendLabels.join(' + ');
  const fallbackLabels = [
    fallback?.liverTox ? 'LiverTox catalog fallback' : null,
    fallback?.rxNav ? 'RxNav catalog fallback' : null,
  ].filter((label): label is string => Boolean(label));
  return fallbackLabels.length ? fallbackLabels.join(' + ') : 'No backend match';
}

function expandDrugCandidates(name: string, detail: ClinicalSessionDetail): string[] {
  if (!looksLikeSentenceFragment(name)) return [name];
  const sections = sectionTextMap(detail);
  const source = `${sections.anamnesis}\n${sections.therapy}\n${detail.session_text}`;
  const candidates = [
    ...extractDrugsAfterStarting(source),
    ...extractCurrentMedicationList(source),
  ];
  return candidates.length ? [...new Set(candidates)] : [name];
}

function looksLikeSentenceFragment(value: string): boolean {
  const normalized = value.trim();
  return normalized.split(/\s+/).length > 4 || /patient|suspected|injury|symptoms/i.test(normalized);
}

function extractDrugsAfterStarting(source: string): string[] {
  const match = source.match(/\bafter starting\s+([^\.\n]+?)(?:\.| symptoms| labs|$)/i);
  return match?.[1] ? splitMedicationList(match[1]) : [];
}

function extractCurrentMedicationList(source: string): string[] {
  const match = source.match(/\bcurrent medications are\s+([^\.\n]+)/i);
  return match?.[1] ? splitMedicationList(match[1]) : [];
}

function splitMedicationList(value: string): string[] {
  const normalized = value
    .replace(/\bamoxicillin\s+clavulanate\b/gi, 'amoxicillin clavulanate,')
    .replace(/\batorvastatin\b/gi, 'atorvastatin,')
    .replace(/\bramipril\b/gi, 'ramipril,');
  return normalized
    .replace(/\band\b/gi, ',')
    .split(',')
    .map((item) => item.trim().replace(/[.;:]$/g, ''))
    .filter((item) => item.length > 2);
}

function drugTemporalReference(name: string, detail: ClinicalSessionDetail): string {
  const structuredCase = recordValue(detail.result_payload?.['structured_case']);
  const therapyDrugs = arrayValue(structuredCase?.['therapy_drugs']);
  for (const item of therapyDrugs) {
    const record = recordValue(item);
    if (!record) continue;
    const drugName = drugNameFromUnknown(record);
    if (!drugName || normalizeDrugName(drugName) !== normalizeDrugName(name)) continue;
    const startDate = stringValue(record['therapy_start_date']);
    if (startDate && !looksLikeSentenceFragment(startDate)) return startDate;
    const temporal = stringValue(record['temporal_classification']);
    if (temporal) return temporal.replace(/_/g, ' ');
  }
  const sections = sectionTextMap(detail);
  const source = `${sections.anamnesis}\n${sections.therapy}`;
  if (textContainsDrug(source, name) && /\bafter starting\b/i.test(source)) return 'after starting';
  if (textContainsDrug(source, name) && /\bcurrent medications?\b/i.test(source)) return 'current medication';
  return 'time not specified';
}

function hasLiverToxEvidence(record: Record<string, unknown>): boolean {
  if (recordValue(record['matched_livertox_row'])) return true;
  if (stringValue(record['nbk_id'])) return true;
  const status = stringValue(record['match_status'])?.toLowerCase();
  if (status === 'matched_with_excerpt' || status === 'matched_no_excerpt' || status === 'matched') return true;
  if (record['missing_livertox'] === false) return true;
  const candidates = arrayValue(record['livertox_candidates']);
  return candidates.some((candidate) => {
    const item = recordValue(candidate);
    return item?.['has_excerpt'] === true;
  });
}

function hasLiverToxReportEvidence(detail: ClinicalSessionDetail, drugName: string): boolean {
  const report = typeof detail.sections?.['final_report'] === 'string'
    ? detail.sections['final_report']
    : '';
  const normalizedReport = report.toLowerCase();
  const reportIndex = normalizedReport.indexOf(drugName.trim().toLowerCase());
  if (reportIndex < 0) return false;
  return /livertox/.test(normalizedReport.slice(reportIndex, reportIndex + 2400));
}

function hasRxNavEvidence(record: Record<string, unknown>): boolean {
  if (stringValue(record['rxnorm_rxcui'])) return true;
  if (stringValue(record['rxcui'])) return true;
  const sources = arrayValue(record['sources']);
  return sources.some((source) => stringValue(source)?.toLowerCase() === 'rxnav');
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
  const sections = detail.sections || {};
  const anamnesis = typeof sections['anamnesis'] === 'string' ? sections['anamnesis'] : '';
  const therapy = typeof sections['therapy'] === 'string'
    ? sections['therapy']
    : typeof sections['drugs'] === 'string'
      ? sections['drugs']
      : '';
  return { anamnesis, therapy };
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
  const fromPayload = detail.result_payload?.['detected_diseases'];
  const fromAnamnesis = detail.result_payload?.['anamnesis_diseases'];
  const structuredCase = recordValue(detail.result_payload?.['structured_case']);
  const structuredDiseases = structuredCase?.['anamnesis_diseases'];
  const direct = collectDiseaseNames(fromPayload);
  if (direct.length) return direct;
  const anamnesis = collectDiseaseNames(fromAnamnesis);
  if (anamnesis.length) return anamnesis;
  const structured = collectDiseaseNames(structuredDiseases);
  if (structured.length) return structured;
  const fromSourceText = collectDiseasesFromSourceText(detail);
  if (fromSourceText.length) return fromSourceText;
  const report = previewReport(detail);
  const lines = report.split(/\r?\n/);
  const diseaseLine = lines.find((line) => /detected diseases?/i.test(line));
  if (!diseaseLine) return [];
  return diseaseLine
    .split(':')
    .slice(1)
    .join(':')
    .split(',')
    .map((item) => item.replace(/[*-]/g, '').trim())
    .filter((item) => item.length > 0);
}

function collectDiseasesFromSourceText(detail: ClinicalSessionDetail): string[] {
  const sectionExtraction = recordValue(detail.result_payload?.['section_extraction']);
  const candidates = [
    detail.sections?.['anamnesis'],
    detail.session_text,
    stringValue(sectionExtraction?.['anamnesis']),
  ];
  const source = candidates
    .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
    .join('\n');
  if (!source.trim()) return [];
  const diseasePhrases: string[] = [];
  const historyMatch = source.match(/\b(?:past history|medical history|history)\s+includes?\s+([^\.\n]+)/i);
  if (historyMatch?.[1]) diseasePhrases.push(...splitDiseasePhrase(historyMatch[1]));
  const suspectedMatch = source.match(/\b(?:suspected|concern is)\s+([^\.\n]*?(?:liver injury|hepatitis|cholestasis|hepatocellular pattern)[^\.\n]*)/i);
  if (suspectedMatch?.[1]) diseasePhrases.push(suspectedMatch[1].trim());
  return [...new Set(diseasePhrases.map((item) => item.trim()).filter((item) => item.length > 0))];
}

function splitDiseasePhrase(value: string): string[] {
  return value
    .replace(/\band\b/gi, ',')
    .split(',')
    .map((item) => item.trim().replace(/[.;:]$/g, ''))
    .filter((item) => item.length > 0);
}

export function diseaseTemporalLabel(disease: string): string {
  const normalized = disease.toLowerCase();
  if (normalized.includes('liver injury') || normalized.includes('hepatocellular')) return 'current concern';
  return 'past history';
}

function collectDiseaseNames(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const names = value
    .map((item) => {
      if (typeof item === 'string') return item.trim();
      const record = recordValue(item);
      if (!record) return '';
      return stringValue(record['name']) || stringValue(record['disease_name']) || '';
    })
    .filter((name) => name.length > 0);
  return [...new Set(names)];
}

export function previewLabTimeline(detail: ClinicalSessionDetail): LabTimelineRow[] {
  return arrayValue(detail.result_payload?.['lab_timeline'])
    .map((item) => recordValue(item))
    .filter((item): item is Record<string, unknown> => item !== null)
    .map((item) => {
      const value = stringValue(item['value']) || stringValue(item['value_text']) || 'N/A';
      const unit = stringValue(item['unit']) || '';
      const upperLimit = stringValue(item['upper_limit_normal']) || stringValue(item['upper_limit_text']) || 'N/A';
      const timing = stringValue(item['sample_date']) || stringValue(item['relative_time']) || 'Unknown';
      return {
        marker: stringValue(item['marker_name']) || 'Lab',
        value,
        unit,
        upperLimit,
        timing,
        source: stringValue(item['source']) || 'N/A',
        evidence: stringValue(item['evidence']) || '',
      };
    });
}

export function previewLaboratorySummary(detail: ClinicalSessionDetail): Array<{ label: string; value: string }> {
  const payload = detail.result_payload || {};
  const flatPayload = flattenPayload(payload);
  const fromPayload = collectLabValues(flatPayload);
  if (fromPayload.length) return fromPayload;

  const report = previewReport(detail).replace(/[*_`]/g, '');
  const regexMap: Array<{ label: string; regex: RegExp }> = [
    { label: 'ALT', regex: /\bALT\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z/%µμ\.]*\/?[A-Za-z]*)/i },
    { label: 'AST', regex: /\bAST\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z/%µμ\.]*\/?[A-Za-z]*)/i },
    { label: 'ALP', regex: /\bALP\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z/%µμ\.]*\/?[A-Za-z]*)/i },
    { label: 'Bilirubin', regex: /\b(?:total\s+)?bilirubin\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?\s*[A-Za-z/%µμ\.]*\/?[A-Za-z]*)/i },
    { label: 'INR', regex: /\bINR\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)/i },
    { label: 'R-score', regex: /\bR-?score\b\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)/i },
  ];
  return regexMap
    .map(({ label, regex }) => {
      const match = report.match(regex);
      return match?.[1] ? { label, value: match[1].trim() } : null;
    })
    .filter((item): item is { label: string; value: string } => item !== null);
}

export function previewHepatotoxicityPattern(detail: ClinicalSessionDetail): string {
  const payload = detail.result_payload || {};
  const flatPayload = flattenPayload(payload);
  const fromPayload = [
    flatPayload['hepatotoxicity_pattern'],
    flatPayload['pattern_classification'],
    flatPayload['classification'],
    flatPayload['hepatotoxicity.classification'],
  ].find((value) => typeof value === 'string' && value.trim().length > 0);
  if (typeof fromPayload === 'string') return fromPayload.trim();

  const report = previewReport(detail).replace(/[*_`]/g, '');
  const patternMatch = report.match(/\b(?:hepatotoxicity pattern|classification)\b\s*[:=]\s*([A-Za-z -]+)/i);
  return patternMatch?.[1]?.trim() || 'N/A';
}

function flattenPayload(
  value: unknown,
  prefix = '',
  acc: Record<string, unknown> = {},
): Record<string, unknown> {
  if (!isRecord(value)) return acc;
  for (const [key, nested] of Object.entries(value)) {
    const fullKey = prefix ? `${prefix}.${key}` : key;
    acc[fullKey.toLowerCase()] = nested;
    if (isRecord(nested)) flattenPayload(nested, fullKey, acc);
  }
  return acc;
}

function collectLabValues(flatPayload: Record<string, unknown>): Array<{ label: string; value: string }> {
  const keys: Array<{ label: string; includes: string[] }> = [
    { label: 'ALT', includes: ['alt'] },
    { label: 'AST', includes: ['ast'] },
    { label: 'ALP', includes: ['alp', 'alkaline_phosphatase'] },
    { label: 'Bilirubin', includes: ['bilirubin', 'tbil'] },
    { label: 'INR', includes: ['inr'] },
    { label: 'R-score', includes: ['r_score', 'rscore', 'r-score'] },
  ];
  return keys
    .map(({ label, includes }) => {
      const payloadEntry = Object.entries(flatPayload).find(([key, val]) =>
        includes.some((needle) => key.includes(needle))
        && val !== null
        && val !== undefined
        && String(val).trim().length > 0,
      );
      if (!payloadEntry) return null;
      return { label, value: String(payloadEntry[1]).trim() };
    })
    .filter((item): item is { label: string; value: string } => item !== null);
}

function drugNameFromUnknown(value: unknown): string | null {
  if (typeof value === 'string') return value.trim() || null;
  const record = recordValue(value);
  if (!record) return null;
  return stringValue(record['name'])
    || stringValue(record['drug_name'])
    || stringValue(record['raw_drug_name'])
    || stringValue(record['matched_drug_name']);
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
