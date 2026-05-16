import { InspectionRagVectorStoreSummary } from '../models/types';

export type InspectionViewId =
  | 'sessions'
  | 'rxnav'
  | 'livertox'
  | 'rag';

export type InspectionViewOption = {
  id: InspectionViewId;
  label: string;
};

export function inspectionTabId(view: InspectionViewId): string {
  return `inspection-tab-${view}`;
}

export function formatInspectionDateTime(value: string | null): string {
  if (!value) return 'N/A';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

export function formatInspectionDuration(seconds: number | null): string {
  if (typeof seconds !== 'number' || Number.isNaN(seconds) || seconds < 0) return 'N/A';
  const rounded = Math.round(seconds);
  if (rounded < 60) return `${rounded}s`;
  return `${Math.floor(rounded / 60)}m ${rounded % 60}s`;
}

export function resolveRagDocumentsPath(
  vectorStore: InspectionRagVectorStoreSummary | null,
): string {
  if (!vectorStore) {
    return '';
  }
  const explicitPath = vectorStore.source_documents_path?.trim();
  if (explicitPath) {
    return explicitPath;
  }
  const vectorDbPath = vectorStore.vector_db_path?.trim();
  if (!vectorDbPath) {
    return '';
  }
  return vectorDbPath.replace(new RegExp(String.raw`[\\/]vectors$`, 'i'), (match) =>
    match.startsWith('\\') ? '\\documents' : '/documents',
  );
}
