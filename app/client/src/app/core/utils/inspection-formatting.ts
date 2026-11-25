// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { InspectionRagVectorStoreSummary } from '../models/inspection-types';
import { formatAppDateTime } from './date-formatting';

export type InspectionViewId =
  | 'rxnav'
  | 'livertox'
  | 'terminology'
  | 'dilirank'
  | 'rag';

export type InspectionViewOption = {
  id: InspectionViewId;
  label: string;
  iconPath: string;
  group: string;
};

export function inspectionTabId(view: InspectionViewId): string {
  return `inspection-tab-${view}`;
}

export function formatInspectionDateTime(value: string | null): string {
  return formatAppDateTime(value);
}

export function resolveRagDocumentsPath(
  vectorStore: InspectionRagVectorStoreSummary | null,
): string {
  return vectorStore?.source_documents_path.trim() ?? '';
}
