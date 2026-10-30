// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { EvidenceSourceId } from "./models/types";

export type EvidenceSourceOption = {
  id: EvidenceSourceId;
  label: string;
  description: string;
};

export const EVIDENCE_SOURCE_OPTIONS: readonly EvidenceSourceOption[] = [
  {
    id: "livertox",
    label: "LiverTox",
    description: "NIH drug-level hepatotoxicity monographs and likelihood grades.",
  },
  {
    id: "dilirank",
    label: "FDA DILIrank 2.0",
    description: "FDA drug-level DILI concern classification.",
  },
  {
    id: "documents",
    label: "Indexed documents",
    description: "Passages retrieved from the document library you indexed.",
  },
];

export const DEFAULT_EVIDENCE_SOURCES: readonly EvidenceSourceId[] = ["livertox"];

export function evidenceSourceLabel(id: string): string {
  return EVIDENCE_SOURCE_OPTIONS.find((option) => option.id === id)?.label ?? id;
}

export function describeEvidenceSources(sources: readonly string[]): string {
  return sources.length
    ? sources.map(evidenceSourceLabel).join(", ")
    : "None (LLM knowledge only)";
}
