// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { GuidedTourDefinition, TipDefinition } from './guidance.types';

export const GUIDANCE_CONTENT_VERSION = 1;
const DILI_ASSESSMENT_TOUR_VERSION = 3;

export const DILI_ASSESSMENT_TOUR: GuidedTourDefinition = {
  id: 'dili-assessment-tour',
  version: DILI_ASSESSMENT_TOUR_VERSION,
  title: 'Run a DILI assessment',
  description: 'A short tour of the controls that shape a first assessment.',
  route: '/',
  steps: [
    {
      target: '[data-guidance-target="dili-clinical-input"]',
      title: 'Start with the clinical input',
      body: 'Keep Anamnesis, Therapy, and Laboratory history separate. Use short, dated entries where possible.',
      preferredPlacement: 'right',
    },
    {
      target: '[data-guidance-target="dili-patient-details"]',
      title: 'Set the patient details',
      body: 'Enter the patient name and visit date here. These details identify the assessment and anchor the clinical timeline.',
      preferredPlacement: 'left',
    },
    {
      target: '[data-guidance-target="dili-review-run"]',
      title: 'Review, then run',
      body: 'Run opens a short setup: pre-flight checks, evidence sources, model overrides for this session, and a final review before the analysis starts.',
      preferredPlacement: 'top',
    },
  ],
};

export const GUIDANCE_DEFINITIONS = {
  diliFirstAssessment: {
    id: 'dili-first-assessment',
    version: GUIDANCE_CONTENT_VERSION,
    title: 'Get started with DILI Agent',
    description: 'Configure a model, structure the clinical input, and review the pre-flight check before running.',
  },
  modelRagSettings: {
    id: 'model-rag-settings',
    version: GUIDANCE_CONTENT_VERSION,
    title: 'RAG settings',
    description: 'These settings control the shared index: chunking, embeddings and storage. Retrieval and reranking are set per agent in Agent Studio.',
  },
  clinicalSessionSections: {
    id: 'clinical-session-sections',
    version: GUIDANCE_CONTENT_VERSION,
    title: 'Session sections',
    description: 'Preview is read-only, Text Editor saves manual edits, Revision creates a new draft, and Timeline stores generated event histories.',
  },
  timelineReviewControls: {
    id: 'timeline-review-controls',
    version: GUIDANCE_CONTENT_VERSION,
    title: 'Timeline filters',
    description: 'Filter by source evidence, adjust density, and inspect any event for its source, timing, and confidence.',
  },
} as const;

export const TIPS_AND_TRICKS: readonly TipDefinition[] = [
  {
    id: 'first-assessment',
    title: 'Run a first assessment',
    body: 'Configure a model, structure the input, then review the pre-flight check before running.',
    actionLabel: 'Show me',
    action: 'tour',
  },
  {
    id: 'model-roles',
    title: 'Choose agent models',
    body: 'Each DILI agent picks its own clinical, extraction and timeline models. Edit an agent in Agent Studio, then save a new revision.',
    actionLabel: 'Open Agent Studio',
    action: 'configurations',
  },
  {
    id: 'rag-boundary',
    title: 'Choose evidence in the analysis wizard',
    body: 'The retrieval policy of each agent defines retrieval behaviour; the evidence selection step in the DILI analysis wizard decides what the current assessment uses.',
  },
  {
    id: 'session-review',
    title: 'Know the session sections',
    body: 'Preview the saved report, edit its source, inspect metadata, create a revision draft, or generate a timeline.',
    actionLabel: 'Open Clinical Sessions',
    action: 'sessions',
  },
  {
    id: 'timeline-review',
    title: 'Review timeline evidence and confidence',
    body: 'Filter by evidence or timing, then inspect an event to see its source, placement, and confidence rationale.',
    actionLabel: 'Open Clinical Sessions',
    action: 'sessions',
  },
  {
    id: 'data-maintenance',
    title: 'Refresh local evidence only when needed',
    body: 'Knowledge exposes explicit catalog and index maintenance actions. Review the current status before starting another update.',
    actionLabel: 'Open Knowledge',
    action: 'data',
  },
];
