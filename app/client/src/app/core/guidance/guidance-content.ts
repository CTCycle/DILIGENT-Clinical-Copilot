import { GuidedTourDefinition, TipDefinition } from './guidance.types';

export const GUIDANCE_CONTENT_VERSION = 1;
const DILI_ASSESSMENT_TOUR_VERSION = 2;

export const DILI_ASSESSMENT_TOUR: GuidedTourDefinition = {
  id: 'dili-assessment-tour',
  version: DILI_ASSESSMENT_TOUR_VERSION,
  title: 'Run a DILI assessment',
  description: 'A short tour of the four controls that shape a first assessment.',
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
      target: '[data-guidance-target="dili-rag-toggle"]',
      title: 'Add indexed evidence when useful',
      body: 'RAG adds references from the configured local index to this run. The checkbox only affects the assessment you are about to start.',
      preferredPlacement: 'left',
    },
    {
      target: '[data-guidance-target="dili-review-run"]',
      title: 'Review, then run',
      body: 'Run opens a pre-flight review first. Fix blocking items or continue only after accepting any warnings.',
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
  modelRuntimeSource: {
    id: 'model-runtime-source',
    version: GUIDANCE_CONTENT_VERSION,
    title: 'Runtime source',
    description: 'Choose Local for Ollama on this machine or Cloud for a configured external provider. This setting controls the runtime used by new work.',
  },
  modelRagSettings: {
    id: 'model-rag-settings',
    version: GUIDANCE_CONTENT_VERSION,
    title: 'RAG settings',
    description: 'These settings control chunking, embeddings, retrieval, and reranking. The DILI Agent checkbox still decides whether a run uses the index.',
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
    title: 'Choose model roles',
    body: 'The clinical role and text-extraction role can use different models. Choose each role in the catalog, then save the configuration.',
    actionLabel: 'Open Configurations',
    action: 'configurations',
  },
  {
    id: 'rag-boundary',
    title: 'Control RAG per assessment',
    body: 'RAG settings define retrieval behaviour; the DILI Agent checkbox decides whether the current assessment uses indexed evidence.',
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
    body: 'Data Inspection exposes explicit catalog and index maintenance actions. Review the current status before starting another update.',
    actionLabel: 'Open Data Inspection',
    action: 'data',
  },
];
