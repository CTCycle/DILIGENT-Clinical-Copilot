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
      body: 'Run opens a short setup: pre-flight checks, the agent that runs the analysis, and a final review before it starts.',
      preferredPlacement: 'top',
    },
  ],
};

export const GUIDANCE_DEFINITIONS = {
  diliFirstAssessment: {
    id: 'dili-first-assessment',
    version: GUIDANCE_CONTENT_VERSION,
    title: 'Get started with DILI Agent',
    description: 'Structure the clinical input, pick an assessment agent, and review the pre-flight check before running.',
  },
} as const;

export const TIPS_AND_TRICKS: readonly TipDefinition[] = [
  {
    id: 'first-assessment',
    title: 'Run a first assessment',
    body: 'Structure the input, pick an assessment agent, then review the pre-flight check before running.',
    actionLabel: 'Show me',
    action: 'tour',
  },
  {
    id: 'model-roles',
    title: 'Choose agent models',
    body: 'Each DILI agent picks its own clinical, extraction and timeline models. Edit an agent in Agents, then save a new revision.',
    actionLabel: 'Open Agents',
    action: 'agents',
  },
  {
    id: 'rag-boundary',
    title: 'Evidence follows the agent',
    body: 'Each agent defines its evidence sources and retrieval policy. The agent you pick in the analysis wizard decides what the current assessment uses.',
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
