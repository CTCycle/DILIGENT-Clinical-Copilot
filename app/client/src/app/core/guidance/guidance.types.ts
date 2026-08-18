export type GuidanceStatus = 'seen' | 'dismissed' | 'skipped' | 'completed' | 'restarted';

export type GuidanceId =
  | 'dili-first-assessment'
  | 'dili-assessment-tour'
  | 'model-runtime-source'
  | 'model-rag-settings'
  | 'clinical-session-sections'
  | 'timeline-review-controls';

export interface GuidanceStateEntry {
  version: number;
  status: GuidanceStatus;
  updatedAt: string;
  restartCount?: number;
  lastRestartedAt?: string;
}

export interface PersistedGuidanceState {
  schemaVersion: 1;
  entries: Partial<Record<GuidanceId, GuidanceStateEntry>>;
}

export interface GuidanceDefinition {
  id: GuidanceId;
  version: number;
  title: string;
  description: string;
}

export interface GuidedTourStep {
  target: string;
  title: string;
  body: string;
  preferredPlacement?: 'top' | 'bottom' | 'left' | 'right';
}

export interface GuidedTourDefinition extends GuidanceDefinition {
  route: string;
  steps: readonly GuidedTourStep[];
}

export type TipAction = 'tour' | 'configurations' | 'sessions' | 'data';

export interface TipDefinition {
  id: string;
  title: string;
  body: string;
  actionLabel?: string;
  action?: TipAction;
}
