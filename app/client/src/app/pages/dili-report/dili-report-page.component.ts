// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, effect, inject, signal, untracked } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { map } from 'rxjs';

import { AnalysisProgressComponent } from '../../components/analysis-progress/analysis-progress.component';
import { ClinicalReportViewerComponent } from '../../components/clinical-report-viewer/clinical-report-viewer.component';
import { describeEvidenceSources } from '../../core/evidence-sources';
import { ClinicalSessionDetail } from '../../core/models/inspection-types';
import { fetchClinicalSessionDetail } from '../../core/services/clinical-sessions-api';
import { AppStateService } from '../../core/state/app-state.service';
import { formatUnknownError, isRecord } from '../../core/utils';
import { previewReport } from '../clinical-sessions/clinical-session-preview';

type ReportPageState = 'loading' | 'running' | 'interrupted' | 'completed' | 'empty' | 'error';

function parseSessionId(value: string | null): number | null {
  if (value === null) {
    return null;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

@Component({
  selector: 'app-dili-report-page',
  imports: [RouterLink, AnalysisProgressComponent, ClinicalReportViewerComponent],
  templateUrl: './dili-report-page.component.html',
  styleUrl: './dili-report-page.component.scss',
})
export class DiliReportPageComponent {
  private readonly stateService = inject(AppStateService);
  private readonly route = inject(ActivatedRoute);

  private readonly routeSessionId = toSignal(
    this.route.paramMap.pipe(map((params) => parseSessionId(params.get('sessionId')))),
    { initialValue: parseSessionId(this.route.snapshot.paramMap.get('sessionId')) },
  );
  private loadGeneration = 0;

  readonly detail = signal<ClinicalSessionDetail | null>(null);
  readonly loading = signal(false);
  readonly loadError = signal<string | null>(null);

  readonly vm = computed(() => this.stateService.state().diliAgent);
  readonly isStoredRoute = computed(() => this.routeSessionId() !== null);
  private readonly isLiveJob = computed(() => {
    const vm = this.vm();
    return !this.isStoredRoute() && (vm.isRunning || vm.isStarting);
  });
  private readonly targetSessionId = computed(() =>
    this.routeSessionId() ?? (this.isLiveJob() ? null : this.vm().sessionId),
  );

  readonly pageState = computed<ReportPageState>(() => {
    const vm = this.vm();
    if (this.isLiveJob()) return 'running';
    if (this.loadError() && !this.isLiveReportAvailable()) return 'error';
    if (this.loading() && !this.isLiveReportAvailable()) return 'loading';
    if (!this.isStoredRoute() && (vm.jobStatus === 'failed' || vm.jobStatus === 'cancelled')) {
      return 'interrupted';
    }
    return this.reportMarkdown() ? 'completed' : 'empty';
  });
  readonly reportMarkdown = computed(() => {
    if (this.isLiveReportAvailable()) return this.vm().message;
    const detail = this.detail();
    return detail ? previewReport(detail) : '';
  });
  readonly patientLabel = computed(() => {
    const detail = this.detail();
    const name = detail ? detail.patient_name : this.vm().form.patientName;
    return name?.trim() || 'Unnamed patient';
  });
  readonly visitDate = computed(() => {
    const detail = this.detail();
    return (detail ? detail.visit_date : this.vm().form.visitDate) || null;
  });
  readonly evidenceSources = computed(() => {
    const sources = this.detail()?.metadata['evidence_sources'];
    return Array.isArray(sources)
      ? describeEvidenceSources(sources.filter((item): item is string => typeof item === 'string'))
      : null;
  });
  readonly models = computed(() => {
    const detail = this.detail();
    if (!detail) return [];
    const overrides = detail.metadata['model_overrides'];
    const overridden = isRecord(overrides) ? overrides : {};
    return [
      { role: 'Text extraction', name: detail.text_extraction_model, overridden: Boolean(overridden['text_extraction_model']) },
      { role: 'Clinical', name: detail.clinical_model, overridden: Boolean(overridden['clinical_model']) },
    ].filter((model) => model.name);
  });

  constructor() {
    effect(() => {
      const sessionId = this.targetSessionId();
      untracked(() => void this.load(sessionId));
    });
  }

  private isLiveReportAvailable(): boolean {
    const vm = this.vm();
    return !this.isStoredRoute() && vm.jobStatus === 'completed' && Boolean(vm.message);
  }

  private async load(sessionId: number | null): Promise<void> {
    const generation = ++this.loadGeneration;
    this.detail.set(null);
    this.loadError.set(null);
    if (sessionId === null) {
      this.loading.set(false);
      return;
    }
    this.loading.set(true);
    try {
      const detail = await fetchClinicalSessionDetail(sessionId);
      if (generation === this.loadGeneration) {
        this.detail.set(detail);
      }
    } catch (error) {
      if (generation === this.loadGeneration) {
        this.loadError.set(formatUnknownError(error, 'Failed to load the report.'));
      }
    } finally {
      if (generation === this.loadGeneration) {
        this.loading.set(false);
      }
    }
  }
}
