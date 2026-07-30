import { CommonModule } from '@angular/common';
import { Component, Input, OnChanges, OnDestroy, OnInit, SimpleChanges, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { ModalShellComponent } from '../../../components/modal-shell/modal-shell.component';
import { ClinicalSessionDetail, CloudProvider, InspectionSessionTimelinePreview, ModelConfigStateResponse } from '../../../core/models/types';
import { deleteInspectionSessionTimeline, fetchInspectionSessionTimelineJobStatus, fetchInspectionSessionTimelineList, startInspectionSessionTimelineJob } from '../../../core/services/inspection-api';
import { fetchModelConfigState } from '../../../core/services/model-config-api';
import { formatUnknownError } from '../../../core/utils';

@Component({
  selector: 'app-clinical-session-timeline-workspace',
  standalone: true,
  imports: [CommonModule, FormsModule, ModalShellComponent],
  templateUrl: './clinical-session-timeline-workspace.component.html',
  styleUrl: './clinical-session-timeline-workspace.component.scss',
})
export class ClinicalSessionTimelineWorkspaceComponent implements OnInit, OnChanges, OnDestroy {
  private readonly router = inject(Router);
  @Input({ required: true }) session!: ClinicalSessionDetail;

  readonly timelinePreviews = signal<InspectionSessionTimelinePreview[]>([]);
  readonly timelineListLoading = signal(false);
  readonly timelineListError = signal<string | null>(null);
  readonly modelConfig = signal<ModelConfigStateResponse | null>(null);
  readonly modelConfigLoading = signal(false);
  readonly modelConfigError = signal<string | null>(null);
  readonly provider = signal<'ollama' | CloudProvider>('ollama');
  readonly modelName = signal('');
  readonly generationRunning = signal(false);
  readonly generationStatus = signal<string | null>(null);
  readonly generationError = signal<string | null>(null);
  readonly generationJobId = signal<string | null>(null);
  readonly generationProgress = signal(0);
  readonly generationProgressMessage = signal<string | null>(null);
  readonly timelinePendingDeletion = signal<InspectionSessionTimelinePreview | null>(null);
  readonly deletingTimelineId = signal<number | null>(null);
  readonly deleteError = signal<string | null>(null);
  readonly settingsRestoreNotice = signal<string | null>(null);
  private generationPollTimer: ReturnType<typeof setTimeout> | null = null;
  private generationPollToken = 0;
  private readonly timelineJobStoragePrefix = 'clinical-session-timeline-job:';

  readonly availableLocalModels = computed(() => (this.modelConfig()?.local_models ?? []).filter((model) => model.available_in_ollama));
  readonly availableCloudProviders = computed(() => this.modelConfig()?.cloud_providers ?? []);
  readonly providerOptions = computed(() => [{ id: 'ollama', display_name: 'Ollama' }, ...this.availableCloudProviders()]);
  readonly availableModels = computed(() => this.provider() === 'ollama'
    ? this.availableLocalModels().map((model) => ({ id: model.name, display_name: model.name }))
    : this.availableCloudProviders().find((candidate) => candidate.id === this.provider())?.models ?? []);
  readonly selectedConfigurationLabel = computed(() => {
    const provider = this.provider() === 'ollama' ? 'Ollama' : this.availableCloudProviders().find((item) => item.id === this.provider())?.display_name || this.provider();
    return [provider, this.modelName()].filter(Boolean).join(' · ');
  });
  readonly canGenerate = computed(() => Boolean(this.modelName()) && !this.generationRunning() && !this.modelConfigLoading() && !this.modelConfigError());
  readonly sortedTimelinePreviews = computed(() => [...this.timelinePreviews()].sort((a, b) => Date.parse(b.generated_at) - Date.parse(a.generated_at)));

  ngOnInit(): void { this.resetAndLoad(); }
  ngOnChanges(changes: SimpleChanges): void { if (changes['session'] && !changes['session'].firstChange) this.resetAndLoad(); }
  ngOnDestroy(): void { this.stopTimelinePolling(); }

  private resetAndLoad(): void {
    this.stopTimelinePolling();
    this.timelinePreviews.set([]); this.timelineListError.set(null); this.generationError.set(null); this.generationStatus.set(null); this.settingsRestoreNotice.set(null);
    this.generationJobId.set(null); this.generationProgress.set(0); this.generationProgressMessage.set(null); this.generationRunning.set(false);
    void this.loadModelConfiguration(); void this.loadTimelineHistory(); void this.restoreTimelineJob();
  }

  async loadModelConfiguration(): Promise<void> {
    this.modelConfigLoading.set(true); this.modelConfigError.set(null);
    try {
      const config = await fetchModelConfigState();
      this.modelConfig.set(config);
      this.provider.set(config.use_cloud_services
        ? config.cloud_providers.find((item) => item.id === config.llm_provider)?.id || config.cloud_providers[0]?.id || 'ollama'
        : 'ollama');
      this.correctModelSelection(config.use_cloud_services ? (config.cloud_model || '') : (config.text_extraction_model || ''));
    } catch (error) { this.modelConfigError.set(formatUnknownError(error, 'Unable to load timeline model options.')); }
    finally { this.modelConfigLoading.set(false); }
  }

  async loadTimelineHistory(): Promise<void> {
    if (!this.session?.session_id) return;
    this.timelineListLoading.set(true); this.timelineListError.set(null);
    try { this.timelinePreviews.set((await fetchInspectionSessionTimelineList(this.session.session_id)).items); }
    catch (error) { this.timelineListError.set(formatUnknownError(error, 'Unable to load timeline history.')); }
    finally { this.timelineListLoading.set(false); }
  }

  setProvider(value: string): void {
    if (value === 'ollama' || this.availableCloudProviders().some((item) => item.id === value)) {
      this.provider.set(value as 'ollama' | CloudProvider); this.correctModelSelection();
    }
  }
  setModelName(value: string): void { this.modelName.set(value); }

  async generateTimeline(): Promise<void> {
    if (!this.canGenerate()) return;
    this.generationRunning.set(true); this.generationError.set(null); this.generationStatus.set('Starting timeline generation…'); this.generationProgress.set(0); this.generationProgressMessage.set('Preparing timeline generation');
    const cloud = this.provider() !== 'ollama';
    try {
      const response = await startInspectionSessionTimelineJob(this.session.session_id, { force_regenerate: true, model_overrides: cloud ? { use_cloud_services: true, llm_provider: String(this.provider()), cloud_model: this.modelName() } : { use_cloud_services: false, text_extraction_model: this.modelName() } });
      this.attachToTimelineJob(response.job_id, response.poll_interval);
    } catch (error) { this.generationRunning.set(false); this.generationStatus.set(null); this.generationError.set(formatUnknownError(error, 'Unable to start timeline generation.')); }
  }

  private timelineJobStorageKey(): string { return `${this.timelineJobStoragePrefix}${this.session.session_id}`; }
  private persistTimelineJob(jobId: string): void { if (typeof localStorage !== 'undefined') localStorage.setItem(this.timelineJobStorageKey(), jobId); }
  private clearPersistedTimelineJob(): void { if (typeof localStorage !== 'undefined') localStorage.removeItem(this.timelineJobStorageKey()); }
  private restoreTimelineJob(): void {
    if (!this.session?.session_id || typeof localStorage === 'undefined') return;
    const jobId = localStorage.getItem(this.timelineJobStorageKey());
    if (jobId) this.attachToTimelineJob(jobId, 1, true);
  }
  private attachToTimelineJob(jobId: string, pollIntervalSeconds: number, restoring = false): void {
    this.stopTimelinePolling(); this.persistTimelineJob(jobId); this.generationJobId.set(jobId); this.generationRunning.set(true);
    this.generationStatus.set(restoring ? 'Restoring timeline generation…' : 'Generating timeline…'); this.pollTimelineJob(jobId, pollIntervalSeconds);
  }
  private stopTimelinePolling(): void { this.generationPollToken += 1; if (this.generationPollTimer !== null) { clearTimeout(this.generationPollTimer); this.generationPollTimer = null; } }
  private pollTimelineJob(jobId: string, pollIntervalSeconds: number): void {
    const token = ++this.generationPollToken;
    const delayMs = Math.max(500, Math.round((Number.isFinite(pollIntervalSeconds) ? pollIntervalSeconds : 1) * 1000));
    let consecutiveErrors = 0;
    const poll = async (): Promise<void> => {
      if (token !== this.generationPollToken || this.generationJobId() !== jobId) return;
      try {
        const job = await fetchInspectionSessionTimelineJobStatus(this.session.session_id, jobId);
        if (token !== this.generationPollToken || this.generationJobId() !== jobId) return;
        consecutiveErrors = 0; this.generationProgress.set(Math.max(0, Math.min(100, Number(job.progress) || 0)));
        const message = job.result?.progress_message;
        if (typeof message === 'string' && message) this.generationProgressMessage.set(message);
        if (job.status === 'completed') {
          this.generationProgress.set(100); this.generationRunning.set(false); this.generationStatus.set('Timeline generated and saved.'); this.clearPersistedTimelineJob(); this.generationJobId.set(null); await this.loadTimelineHistory(); return;
        }
        if (job.status === 'failed' || job.status === 'cancelled') {
          this.generationRunning.set(false); this.generationStatus.set(null); this.generationError.set(job.error || 'Timeline generation did not complete.'); this.clearPersistedTimelineJob(); this.generationJobId.set(null); return;
        }
      } catch (error) {
        consecutiveErrors += 1;
        if (consecutiveErrors >= 5) { this.generationRunning.set(false); this.generationStatus.set(null); this.generationError.set(formatUnknownError(error, 'Unable to restore timeline generation status.')); this.clearPersistedTimelineJob(); this.generationJobId.set(null); return; }
      }
      this.generationPollTimer = setTimeout(() => void poll(), delayMs);
    };
    void poll();
  }

  openTimeline(preview: InspectionSessionTimelinePreview): void { if (preview.timeline_id) void this.router.navigate(['/sessions', this.session.session_id, 'timetable', preview.timeline_id]); }
  useTimelineSettings(preview: InspectionSessionTimelinePreview): void {
    this.settingsRestoreNotice.set(null); const isCloud = preview.source_kind === 'cloud';
    const cloudProvider = isCloud && preview.model_provider && this.availableCloudProviders().some((item) => item.id === preview.model_provider)
      ? preview.model_provider as CloudProvider
      : 'ollama';
    this.provider.set(cloudProvider);
    const choices = this.availableModels().map((item) => item.id);
    if (preview.source_model && choices.includes(preview.source_model)) this.modelName.set(preview.source_model);
    else { this.modelName.set(choices[0] || ''); this.settingsRestoreNotice.set('The original model is unavailable; a compatible currently available model was selected.'); }
  }
  requestTimelineDeletion(preview: InspectionSessionTimelinePreview): void { this.timelinePendingDeletion.set(preview); this.deleteError.set(null); }
  cancelTimelineDeletion(): void { if (this.deletingTimelineId() === null) { this.timelinePendingDeletion.set(null); this.deleteError.set(null); } }
  async confirmTimelineDeletion(): Promise<void> {
    const preview = this.timelinePendingDeletion(); if (!preview?.timeline_id) return;
    this.deletingTimelineId.set(preview.timeline_id); this.deleteError.set(null);
    try { await deleteInspectionSessionTimeline(this.session.session_id, preview.timeline_id); this.timelinePreviews.update((items) => items.filter((item) => item.timeline_id !== preview.timeline_id)); this.timelinePendingDeletion.set(null); }
    catch (error) { this.deleteError.set(formatUnknownError(error, 'Unable to delete timeline.')); }
    finally { this.deletingTimelineId.set(null); }
  }

  timelineRangeLabel(preview: InspectionSessionTimelinePreview): string { return preview.start_date && preview.end_date ? `${preview.start_date} – ${preview.end_date}` : preview.start_date || preview.end_date || 'No dated events'; }
  timelineProviderLabel(preview: InspectionSessionTimelinePreview): string { return preview.source_kind === 'cloud' ? preview.model_provider || 'Cloud provider not recorded' : 'Local'; }
  timelineModelLabel(preview: InspectionSessionTimelinePreview): string { return preview.source_model || 'Model not recorded'; }
  timelineQualityLabel(preview: InspectionSessionTimelinePreview): string { return preview.generation_status === 'fallback' ? 'Fallback chronology' : 'LLM generated'; }

  private correctModelSelection(preferred = this.modelName()): void {
    const values = this.availableModels().map((item) => item.id);
    this.modelName.set(values.includes(preferred) ? preferred : values[0] || '');
  }
}
