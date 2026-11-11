import { CommonModule } from '@angular/common';
import { Component, DestroyRef, ElementRef, ViewChild, effect, inject, signal, untracked, viewChild } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router } from '@angular/router';

import { GUIDANCE_DEFINITIONS, DILI_ASSESSMENT_TOUR } from '../../core/guidance/guidance-content';
import { FeatureTipComponent } from '../../core/guidance/feature-tip.component';
import { GuidanceTourService } from '../../core/guidance/guidance-tour.service';
import { fetchClinicalSectionTemplate } from '../../core/services/clinical-api';
import {
  ClinicalSessionDraftService,
  DraftNotEditableError,
} from '../../core/services/clinical-session-draft.service';
import { AppStateService } from '../../core/state/app-state.service';
import { formatUnknownError, normalizeVisitDateInput } from '../../core/utils';
import { AnalysisWizardComponent } from './components/analysis-wizard/analysis-wizard.component';
import { SourceSelection } from './components/analysis-wizard/section-preview';

const todayIso = new Date().toISOString().slice(0, 10);
const CLINICAL_INPUT_TEMPLATE_FALLBACK = `Chief concern:
History of present illness:
Current medications (dose/start date):
Recent labs (ALT/AST/ALP/Total bilirubin with dates):
Relevant imaging or procedures:
Comorbidities / liver risk factors:
Timeline of symptoms and treatment changes:
Working clinical question:`;

function parseDraftId(value: string | null): number | null {
  const parsed = Number(value);
  return value !== null && Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

@Component({
  selector: 'app-dili-agent-page',
  imports: [CommonModule, FormsModule, AnalysisWizardComponent, FeatureTipComponent],
  templateUrl: './dili-agent-page.component.html',
  styleUrl: './dili-agent-page.component.scss',
})
export class DiliAgentPageComponent {
  @ViewChild('patientImageInput') private patientImageInput?: ElementRef<HTMLInputElement>;

  readonly stateService = inject(AppStateService);
  readonly guidanceDefinitions = GUIDANCE_DEFINITIONS;
  private readonly router = inject(Router);
  private readonly route = inject(ActivatedRoute);
  private readonly destroyRef = inject(DestroyRef);
  private readonly guidanceTour = inject(GuidanceTourService);
  private readonly drafts = inject(ClinicalSessionDraftService);
  private readonly wizard = viewChild(AnalysisWizardComponent);

  readonly wizardOpen = signal(false);
  readonly todayIso = todayIso;
  readonly clinicalInputTemplate = signal('');
  readonly loadingDraft = signal(false);
  readonly pageError = signal<string | null>(null);

  constructor() {
    void this.loadClinicalSectionTemplate();
    this.route.queryParamMap
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((params) => {
        const requested = parseDraftId(params.get('session'));
        if (requested !== null && requested !== this.vm.draftSessionId) {
          void this.openDraft(requested);
        }
      });
    // Keep `?session=` in step with the draft so a refresh restores it.
    effect(() => {
      const draftId = this.stateService.state().diliAgent.draftSessionId;
      untracked(() => {
        const current = parseDraftId(this.route.snapshot.queryParamMap.get('session'));
        if (draftId !== current && !this.loadingDraft()) {
          void this.router.navigate([], {
            queryParams: { session: draftId },
            queryParamsHandling: 'merge',
            replaceUrl: true,
          });
        }
      });
    });
  }

  get vm() {
    return this.stateService.state().diliAgent;
  }

  startAssessmentTour(): void {
    this.guidanceTour.start(DILI_ASSESSMENT_TOUR);
  }

  openConfigurations(): void {
    void this.router.navigateByUrl('/agents');
  }

  handleFormChange<K extends keyof typeof this.vm.form>(key: K, value: (typeof this.vm.form)[K]): void {
    this.stateService.updateDiliAgent({
      form: { ...this.vm.form, [key]: value },
      isDirty: true,
      saveStatus: this.vm.saveStatus === 'conflict' ? 'conflict' : 'idle',
    });
  }

  private async loadClinicalSectionTemplate(): Promise<void> {
    try {
      const response = await fetchClinicalSectionTemplate();
      if (response.template?.trim()) {
        this.clinicalInputTemplate.set(response.template.trim());
        return;
      }
    } catch {}
    this.clinicalInputTemplate.set(CLINICAL_INPUT_TEMPLATE_FALLBACK);
  }

  handleVisitDateChange(value: string): void {
    this.handleFormChange('visitDate', normalizeVisitDateInput(value));
  }

  openPatientImagePicker(): void {
    this.patientImageInput?.nativeElement.click();
  }

  handlePatientImageSelection(event: Event): void {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
      return;
    }
    const file = target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = typeof reader.result === 'string' ? reader.result : null;
      this.handleFormChange('patientImageDataUrl', dataUrl);
    };
    reader.readAsDataURL(file);
  }

  closeWizard(): void {
    this.wizardOpen.set(false);
  }

  fixInput(targetId: string | null): void {
    this.closeWizard();
    globalThis.setTimeout(() => document.getElementById(targetId ?? 'clinical-input')?.focus());
  }

  editSource(selection: SourceSelection | null): void {
    this.closeWizard();
    globalThis.setTimeout(() => {
      const input = document.getElementById('clinical-input') as HTMLTextAreaElement | null;
      input?.focus();
      if (input && selection && selection.end <= input.value.length) {
        input.setSelectionRange(selection.start, selection.end);
      }
    });
  }

  async saveDraft(): Promise<void> {
    this.pageError.set(null);
    await this.drafts.save();
  }

  async reloadDraft(): Promise<void> {
    const draftId = this.vm.draftSessionId;
    if (draftId !== null) {
      await this.openDraft(draftId);
    }
  }

  clearAll(): void {
    if (
      this.vm.isDirty &&
      !globalThis.confirm('Clear all case input? Unsaved changes will be discarded and cannot be recovered.')
    ) {
      return;
    }
    this.pageError.set(null);
    this.wizard()?.reset();
    this.drafts.reset();
  }

  runAnalysis(): void {
    this.wizardOpen.set(true);
  }

  private async openDraft(sessionId: number): Promise<void> {
    this.loadingDraft.set(true);
    this.pageError.set(null);
    try {
      await this.drafts.load(sessionId);
    } catch (error) {
      if (error instanceof DraftNotEditableError) {
        void this.router.navigate(['/report', sessionId], { replaceUrl: true });
      } else {
        this.pageError.set(formatUnknownError(error, 'The draft could not be opened.'));
      }
    } finally {
      this.loadingDraft.set(false);
    }
  }

  get draftStatusText(): string {
    const { saveStatus, isDirty, draftSessionId, lastSavedAt } = this.vm;
    switch (saveStatus) {
      case 'saving':
        return 'Saving draft…';
      case 'conflict':
        return 'This draft changed elsewhere or was already launched. Reload it before saving.';
      case 'error':
        return 'Unsaved changes. The draft could not be saved.';
      default:
        break;
    }
    if (isDirty) {
      return draftSessionId === null ? 'Not saved yet' : 'Unsaved changes';
    }
    if (draftSessionId !== null && lastSavedAt) {
      const time = new Date(lastSavedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      return `Draft #${draftSessionId} · Saved ${time}`;
    }
    return '';
  }

  get patientNameLabel(): string {
    return this.vm.form.patientName.trim() || 'Unnamed patient';
  }

  get patientImageDataUrl(): string | null {
    return this.vm.form.patientImageDataUrl;
  }
}
