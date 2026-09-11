import { Component, input, output } from '@angular/core';

import {
  INSPECTION_UPDATE_ALL_TARGETS,
  InspectionUpdateAllConfigErrorMap,
  InspectionUpdateAllConfigLoadingMap,
  InspectionUpdateAllConfigMap,
  InspectionUpdateAllState,
  InspectionUpdateAllTarget,
  InspectionUpdateAllTargetState,
} from '../../../core/models/inspection-types';
import { ModalShellComponent } from '../../../components/modal-shell/modal-shell.component';
import {
  InspectionUpdateControlsComponent,
  InspectionUpdateFieldChange,
} from './inspection-update-controls.component';

type InspectionUpdateAllConfigChange = {
  target: InspectionUpdateAllTarget;
  change: InspectionUpdateFieldChange;
};

@Component({
  selector: 'app-inspection-update-all-modal',
  standalone: true,
  imports: [ModalShellComponent, InspectionUpdateControlsComponent],
  template: `
    <app-modal-shell
      [isOpen]="isOpen()"
      title="Update All Sources"
      subtitle="Configure LiverTox, RxNav, and DILIrank together. RAG updates remain separate."
      dialogClassName="modal-container inspection-modal inspection-update-all-modal"
      [footer]="true"
      closeLabel="Close all sources update dialog"
      (close)="close.emit()"
    >
      <div class="inspection-update-all">
        <div class="inspection-update-all-tabs" role="tablist" aria-label="Structured data sources">
          @for (target of targets; track target) {
            <button
              type="button"
              class="inspection-update-all-tab"
              role="tab"
              [attr.id]="tabId(target)"
              [attr.aria-controls]="panelId(target)"
              [attr.aria-selected]="activeTarget() === target"
              [attr.tabindex]="activeTarget() === target ? '0' : '-1'"
              (click)="targetChange.emit(target)"
              (keydown)="onTabKeydown($event, target)"
            >
              <span>{{ targetLabel(target) }}</span>
              <span class="inspection-update-all-tab-status">{{ targetStatusLabel(target) }}</span>
            </button>
          }
        </div>

        <section
          class="inspection-update-all-config"
          role="tabpanel"
          [attr.id]="panelId(activeTarget())"
          [attr.aria-labelledby]="tabId(activeTarget())"
          tabindex="0"
        >
          <div class="inspection-update-all-config-heading">
            <div>
              <p class="inspection-update-all-eyebrow">Source configuration</p>
              <h3>{{ targetLabel(activeTarget()) }}</h3>
            </div>
            <span class="inspection-update-all-config-note">Saved for this update run</span>
          </div>

          @if (configLoading()[activeTarget()]) {
            <p class="inspection-job-message">Loading {{ targetLabel(activeTarget()) }} configuration...</p>
          } @else if (configErrors()[activeTarget()]; as configError) {
            <div class="inspection-update-all-inline-error">
              <p class="inspection-error-text">{{ configError }}</p>
              <button class="btn btn-secondary inspection-mini-btn" type="button" (click)="retry.emit(activeTarget())">
                Retry configuration
              </button>
            </div>
          } @else if (configs()[activeTarget()]; as config) {
            <app-inspection-update-controls
              [target]="activeTarget()"
              [config]="config"
              [disabled]="isRunning()"
              (configChange)="configChange.emit({ target: activeTarget(), change: $event })"
            />
          } @else {
            <p class="inspection-job-message">Configuration is not available yet.</p>
          }
        </section>

        <section class="inspection-update-all-status" aria-live="polite" [attr.aria-busy]="isRunning()">
          <div class="inspection-update-all-status-heading">
            <div>
              <p class="inspection-update-all-eyebrow">Combined progress</p>
              <p class="inspection-update-all-message">{{ state().message || 'Ready to start.' }}</p>
            </div>
            <span class="inspection-update-all-percent">{{ normalizedProgress() }}%</span>
          </div>
          <div
            class="inspection-job-bar-track"
            role="progressbar"
            [attr.aria-valuenow]="normalizedProgress()"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-label="Combined update progress"
          >
            <div class="inspection-job-bar-fill" [style.width.%]="normalizedProgress()"></div>
          </div>

          <div class="inspection-update-all-source-list" aria-label="Source update status">
            @for (target of targets; track target) {
              <div class="inspection-update-all-source" [class.is-failed]="targetState(target).status === 'failed' || !!targetState(target).error">
                <div class="inspection-update-all-source-heading">
                  <strong>{{ targetLabel(target) }}</strong>
                  <span>{{ targetStatusLabel(target) }}</span>
                </div>
                <div class="inspection-job-bar-track">
                  <div class="inspection-job-bar-fill" [style.width.%]="targetProgress(target)"></div>
                </div>
                <p class="inspection-job-message">{{ targetMessage(target) }}</p>
                @if (targetState(target).error) {
                  <p class="inspection-error-text">{{ targetState(target).error }}</p>
                }
              </div>
            }
          </div>

          @if (validationError()) {
            <p class="inspection-error-text">{{ validationError() }}</p>
          }
        </section>
      </div>

      <div modal-footer class="inspection-pager-actions inspection-update-actions">
        <button class="btn btn-secondary inspection-mini-btn" type="button" (click)="close.emit()">Close</button>
        <button
          class="btn btn-secondary inspection-mini-btn"
          type="button"
          (click)="cancel.emit()"
          [disabled]="!isRunning()"
        >
          Cancel updates
        </button>
        <button
          class="btn btn-primary inspection-mini-btn"
          type="button"
          (click)="start.emit()"
          [disabled]="!canStart() || isRunning()"
        >
          Update All
        </button>
      </div>
    </app-modal-shell>
  `,
  styles: `
    :host {
      display: contents;
    }

    .inspection-update-all {
      display: flex;
      flex-direction: column;
      gap: var(--space-lg);
    }

    .inspection-update-all-tabs {
      display: flex;
      gap: var(--space-xs);
      overflow-x: auto;
      padding-bottom: 2px;
      border-bottom: 1px solid var(--color-border-subtle);
    }

    .inspection-update-all-tab {
      display: inline-flex;
      flex: 0 0 auto;
      align-items: center;
      gap: var(--space-sm);
      min-height: var(--control-height-md);
      padding: var(--space-sm) var(--space-md);
      border: 0;
      border-bottom: 2px solid transparent;
      background: transparent;
      color: var(--color-text-secondary);
      font: inherit;
      font-size: var(--font-sm);
      font-weight: 700;
      cursor: pointer;
    }

    .inspection-update-all-tab:hover,
    .inspection-update-all-tab:focus-visible {
      color: var(--color-text-primary);
      outline: none;
    }

    .inspection-update-all-tab[aria-selected="true"] {
      border-bottom-color: var(--color-brand-ui);
      color: var(--color-text-primary);
    }

    .inspection-update-all-tab-status {
      color: var(--color-text-muted);
      font-size: var(--font-xs);
      font-weight: 500;
      white-space: nowrap;
    }

    .inspection-update-all-config,
    .inspection-update-all-status {
      display: flex;
      flex-direction: column;
      gap: var(--space-md);
      padding: var(--space-md);
      border: 1px solid var(--color-border-subtle);
      border-radius: var(--radius-lg);
      background: var(--color-surface-alt);
    }

    .inspection-update-all-config:focus-visible {
      outline: 2px solid var(--color-brand-light);
      outline-offset: 2px;
    }

    .inspection-update-all-config-heading,
    .inspection-update-all-status-heading,
    .inspection-update-all-source-heading {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: var(--space-md);
    }

    .inspection-update-all-config-heading h3,
    .inspection-update-all-message,
    .inspection-update-all-eyebrow {
      margin: 0;
    }

    .inspection-update-all-config-heading h3 {
      font-size: var(--font-lg);
    }

    .inspection-update-all-eyebrow {
      margin-bottom: 2px;
      color: var(--color-text-muted);
      font-size: var(--font-xs);
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .inspection-update-all-config-note,
    .inspection-update-all-percent,
    .inspection-update-all-source-heading span {
      color: var(--color-text-muted);
      font-size: var(--font-xs);
    }

    .inspection-update-all-percent {
      color: var(--color-text-primary);
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }

    .inspection-update-all-inline-error {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: var(--space-md);
      flex-wrap: wrap;
    }

    .inspection-update-all-inline-error .inspection-error-text {
      margin: 0;
    }

    .inspection-update-all-source-list {
      display: grid;
      gap: var(--space-sm);
    }

    .inspection-update-all-source {
      display: grid;
      gap: var(--space-xs);
      padding-top: var(--space-sm);
      border-top: 1px solid var(--color-border-subtle);
    }

    .inspection-update-all-source:first-child {
      padding-top: 0;
      border-top: 0;
    }

    .inspection-update-all-source.is-failed .inspection-update-all-source-heading {
      color: var(--color-danger, #b91c1c);
    }

    .inspection-update-all-status .inspection-job-message {
      margin: 0;
    }

    @media (max-width: 680px) {
      .inspection-update-all-config-heading,
      .inspection-update-all-status-heading,
      .inspection-update-all-source-heading {
        align-items: flex-start;
        flex-direction: column;
        gap: var(--space-xs);
      }
    }
  `,
})
export class InspectionUpdateAllModalComponent {
  readonly targets = INSPECTION_UPDATE_ALL_TARGETS;
  readonly isOpen = input(false);
  readonly activeTarget = input.required<InspectionUpdateAllTarget>();
  readonly configs = input.required<InspectionUpdateAllConfigMap>();
  readonly configLoading = input.required<InspectionUpdateAllConfigLoadingMap>();
  readonly configErrors = input.required<InspectionUpdateAllConfigErrorMap>();
  readonly state = input.required<InspectionUpdateAllState>();
  readonly canStart = input(false);
  readonly validationError = input<string | null>(null);

  readonly close = output<void>();
  readonly targetChange = output<InspectionUpdateAllTarget>();
  readonly configChange = output<InspectionUpdateAllConfigChange>();
  readonly retry = output<InspectionUpdateAllTarget>();
  readonly start = output<void>();
  readonly cancel = output<void>();

  targetLabel(target: InspectionUpdateAllTarget): string {
    if (target === 'livertox') return 'LiverTox';
    if (target === 'rxnav') return 'RxNav';
    return 'DILIrank';
  }

  targetState(target: InspectionUpdateAllTarget): InspectionUpdateAllTargetState {
    return this.state().targets[target];
  }

  targetStatusLabel(target: InspectionUpdateAllTarget): string {
    const configError = this.configErrors()[target];
    if (configError) return 'Configuration error';
    const status = this.targetState(target).status;
    if (status === 'completed') return 'Completed';
    if (status === 'failed') return 'Failed';
    if (status === 'cancelled') return 'Cancelled';
    if (status === 'running') return 'Running';
    if (status === 'pending') return 'Queued';
    if (this.configLoading()[target]) return 'Loading';
    return 'Ready';
  }

  targetProgress(target: InspectionUpdateAllTarget): number {
    const state = this.targetState(target);
    if (state.status === 'completed') return 100;
    const value = Number(state.progress);
    if (!Number.isFinite(value)) return 0;
    return Math.min(100, Math.max(0, Math.round(value)));
  }

  targetMessage(target: InspectionUpdateAllTarget): string {
    const configError = this.configErrors()[target];
    if (configError) return configError;
    const state = this.targetState(target);
    if (state.error) return state.error;
    if (state.message.trim()) return state.message;
    return this.targetStatusLabel(target);
  }

  normalizedProgress(): number {
    const value = Number(this.state().progress);
    if (!Number.isFinite(value)) return 0;
    return Math.min(100, Math.max(0, Math.round(value)));
  }

  isRunning(): boolean {
    return this.state().phase === 'starting' || this.state().phase === 'running';
  }

  tabId(target: InspectionUpdateAllTarget): string {
    return `inspection-update-all-tab-${target}`;
  }

  panelId(target: InspectionUpdateAllTarget): string {
    return `inspection-update-all-panel-${target}`;
  }

  onTabKeydown(event: KeyboardEvent, target: InspectionUpdateAllTarget): void {
    const currentIndex = this.targets.indexOf(target);
    if (currentIndex < 0) return;
    let nextIndex: number | null = null;
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      nextIndex = (currentIndex + 1) % this.targets.length;
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      nextIndex = (currentIndex - 1 + this.targets.length) % this.targets.length;
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else if (event.key === 'End') {
      nextIndex = this.targets.length - 1;
    }
    if (nextIndex === null) return;
    event.preventDefault();
    this.targetChange.emit(this.targets[nextIndex]);
  }
}
