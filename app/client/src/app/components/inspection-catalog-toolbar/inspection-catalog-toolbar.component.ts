import { CommonModule } from '@angular/common';
import { Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { InspectionProgressButtonComponent } from '../inspection-progress-button/inspection-progress-button.component';

@Component({
  selector: 'app-inspection-catalog-toolbar',
  standalone: true,
  imports: [CommonModule, FormsModule, InspectionProgressButtonComponent],
  template: `
    <div class="inspection-widget-header">
      <div class="inspection-controls inspection-controls-half inspection-controls-knowledge">
        <input
          type="search"
          class="inspection-search"
          [placeholder]="searchPlaceholder()"
          [ngModel]="searchValue()"
          (ngModelChange)="searchChange.emit($event)"
          [attr.aria-label]="searchAriaLabel()"
        />
      </div>
      <div class="inspection-widget-header-actions inspection-header-actions-fixed">
        <ng-content select="[inspectionToolbarBeforeActions]"></ng-content>
        <app-inspection-progress-button
          [buttonLabel]="updateButtonLabel()"
          [inProgress]="updateInProgress()"
          [progress]="updateProgress()"
          [progressMessage]="updateProgressMessage()"
          [disabled]="updateDisabled()"
          (activated)="updateClick.emit()"
        />
      </div>
    </div>
  `,
})
export class InspectionCatalogToolbarComponent {
  readonly searchPlaceholder = input.required<string>();
  readonly searchAriaLabel = input.required<string>();
  readonly searchValue = input.required<string>();
  readonly updateButtonLabel = input.required<string>();
  readonly updateInProgress = input(false);
  readonly updateProgress = input(0);
  readonly updateProgressMessage = input('');
  readonly updateDisabled = input(false);

  readonly searchChange = output<string>();
  readonly updateClick = output<void>();
}
