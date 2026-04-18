import { CommonModule } from '@angular/common';
import { Component, input, output } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-inspection-catalog-toolbar',
  standalone: true,
  imports: [CommonModule, FormsModule],
  styles: [
    `
      :host {
        display: contents;
      }
    `,
  ],
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
        <button class="btn btn-secondary inspection-mini-btn" type="button" (click)="updateClick.emit()">
          {{ updateButtonLabel() }}
        </button>
      </div>
    </div>
  `,
})
export class InspectionCatalogToolbarComponent {
  readonly searchPlaceholder = input.required<string>();
  readonly searchAriaLabel = input.required<string>();
  readonly searchValue = input.required<string>();
  readonly updateButtonLabel = input.required<string>();

  readonly searchChange = output<string>();
  readonly updateClick = output<void>();
}
