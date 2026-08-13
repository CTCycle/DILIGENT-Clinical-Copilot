import { Component, input } from '@angular/core';

@Component({
  selector: 'app-inspection-catalog-status',
  standalone: true,
  template: `
    @if (loading()) { <p class="inspection-loading-note">{{ loadingMessage() }}</p> }
    @if (loadingMore()) { <p class="inspection-loading-note">{{ loadingMoreMessage() }}</p> }
    @if (!loading() && hasMore()) { <p class="inspection-loading-note">{{ hasMoreMessage() }}</p> }
    @if (error()) { <p class="inspection-error-text">{{ error() }}</p> }
  `,
  styles: ':host { display: contents; }',
})
export class InspectionCatalogStatusComponent {
  readonly loading = input.required<boolean>();
  readonly loadingMessage = input.required<string>();
  readonly loadingMore = input.required<boolean>();
  readonly loadingMoreMessage = input.required<string>();
  readonly hasMore = input.required<boolean>();
  readonly hasMoreMessage = input('Scroll to load more...');
  readonly error = input<string | null>(null);
}
