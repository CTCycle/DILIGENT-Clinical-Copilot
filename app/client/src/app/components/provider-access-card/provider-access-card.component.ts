import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

export type ProviderAccessCardVariant = 'selectable' | 'compact';

@Component({
  selector: 'app-provider-access-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './provider-access-card.component.html',
})
export class ProviderAccessCardComponent {
  @Input() variant: ProviderAccessCardVariant = 'selectable';
  @Input() label = '';
  @Input() isActive = false;
  @Input() disabled = false;
  @Input() hint = '';
  @Input() manageKeyAriaLabel = 'Manage provider access keys';

  @Output() select = new EventEmitter<void>();
  @Output() manageKeys = new EventEmitter<void>();

  get isCompact(): boolean {
    return this.variant === 'compact';
  }

  get className(): string {
    return [
      'model-config-provider-card',
      !this.isCompact && this.isActive ? 'is-active' : '',
      this.isCompact ? 'model-config-provider-card-compact' : '',
    ]
      .filter(Boolean)
      .join(' ');
  }
}
