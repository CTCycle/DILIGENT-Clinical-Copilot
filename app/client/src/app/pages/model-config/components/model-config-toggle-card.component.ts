import { Component, EventEmitter, Input, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';

export type ModelConfigToggleIcon = 'rag' | 'reasoning';

@Component({
  selector: 'app-model-config-toggle-card',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './model-config-toggle-card.component.html',
  styleUrl: './model-config-toggle-card.component.scss',
})
export class ModelConfigToggleCardComponent {
  @Input({ required: true }) title = '';
  @Input({ required: true }) description = '';
  @Input({ required: true }) ariaLabel = '';
  @Input({ required: true }) icon: ModelConfigToggleIcon = 'rag';
  @Input() checked = false;
  @Input() disabled = false;
  @Input() showSettings = false;
  @Input() settingsLabel = 'Open settings';
  @Input() reasoningLayout = false;

  @Output() checkedChange = new EventEmitter<boolean>();
  @Output() settingsClick = new EventEmitter<void>();

  handleSettingsClick(event: Event): void {
    event.stopPropagation();
    this.settingsClick.emit();
  }
}
