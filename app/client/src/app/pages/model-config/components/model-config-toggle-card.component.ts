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
  @Input() reasoningLevel = 0;
  @Input() disabled = false;
  @Input() showSettings = false;
  @Input() settingsLabel = 'Open settings';
  @Input() reasoningLayout = false;

  @Output() checkedChange = new EventEmitter<boolean>();
  @Output() reasoningLevelChange = new EventEmitter<number>();
  @Output() settingsClick = new EventEmitter<void>();

  get reasoningLevelLabel(): string {
    return ['Off', 'Low', 'Medium', 'High'][this.reasoningLevel] || 'Off';
  }

  handleReasoningLevelInput(event: Event): void {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
      return;
    }
    this.reasoningLevelChange.emit(Number(target.value));
  }

  handleSettingsClick(event: Event): void {
    event.stopPropagation();
    this.settingsClick.emit();
  }
}
