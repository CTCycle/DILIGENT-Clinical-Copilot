import { Component, EventEmitter, Input, Output } from '@angular/core';
import { LucideBookOpen, LucideBraces, LucideSave } from '@lucide/angular';

import { EditorCommandEvent, EditorViewMode } from '../clinical-sessions.types';

@Component({
  selector: 'app-clinical-session-editor-toolbar',
  standalone: true,
  imports: [LucideBookOpen, LucideBraces, LucideSave],
  template: `
    <div class="clinical-session-editor-toolbar" role="toolbar" aria-label="Text editor toolbar">
      <div class="clinical-session-toolbar-group clinical-session-view-mode" aria-label="Editor view mode">
        <button
          type="button"
          class="clinical-session-icon-toggle"
          [class.is-active]="viewMode === 'source'"
          (click)="viewModeChange.emit('source')"
          title="Source"
          aria-label="Source view"
        >
          <svg lucideBraces size="16"></svg>
        </button>
        <button
          type="button"
          class="clinical-session-icon-toggle"
          [class.is-active]="viewMode === 'rendered'"
          (click)="viewModeChange.emit('rendered')"
          title="Rendered"
          aria-label="Rendered view"
        >
          <svg lucideBookOpen size="16"></svg>
        </button>
      </div>
      <div class="clinical-session-toolbar-group">
        <button type="button" (click)="fontSizeDelta.emit(-1)" [disabled]="fontSize <= 12" aria-label="Decrease font size">A-</button>
        <span class="clinical-session-editor-font-size">{{ fontSize }}px</span>
        <button type="button" (click)="fontSizeDelta.emit(1)" [disabled]="fontSize >= 22" aria-label="Increase font size">A+</button>
      </div>
      <div class="clinical-session-toolbar-group clinical-session-toolbar-group-format">
        <select aria-label="Text format" (change)="handleFormatChange($event)">
          <option value="p">Format</option>
          <option value="h1">Heading 1</option>
          <option value="h2">Heading 2</option>
          <option value="h3">Heading 3</option>
          <option value="p">Paragraph</option>
          <option value="blockquote">Quote</option>
          <option value="pre">Code block</option>
        </select>
      </div>
      <div class="clinical-session-toolbar-group clinical-session-toolbar-group-editing">
        <button type="button" (click)="emitCommand('undo')" aria-label="Undo">↶</button>
        <button type="button" (click)="emitCommand('redo')" aria-label="Redo">↷</button>
        <button type="button" (click)="emitCommand('bold')" aria-label="Bold"><strong>B</strong></button>
        <button type="button" (click)="emitCommand('italic')" aria-label="Italic"><em>I</em></button>
        <button type="button" (click)="emitCommand('underline')" aria-label="Underline"><u>U</u></button>
        <button type="button" (click)="emitCommand('strikeThrough')" aria-label="Strike"><s>S</s></button>
        <button type="button" (click)="emitCommand('hiliteColor', '#fff59d')" aria-label="Highlight">HL</button>
        <button type="button" (click)="emitCommand('insertUnorderedList')" aria-label="Bullet list">•</button>
        <button type="button" (click)="emitCommand('insertOrderedList')" aria-label="Number list">1.</button>
        <button type="button" (click)="emitCommand('justifyLeft')" aria-label="Align left">≡</button>
        <button type="button" (click)="emitCommand('justifyCenter')" aria-label="Align center">≣</button>
        <button type="button" (click)="emitCommand('justifyRight')" aria-label="Align right">☰</button>
      </div>
      <div class="clinical-session-toolbar-group clinical-session-toolbar-group-standalone">
        <button type="button" (click)="insertLink.emit()" aria-label="Insert link">🔗</button>
        <button type="button" (click)="clearFormatting.emit()" aria-label="Clear formatting">⌫</button>
        <button type="button" (click)="removeSelection.emit()" aria-label="Remove selected text">✖</button>
        <button
          type="button"
          class="btn btn-primary clinical-session-save-icon-button"
          (click)="save.emit()"
          aria-label="Save manual report edit"
          title="Save manual report edit"
        >
          <svg lucideSave size="16"></svg>
        </button>
      </div>
    </div>
  `,
})
export class ClinicalSessionEditorToolbarComponent {
  @Input({ required: true }) viewMode: EditorViewMode = 'source';
  @Input({ required: true }) fontSize = 16;

  @Output() viewModeChange = new EventEmitter<EditorViewMode>();
  @Output() fontSizeDelta = new EventEmitter<number>();
  @Output() editorCommand = new EventEmitter<EditorCommandEvent>();
  @Output() insertLink = new EventEmitter<void>();
  @Output() clearFormatting = new EventEmitter<void>();
  @Output() removeSelection = new EventEmitter<void>();
  @Output() save = new EventEmitter<void>();

  emitCommand(command: EditorCommandEvent['command'], value?: string): void {
    this.editorCommand.emit({ command, value });
  }

  handleFormatChange(event: Event): void {
    const target = event.target;
    if (!(target instanceof HTMLSelectElement)) {
      return;
    }
    this.emitCommand('formatBlock', target.value);
  }
}
