// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, input, output, signal } from '@angular/core';

import { ReportTemplate, ReportTemplateFormat } from '../../core/models/agent-types';
import { REPORT_TEMPLATE_MAX_LENGTH } from './agent-form';

const FORMATS: readonly ReportTemplateFormat[] = ['md', 'json', 'xml', 'txt'];
const ACCEPT = FORMATS.map((format) => `.${format}`).join(',');

/**
 * Report template of an agent. A single value, supplied either by typing in the editor or by uploading a
 * file; an uploaded file lands in the same editor so it can be inspected and changed.
 */
@Component({
  selector: 'app-report-template-field',
  standalone: true,
  styleUrl: './agent-sections.scss',
  template: `
    <div class="template-field">
      <input #picker class="visually-hidden" type="file" [accept]="accept" [disabled]="disabled()"
        (change)="onFile($event)" />
      <div class="template-bar">
        @if (template(); as current) {
          <span class="template-state is-set">Configured</span>
          <span class="template-name">{{ current.file_name }}</span>
          <span class="field-hint">{{ current.format.toUpperCase() }} · {{ current.content.length }} characters</span>
        } @else {
          <span class="template-state">Not configured</span>
          <span class="field-hint">The default report format is used.</span>
        }
        <span class="template-actions">
          @if (template()) {
            <button type="button" class="btn btn-tertiary btn-compact" [disabled]="disabled()" (click)="clear()">
              Clear
            </button>
          }
          <span class="button-group" role="group" aria-label="Report template source">
            <button type="button" [class.is-active]="editing()" [attr.aria-pressed]="editing()"
              [disabled]="disabled() && !template()" (click)="editing.set(!editing())">
              {{ template() ? 'Edit' : 'Write' }}
            </button>
            <button type="button" [disabled]="disabled()" (click)="picker.click()">
              {{ template() ? 'Replace file' : 'Upload file' }}
            </button>
          </span>
        </span>
      </div>
      @if (editing()) {
        <div class="template-editor">
          <label class="field template-format">Format
            <select [disabled]="disabled()" (change)="setFormat($any($event.target).value)">
              @for (format of formats; track format) {
                <option [value]="format" [selected]="format === activeFormat()">{{ format.toUpperCase() }}</option>
              }
            </select>
          </label>
          <label class="field">
            <span class="visually-hidden">Report template content</span>
            <textarea class="template-text" rows="10" spellcheck="false" [disabled]="disabled()"
              [attr.maxlength]="maxLength" [value]="content()"
              placeholder="Write or paste the report structure, e.g. Markdown headings, a JSON schema or XML skeleton."
              (input)="setContent($any($event.target).value)"></textarea>
            <span class="field-counter">{{ content().length }} / {{ maxLength }}</span>
          </label>
        </div>
      }
      @if (error()) {
        <p class="field-error" role="alert">{{ error() }}</p>
      }
    </div>
  `,
})
export class ReportTemplateFieldComponent {
  readonly template = input<ReportTemplate | null>(null);
  readonly disabled = input(false);
  readonly templateChange = output<ReportTemplate | null>();

  readonly accept = ACCEPT;
  readonly formats = FORMATS;
  readonly maxLength = REPORT_TEMPLATE_MAX_LENGTH;
  readonly editing = signal(false);
  readonly error = signal('');
  private readonly draftFormat = signal<ReportTemplateFormat>('md');
  readonly content = computed(() => this.template()?.content ?? '');
  readonly activeFormat = computed(() => this.template()?.format ?? this.draftFormat());

  setContent(content: string): void {
    this.error.set('');
    if (!content.trim()) {
      this.templateChange.emit(null);
      return;
    }
    const format = this.activeFormat();
    this.templateChange.emit({ file_name: this.template()?.file_name ?? `template.${format}`, format, content });
  }

  setFormat(value: string): void {
    const format = FORMATS.find((candidate) => candidate === value) ?? 'md';
    this.draftFormat.set(format);
    const current = this.template();
    if (!current) return;
    const stem = current.file_name.replace(/\.[^.]+$/, '') || 'template';
    this.templateChange.emit({ ...current, format, file_name: `${stem}.${format}` });
  }

  async onFile(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = '';
    if (!file) return;
    const extension = file.name.includes('.') ? file.name.split('.').pop()!.toLowerCase() : '';
    const format = FORMATS.find((candidate) => candidate === extension);
    if (!format) {
      this.error.set('Unsupported file type. Use a .md, .json, .xml or .txt file.');
      return;
    }
    const content = await file.text();
    if (!content.trim()) {
      this.error.set('The template file is empty.');
      return;
    }
    if (content.length > REPORT_TEMPLATE_MAX_LENGTH) {
      this.error.set(`The template must be at most ${REPORT_TEMPLATE_MAX_LENGTH} characters.`);
      return;
    }
    this.error.set('');
    this.editing.set(true);
    this.templateChange.emit({ file_name: file.name, format, content });
  }

  clear(): void {
    this.error.set('');
    this.editing.set(false);
    this.templateChange.emit(null);
  }
}
