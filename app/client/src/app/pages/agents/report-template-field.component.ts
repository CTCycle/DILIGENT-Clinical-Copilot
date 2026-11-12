// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, input, output, signal } from '@angular/core';

import { ReportTemplate, ReportTemplateFormat } from '../../core/models/agent-types';
import { REPORT_TEMPLATE_MAX_LENGTH } from './agent-form';

const FORMATS: readonly ReportTemplateFormat[] = ['md', 'json', 'xml', 'txt'];
const ACCEPT = FORMATS.map((format) => `.${format}`).join(',');

/** Upload, preview, replace and remove the report template of an agent. */
@Component({
  selector: 'app-report-template-field',
  standalone: true,
  styleUrl: './agent-sections.scss',
  template: `
    <div class="template-field">
      <input #picker class="visually-hidden" type="file" [accept]="accept" [disabled]="disabled()"
        (change)="onFile($event)" />
      @if (template(); as current) {
        <div class="template-summary">
          <div>
            <strong>{{ current.file_name }}</strong>
            <span class="field-hint">
              {{ current.format.toUpperCase() }} · {{ current.content.length }} characters
            </span>
          </div>
          <div class="template-actions">
            <button type="button" class="btn" (click)="previewOpen.set(!previewOpen())"
              [attr.aria-expanded]="previewOpen()">
              {{ previewOpen() ? 'Hide preview' : 'Preview' }}
            </button>
            <button type="button" class="btn" [disabled]="disabled()" (click)="picker.click()">Replace</button>
            <button type="button" class="btn" [disabled]="disabled()" (click)="remove()">Remove</button>
          </div>
        </div>
        @if (previewOpen()) {
          <pre class="template-preview" aria-label="Report template preview">{{ current.content }}</pre>
        }
      } @else {
        <p class="agent-section-note">Using the default report format.</p>
        <div>
          <button type="button" class="btn" [disabled]="disabled()" (click)="picker.click()">
            Upload template
          </button>
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
  readonly previewOpen = signal(false);
  readonly error = signal('');

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
    this.templateChange.emit({ file_name: file.name, format, content });
  }

  remove(): void {
    this.error.set('');
    this.previewOpen.set(false);
    this.templateChange.emit(null);
  }
}
