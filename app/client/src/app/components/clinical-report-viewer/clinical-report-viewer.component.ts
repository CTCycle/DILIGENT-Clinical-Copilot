// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, HostListener, computed, inject, input } from '@angular/core';

import { REPORT_EXPORT_FILENAME } from '../../core/constants';
import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';
import { AppStateService } from '../../core/state/app-state.service';
import { createDownloadUrl } from '../../core/utils';

@Component({
  selector: 'app-clinical-report-viewer',
  templateUrl: './clinical-report-viewer.component.html',
})
export class ClinicalReportViewerComponent {
  readonly markdown = input('');

  private readonly markdownRenderer = inject(MarkdownRendererService);
  private readonly stateService = inject(AppStateService);

  readonly rendered = computed(() => this.markdownRenderer.render(this.markdown()));
  readonly hasReport = computed(() => this.markdown().trim().length > 0);
  readonly isExpanded = computed(() => this.stateService.state().diliAgent.isExpanded);

  async copyReport(): Promise<void> {
    const rendered = this.rendered();
    if (!rendered.text) {
      return;
    }
    const clipboardItemCtor = (globalThis as { ClipboardItem?: typeof ClipboardItem }).ClipboardItem;
    if (navigator.clipboard && clipboardItemCtor) {
      const item = new clipboardItemCtor({
        'text/html': new Blob([rendered.html], { type: 'text/html' }),
        'text/plain': new Blob([rendered.text], { type: 'text/plain' }),
      });
      await navigator.clipboard.write([item]);
      return;
    }
    if (navigator.clipboard) {
      await navigator.clipboard.writeText(rendered.text);
    }
  }

  toggleExpanded(): void {
    this.stateService.updateDiliAgent({ isExpanded: !this.isExpanded() });
  }

  downloadReport(): void {
    if (!this.hasReport()) {
      return;
    }
    const url = createDownloadUrl(this.markdown(), REPORT_EXPORT_FILENAME);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = REPORT_EXPORT_FILENAME;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.isExpanded()) {
      this.stateService.updateDiliAgent({ isExpanded: false });
    }
  }
}
