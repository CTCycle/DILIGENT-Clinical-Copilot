// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { Component, computed, inject, input } from '@angular/core';

import { MarkdownRendererService } from '../../core/services/markdown-renderer.service';

@Component({
  selector: 'app-clinical-report-viewer',
  templateUrl: './clinical-report-viewer.component.html',
})
export class ClinicalReportViewerComponent {
  readonly markdown = input('');

  private readonly markdownRenderer = inject(MarkdownRendererService);

  readonly rendered = computed(() => this.markdownRenderer.render(this.markdown()));
}
