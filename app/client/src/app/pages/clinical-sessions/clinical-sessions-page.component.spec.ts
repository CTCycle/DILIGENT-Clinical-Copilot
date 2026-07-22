import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('ClinicalSessionsPage revision template', () => {
  it('renders revision controls and the editor toolbar', () => {
    const templatePath = resolve(
      __dirname,
      './clinical-sessions-page.component.html',
    );
    const template = readFileSync(templatePath, 'utf-8');
    const toolbarTemplate = readFileSync(
      resolve(__dirname, './components/clinical-session-editor-toolbar.component.ts'),
      'utf-8',
    );

    expect(template).toContain('clinical-session-revision-model-panel');
    expect(template).toContain('Revision model');
    expect(template).toContain('clinical-session-revision-instruction');
    expect(template).toContain('setRevisionModelProvider');
    expect(template).toContain('app-clinical-session-editor-toolbar');
    expect(template).toContain('clinical-session-metadata-status');
    expect(template).toContain('metadataSaveStatus()');
    expect(template).toContain('linkDialogOpen()');
    expect(template).toContain('aria-label="Link URL"');
    expect(toolbarTemplate).toContain('Save manual report edit');
    expect(toolbarTemplate).toContain('role="toolbar"');
    expect(template).not.toContain('Reviewer name');
    expect(template).not.toContain('Manual Edit History');
  });

  it('labels frontend-derived clinical evidence as fallback', () => {
    const templatePath = resolve(
      __dirname,
      './clinical-sessions-page.component.html',
    );
    const template = readFileSync(templatePath, 'utf-8');

    expect(template).toContain('Display fallback');
    expect(template).toContain('Not backend-confirmed');
    expect(template).toContain('drug.bibliographyLabel');
    expect(template).toContain('drug.bibliographyFallback');
  });

  it('guards closed timeline deletion modal content', () => {
    const template = readFileSync(
      resolve(
        __dirname,
        './components/clinical-session-timeline-workspace.component.html',
      ),
      'utf-8',
    );

    expect(template).toContain('@if (timelinePendingDeletion(); as preview)');
    expect(template).not.toContain('timelineProviderLabel(timelinePendingDeletion()!)');
    expect(template).not.toContain('timelineModelLabel(timelinePendingDeletion()!)');
  });

  it('keeps timeline selectors visible while model options load and uses compact history rows', () => {
    const template = readFileSync(
      resolve(
        __dirname,
        './components/clinical-session-timeline-workspace.component.html',
      ),
      'utf-8',
    );

    expect(template).toContain('class="timeline-model-fields"');
    expect(template).toContain('modelConfigLoading() || generationRunning()');
    expect(template).toContain('class="timeline-row"');
    expect(template).toContain('class="timeline-row-summary"');
    expect(template).not.toContain('class="timeline-gallery">@for');
  });
});
