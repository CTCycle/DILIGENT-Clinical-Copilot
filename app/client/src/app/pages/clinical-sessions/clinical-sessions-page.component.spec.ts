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
    expect(template).toContain('setRevisionModelRuntime');
    expect(template).toContain('app-clinical-session-editor-toolbar');
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
});
