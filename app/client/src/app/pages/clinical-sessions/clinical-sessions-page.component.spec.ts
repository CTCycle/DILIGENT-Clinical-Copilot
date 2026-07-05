import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('ClinicalSessionsPage revision template', () => {
  it('renders the current revision placeholder and retained manual edit affordances', () => {
    const templatePath = resolve(
      __dirname,
      './clinical-sessions-page.component.html',
    );
    const template = readFileSync(templatePath, 'utf-8');
    const toolbarTemplate = readFileSync(
      resolve(__dirname, './components/clinical-session-editor-toolbar.component.ts'),
      'utf-8',
    );

    expect(template).toContain('Session revision rewrite pending');
    expect(template).toContain('The previous LLM-assisted session revision workflow has been removed.');
    expect(template).toContain('This area is intentionally reserved for the replacement implementation.');
    expect(template).toContain('app-clinical-session-editor-toolbar');
    expect(toolbarTemplate).toContain('Save manual report edit');
    expect(template).toContain('Manual Edit History');
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
