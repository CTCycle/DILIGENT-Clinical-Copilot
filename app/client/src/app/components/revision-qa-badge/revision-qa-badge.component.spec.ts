import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('RevisionQaBadgeComponent template', () => {
  it('renders the reusable revision QA badge shell', () => {
    const templatePath = resolve(__dirname, './revision-qa-badge.component.html');
    const template = readFileSync(templatePath, 'utf-8');

    expect(template).toContain('revision-qa-badge');
    expect(template).toContain('{{ resolvedLabel }}');
  });
});

describe('RevisionQaBadgeComponent source', () => {
  it('treats approval, warning, failure, and manual-edit states distinctly', () => {
    const sourcePath = resolve(__dirname, './revision-qa-badge.component.ts');
    const source = readFileSync(sourcePath, 'utf-8');

    expect(source).toContain("case 'approved_by_human':");
    expect(source).toContain("case 'requires_human_review':");
    expect(source).toContain("case 'qa_failed':");
    expect(source).toContain("case 'manual_edit':");
    expect(source).toContain("return 'is-good';");
    expect(source).toContain("return 'is-warn';");
    expect(source).toContain("return 'is-bad';");
    expect(source).toContain("return 'is-neutral';");
  });

  it('preserves custom labels and otherwise humanizes underscore-delimited states', () => {
    const sourcePath = resolve(__dirname, './revision-qa-badge.component.ts');
    const source = readFileSync(sourcePath, 'utf-8');

    expect(source).toContain("return this.label.trim() || this.statusLabel(this.status);");
    expect(source).toContain("import { humanizeStatusLabel }");
    expect(source).toContain("return humanizeStatusLabel(value);");
  });
});
