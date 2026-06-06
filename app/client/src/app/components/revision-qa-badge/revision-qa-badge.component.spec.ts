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
