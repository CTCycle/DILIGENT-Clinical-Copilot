import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('ClinicalSessionsPage revision template', () => {
  it('renders separate official version history and pipeline retry affordances', () => {
    const templatePath = resolve(
      __dirname,
      './clinical-sessions-page.component.html',
    );
    const template = readFileSync(templatePath, 'utf-8');

    expect(template).toContain('Official Version History');
    expect(template).toContain('Revision Pipeline Details');
    expect(template).toContain('Revision QA And Artifacts');
    expect(template).toContain('Structured Case Entities');
    expect(template).toContain('Retry draft revision');
    expect(template).toContain('Manual Edit History');
  });
});
