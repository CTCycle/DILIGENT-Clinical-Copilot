import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('DataInspectionPage template', () => {
  it('renders the Vector model column in the RAG preview table', () => {
    const templatePath = resolve(
      __dirname,
      './data-inspection-page.component.html',
    );
    const template = readFileSync(templatePath, 'utf-8');
    expect(template).toContain('Vector model');
    expect(template).toContain("row.vector_model || 'Not vectorized'");
  });
});
