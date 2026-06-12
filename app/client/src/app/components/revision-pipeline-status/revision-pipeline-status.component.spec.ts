import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

describe('RevisionPipelineStatusComponent template', () => {
  it('renders persisted run status and checkpoint-step affordances', () => {
    const templatePath = resolve(__dirname, './revision-pipeline-status.component.html');
    const template = readFileSync(templatePath, 'utf-8');

    expect(template).toContain('Run {{ run.pipeline_run_id }}');
    expect(template).toContain('Checkpoint Steps');
    expect(template).toContain('Retry draft revision');
    expect(template).toContain('Persisted run configuration');
  });
});

describe('RevisionPipelineStatusComponent source', () => {
  it('counts completed steps and summarizes persisted structured output before input', () => {
    const sourcePath = resolve(__dirname, './revision-pipeline-status.component.ts');
    const source = readFileSync(sourcePath, 'utf-8');

    expect(source).toContain(
      "return this.steps.filter((step) => step.status === 'completed').length;",
    );
    expect(source).toContain('const outputSummary = step.output_summary;');
    expect(source).toContain('const inputSummary = step.input_summary;');
    expect(source).toContain(".slice(0, 3)");
    expect(source).toContain("return 'No structured summary saved.';");
  });
});
