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
