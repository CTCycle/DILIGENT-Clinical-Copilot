import { describe, expect, it } from 'vitest';

import { applyMarkdownCommand } from './markdown-editor';

describe('clinical session Markdown editor', () => {
  it('wraps a selection without changing surrounding spaces', () => {
    const result = applyMarkdownCommand('before  text  after', 8, 12, 'bold');
    expect(result.text).toBe('before  **text**  after');
  });

  it('keeps blank lines when formatting a block', () => {
    const result = applyMarkdownCommand('one\n\ntwo', 0, 7, 'formatBlock', 'blockquote');
    expect(result.text).toBe('> one\n> \n> two');
  });

  it('adds ordered list markers to every selected line', () => {
    const result = applyMarkdownCommand('a\nb', 0, 3, 'insertOrderedList');
    expect(result.text).toBe('1. a\n2. b');
  });

  it('creates a Markdown link from the selected text', () => {
    const result = applyMarkdownCommand('Read this', 5, 9, 'createLink', 'https://example.test');
    expect(result.text).toBe('Read [this](https://example.test)');
  });

  it('preserves source text exactly when no command is applied', () => {
    const source = 'line 1\n\nline 2  ';
    expect(applyMarkdownCommand(source, 2, 2, 'undo').text).toBe(source);
  });
});
