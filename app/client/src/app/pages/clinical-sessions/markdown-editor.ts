import type { EditorCommandName } from './clinical-sessions.types';

export type MarkdownEdit = { text: string; selectionStart: number; selectionEnd: number };

const blockPrefixes = new Map<string, string>([['h1', '# '], ['h2', '## '], ['h3', '### '], ['blockquote', '> '], ['p', '']]);

export function applyMarkdownCommand(source: string, selectionStart: number, selectionEnd: number, command: EditorCommandName, value?: string): MarkdownEdit {
  const start = Math.min(selectionStart, selectionEnd);
  const end = Math.max(selectionStart, selectionEnd);
  const selected = source.slice(start, end);
  if (command === 'formatBlock') return formatBlock(source, start, end, value || 'p');
  if (command === 'bold' || command === 'italic' || command === 'strikeThrough') {
    const marker = command === 'bold' ? '**' : command === 'italic' ? '*' : '~~';
    const isWrapped = selected.startsWith(marker) && selected.endsWith(marker);
    const inner = isWrapped ? selected.slice(marker.length, -marker.length) : selected || 'text';
    const replacement = isWrapped ? inner : `${marker}${inner}${marker}`;
    return replaceSelection(source, start, end, replacement, replacement.length);
  }
  if (command === 'insertUnorderedList' || command === 'insertOrderedList') return prefixLines(source, start, end, command === 'insertOrderedList' ? '1. ' : '- ');
  if (command === 'createLink') {
    const url = (value || '').trim();
    if (!url) return { text: source, selectionStart: start, selectionEnd: end };
    const replacement = `[${selected || 'link'}](${url})`;
    return replaceSelection(source, start, end, replacement, selected ? replacement.length : 4);
  }
  return { text: source, selectionStart: start, selectionEnd: end };
}

function replaceSelection(source: string, start: number, end: number, replacement: string, caret: number): MarkdownEdit {
  const text = `${source.slice(0, start)}${replacement}${source.slice(end)}`;
  const next = start + caret;
  return { text, selectionStart: next, selectionEnd: next };
}

function formatBlock(source: string, start: number, end: number, format: string): MarkdownEdit {
  const lineStart = source.lastIndexOf('\n', Math.max(0, start - 1)) + 1;
  const foundEnd = source.indexOf('\n', end);
  const lineEnd = foundEnd < 0 ? source.length : foundEnd;
  const block = source.slice(lineStart, lineEnd);
  const contentLines = block.split('\n').map((line) => line.replace(/^(#{1,6}\s|>\s|-\s|\d+\.\s)/, ''));
  const content = contentLines.join('\n');
  const replacement = format === 'pre'
    ? `\`\`\`\n${content}\n\`\`\``
    : format === 'blockquote'
      ? contentLines.map((line) => `> ${line}`).join('\n')
      : `${blockPrefixes.get(format) ?? ''}${content}`;
  const text = `${source.slice(0, lineStart)}${replacement}${source.slice(lineEnd)}`;
  const offset = replacement.length - block.length;
  return { text, selectionStart: start + offset, selectionEnd: end + offset };
}

function prefixLines(source: string, start: number, end: number, prefix: string): MarkdownEdit {
  const lineStart = source.lastIndexOf('\n', Math.max(0, start - 1)) + 1;
  const foundEnd = source.indexOf('\n', end);
  const lineEnd = foundEnd < 0 ? source.length : foundEnd;
  const block = source.slice(lineStart, lineEnd);
  const replacement = block.split('\n').map((line, index) => {
    const content = line.replace(/^(?:-\s|\d+\.\s)/, '');
    return content ? `${prefix === '1. ' ? `${index + 1}. ` : prefix}${content}` : prefix.trimEnd();
  }).join('\n');
  const text = `${source.slice(0, lineStart)}${replacement}${source.slice(lineEnd)}`;
  const offset = replacement.length - block.length;
  return { text, selectionStart: start + offset, selectionEnd: end + offset };
}
