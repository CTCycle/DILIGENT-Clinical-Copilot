import { Injectable } from '@angular/core';

export interface RenderedMarkdown {
  html: string;
  text: string;
}

@Injectable({ providedIn: 'root' })
export class MarkdownRendererService {
  render(markdown: string | null | undefined): RenderedMarkdown {
    const source = markdown ?? '';
    return {
      html: this.renderBlocks(source.split(/\r?\n/)),
      text: this.stripMarkdownToText(source),
    };
  }

  private renderBlocks(lines: string[]): string {
    const out: string[] = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) {
        i += 1;
        continue;
      }
      if (/^```/.test(line.trim())) {
        const code: string[] = [];
        i += 1;
        while (i < lines.length && !/^```/.test(lines[i].trim())) {
          code.push(lines[i]);
          i += 1;
        }
        i += 1;
        out.push(`<pre><code>${this.escapeHtml(code.join('\n'))}</code></pre>`);
        continue;
      }
      const heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        out.push(`<h${heading[1].length}>${this.renderInline(heading[2])}</h${heading[1].length}>`);
        i += 1;
        continue;
      }
      if (/^\s*[-*]\s+/.test(line)) {
        const items: string[] = [];
        while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
          items.push(lines[i].replace(/^\s*[-*]\s+/, ''));
          i += 1;
        }
        out.push(this.renderUnorderedList(items));
        continue;
      }
      if (/^\s*\d+\.\s+/.test(line)) {
        const items: string[] = [];
        while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
          items.push(lines[i].replace(/^\s*\d+\.\s+/, ''));
          i += 1;
        }
        out.push(this.renderOrderedList(items));
        continue;
      }
      if (/^>\s?/.test(line)) {
        out.push(`<blockquote>${this.renderInline(line.replace(/^>\s?/, ''))}</blockquote>`);
        i += 1;
        continue;
      }
      if (/^\s*([-*_])\1\1+\s*$/.test(line)) {
        out.push('<hr />');
        i += 1;
        continue;
      }
      const para: string[] = [];
      while (i < lines.length && lines[i].trim()) {
        para.push(lines[i]);
        i += 1;
      }
      out.push(this.renderParagraph(para));
    }
    return out.join('\n');
  }

  private renderInline(value: string): string {
    let html = this.escapeHtml(value);
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_m, text: string, href: string) => {
      const safe = this.sanitizeHref(href);
      return safe ? `<a href="${this.escapeHtml(safe)}" target="_blank" rel="noopener noreferrer">${this.escapeHtml(text)}</a>` : this.escapeHtml(text);
    });
    return html;
  }

  private escapeHtml(value: string): string {
    return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  private sanitizeHref(value: string): string | null {
    try {
      const parsed = new URL(value, 'https://example.com');
      if (['http:', 'https:', 'mailto:'].includes(parsed.protocol)) {
        return value;
      }
      return null;
    } catch {
      return null;
    }
  }

  private renderParagraph(lines: string[]): string {
    return `<p>${this.renderInline(lines.join(' '))}</p>`;
  }

  private renderUnorderedList(items: string[]): string {
    return `<ul>${items.map((item) => `<li>${this.renderInline(item)}</li>`).join('')}</ul>`;
  }

  private renderOrderedList(items: string[]): string {
    return `<ol>${items.map((item) => `<li>${this.renderInline(item)}</li>`).join('')}</ol>`;
  }

  private stripMarkdownToText(markdown: string): string {
    return markdown
      .replace(/```[\s\S]*?```/g, '')
      .replace(/[#>*_`\-]/g, '')
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
      .trim();
  }
}
