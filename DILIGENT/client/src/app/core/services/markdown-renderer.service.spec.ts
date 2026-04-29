import { MarkdownRendererService } from './markdown-renderer.service';

describe('MarkdownRendererService', () => {
  it('renders headings and inline formatting', () => {
    const service = new MarkdownRendererService();
    const rendered = service.render('# Title\n\n**bold** and *italic*');
    expect(rendered.html).toContain('<h1>Title</h1>');
    expect(rendered.html).toContain('<strong>bold</strong>');
    expect(rendered.html).toContain('<em>italic</em>');
  });

  it('blocks javascript links and escapes html', () => {
    const service = new MarkdownRendererService();
    const rendered = service.render('[x](javascript:alert(1)) <script>x</script>');
    expect(rendered.html).not.toContain('javascript:');
    expect(rendered.html).toContain('&lt;script&gt;');
  });
});
