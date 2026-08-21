import { formatAppDateOnly, formatAppDateTime } from './date-formatting';

describe('application date formatting', () => {
  it('formats date-only values in English with UTC-safe calendar handling', () => {
    expect(formatAppDateOnly('2026-08-12')).toBe('12 August 2026');
  });

  it('uses the supplied fallback for missing or invalid dates', () => {
    expect(formatAppDateOnly(null, 'Not set')).toBe('Not set');
    expect(formatAppDateOnly('2026-02-30', 'Not set')).toBe('Not set');
    expect(formatAppDateTime('not-a-date', 'Unavailable')).toBe('Unavailable');
  });

  it('formats timestamps with the shared English locale', () => {
    expect(formatAppDateTime('2026-08-12T13:45:00Z')).toContain('12 Aug 2026');
  });
});
