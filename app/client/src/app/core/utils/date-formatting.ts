// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { APP_LOCALE } from '../constants';

export function formatAppDateTime(value: string | null | undefined, fallback = 'N/A'): string {
  if (!value?.trim()) return fallback;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return fallback;
  return new Intl.DateTimeFormat(APP_LOCALE, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(parsed);
}
