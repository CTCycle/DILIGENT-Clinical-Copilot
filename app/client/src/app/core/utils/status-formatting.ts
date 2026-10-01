// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

export function humanizeStatusLabel(value: string | null | undefined): string {
  if (!value) {
    return 'Unknown';
  }
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
