export function humanizeStatusLabel(value: string | null | undefined): string {
  if (!value) {
    return 'Unknown';
  }
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
}
