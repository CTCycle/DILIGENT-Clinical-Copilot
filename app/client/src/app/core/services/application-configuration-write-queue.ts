// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

let pendingWrite: Promise<void> = Promise.resolve();

export function serializeApplicationConfigurationWrite<T>(
  write: () => Promise<T>,
): Promise<T> {
  const result = pendingWrite.then(write, write);
  pendingWrite = result.then(() => undefined, () => undefined);
  return result;
}
