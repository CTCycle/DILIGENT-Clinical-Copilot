// Copyright © 2023–2025 Thomas Virdis
// Licensed under the GNU General Public License, version 3 or later.

import { describe, expect, it } from 'vitest';

import { serializeApplicationConfigurationWrite } from './application-configuration-write-queue';

describe('application configuration write queue', () => {
  it('runs singleton updates in invocation order', async () => {
    const events: string[] = [];
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>((resolve) => { releaseFirst = resolve; });

    const first = serializeApplicationConfigurationWrite(async () => {
      events.push('first-start');
      await firstGate;
      events.push('first-end');
      return 1;
    });
    const second = serializeApplicationConfigurationWrite(async () => {
      events.push('second');
      return 2;
    });

    await Promise.resolve();
    expect(events).toEqual(['first-start']);
    releaseFirst();

    await expect(Promise.all([first, second])).resolves.toEqual([1, 2]);
    expect(events).toEqual(['first-start', 'first-end', 'second']);
  });

  it('continues processing after a failed update', async () => {
    const failed = serializeApplicationConfigurationWrite(async () => {
      throw new Error('write failed');
    });
    const following = serializeApplicationConfigurationWrite(async () => 'saved');

    await expect(failed).rejects.toThrow('write failed');
    await expect(following).resolves.toBe('saved');
  });
});
