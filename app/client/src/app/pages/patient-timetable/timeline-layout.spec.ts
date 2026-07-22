import { describe, expect, it } from 'vitest';

import { packTimelineItems } from './timeline-layout';

describe('packTimelineItems', () => {
  it('packs deterministic pixel intervals into non-overlapping rows', () => {
    const result = packTimelineItems([
      { id: 'b', positionPercent: 10, width: 180, sortOrder: 1 },
      { id: 'a', positionPercent: 10, width: 180, sortOrder: 0 },
      { id: 'c', positionPercent: 90, width: 180, sortOrder: 2 },
    ], 1000);

    expect(result.placements.map((item) => [item.id, item.row])).toEqual([
      ['a', 0], ['b', 1], ['c', 0],
    ]);
    expect(result.clusters).toEqual([]);
  });

  it('clusters overflow while preserving stable member order', () => {
    const result = packTimelineItems([
      { id: 'a', positionPercent: 50, width: 300, sortOrder: 0 },
      { id: 'b', positionPercent: 50, width: 300, sortOrder: 1 },
      { id: 'c', positionPercent: 50, width: 300, sortOrder: 2 },
      { id: 'd', positionPercent: 50, width: 300, sortOrder: 3 },
    ], 1000, 3);

    expect(result.clusters).toEqual([{ id: 'cluster-a', memberIds: ['a', 'b', 'c', 'd'], positionPercent: 50 }]);
    expect(result.placements).toEqual([]);
    expect(result.rowCount).toBe(1);
  });
});
