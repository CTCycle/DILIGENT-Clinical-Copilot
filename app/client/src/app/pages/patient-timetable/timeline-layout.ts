export type TimelineLayoutItem = {
  id: string;
  positionPercent: number;
  width: number;
  sortOrder: number;
};

export type TimelineLayoutPlacement = TimelineLayoutItem & { row: number };

export type TimelineCluster = {
  id: string;
  memberIds: string[];
  positionPercent: number;
};

export type TimelineLayoutResult = {
  placements: TimelineLayoutPlacement[];
  clusters: TimelineCluster[];
  rowCount: number;
};

export function packTimelineItems(
  items: TimelineLayoutItem[],
  canvasWidth: number,
  rowCap = 3,
): TimelineLayoutResult {
  const sorted = [...items].sort((a, b) => a.positionPercent - b.positionPercent || a.sortOrder - b.sortOrder || a.id.localeCompare(b.id));
  const rows: number[] = [];
  const placements: TimelineLayoutPlacement[] = [];
  const overflow: TimelineLayoutItem[] = [];
  const clusters: TimelineCluster[] = [];
  const clusteredIds = new Set<string>();
  const exactGroups = new Map<number, TimelineLayoutItem[]>();
  for (const item of sorted) {
    const group = exactGroups.get(item.positionPercent) ?? [];
    group.push(item);
    exactGroups.set(item.positionPercent, group);
  }
  for (const group of exactGroups.values()) {
    if (group.length <= rowCap) continue;
    clusters.push({ id: `cluster-${group[0].id}`, memberIds: group.map((item) => item.id), positionPercent: group[0].positionPercent });
    for (const item of group) clusteredIds.add(item.id);
  }
  for (const item of sorted) {
    if (clusteredIds.has(item.id)) continue;
    const center = (item.positionPercent / 100) * canvasWidth;
    const left = center - item.width / 2;
    const right = center + item.width / 2;
    const row = rows.findIndex((latestRight) => left >= latestRight + 12);
    if (row < 0 && rows.length >= rowCap) {
      overflow.push(item);
      continue;
    }
    const selectedRow = row < 0 ? rows.length : row;
    rows[selectedRow] = right;
    placements.push({ ...item, row: selectedRow });
  }
  for (const item of overflow) {
    const existing = clusters.find((cluster) => Math.abs(cluster.positionPercent - item.positionPercent) <= 3);
    if (existing) existing.memberIds.push(item.id);
    else clusters.push({ id: `cluster-${item.id}`, memberIds: [item.id], positionPercent: item.positionPercent });
  }
  return { placements, clusters, rowCount: Math.max(1, rows.length) };
}
