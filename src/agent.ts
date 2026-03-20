/**
 * Agent for the grid coloring puzzle.
 *
 * Input:
 *   - regions: array of { id, cells: [row, col][] }
 *   - adjacency: Record<number, Set<number>>  (region id -> neighbor ids)
 *   - currentColors: (string | null)[]  (current color assignment per region)
 *   - colors: string[]  (available palette)
 *
 * Strategy:
 *   1. Pick the uncolored region closest to the grid center.
 *   2. Among ties, prefer the region whose neighbors have the most determined
 *      (already-colored) entries — i.e. lowest remaining uncertainty.
 *   3. Assign the first legal color (no conflict with neighbors).
 */

// Grid dimensions — keep in sync with App.tsx
const ROWS = 12;
const COLS = 20;

export interface Region {
  id: number;
  cells: [number, number][];
}

export interface AgentMove {
  regionId: number;
  color: string;
}

/** Euclidean distance from a region's centroid to the grid center. */
function centroidDistance(region: Region): number {
  const cx = COLS / 2;
  const cy = ROWS / 2;
  let sumR = 0, sumC = 0;
  for (const [r, c] of region.cells) {
    sumR += r;
    sumC += c;
  }
  const avgR = sumR / region.cells.length;
  const avgC = sumC / region.cells.length;
  return Math.hypot(avgR - cy, avgC - cx);
}

/**
 * Count how many of a region's neighbors are already colored.
 */
function coloredNeighborCount(
  regionId: number,
  adjacency: Record<number, Set<number>>,
  currentColors: (string | null)[],
): number {
  let count = 0;
  for (const nid of adjacency[regionId]) {
    if (currentColors[nid] !== null) count++;
  }
  return count;
}

/**
 * Pick the next region to color and the color to use.
 * Returns null if all regions are colored.
 */
export function agentPickMove(
  regions: Region[],
  adjacency: Record<number, Set<number>>,
  currentColors: (string | null)[],
  colors: string[],
): AgentMove | null {
  // Gather uncolored regions
  const uncolored = regions.filter((r) => currentColors[r.id] === null);
  if (uncolored.length === 0) return null;

  // Score each uncolored region: lower distance is better, higher colored-neighbor count is better.
  // We sort by: (1) more colored neighbors first, (2) closer to center first.
  uncolored.sort((a, b) => {
    const coloredA = coloredNeighborCount(a.id, adjacency, currentColors);
    const coloredB = coloredNeighborCount(b.id, adjacency, currentColors);
    if (coloredB !== coloredA) return coloredB - coloredA; // more colored neighbors first
    return centroidDistance(a) - centroidDistance(b); // closer to center first
  });

  const chosen = uncolored[0];

  // Collect neighbor colors
  const usedColors = new Set<string>();
  for (const nid of adjacency[chosen.id]) {
    const c = currentColors[nid];
    if (c !== null) usedColors.add(c);
  }

  // Pick the first legal color
  const color = colors.find((c) => !usedColors.has(c));
  if (!color) {
    // All colors conflict — shouldn't happen with 4 colors on a planar graph,
    // but fall back to first color if it does.
    return { regionId: chosen.id, color: colors[0] };
  }

  return { regionId: chosen.id, color };
}

export interface AgentStep {
  step: number;
  regionId: number;
  color: string;
  reason: string;
}

/**
 * Run the agent to completion, returning every step with an explanation.
 */
export function agentSolve(
  regions: Region[],
  adjacency: Record<number, Set<number>>,
  colors: string[],
): AgentStep[] {
  const currentColors: (string | null)[] = Array(regions.length).fill(null);
  const steps: AgentStep[] = [];

  let step = 1;
  while (true) {
    const move = agentPickMove(regions, adjacency, currentColors, colors);
    if (!move) break;

    const dist = centroidDistance(regions[move.regionId]).toFixed(2);
    const colored = coloredNeighborCount(move.regionId, adjacency, currentColors);
    const totalNeighbors = adjacency[move.regionId].size;

    steps.push({
      step,
      regionId: move.regionId,
      color: move.color,
      reason: `colored neighbors: ${colored}/${totalNeighbors}, distance to center: ${dist}`,
    });

    currentColors[move.regionId] = move.color;
    step++;
  }

  return steps;
}
