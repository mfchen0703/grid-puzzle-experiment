export const ROWS = 12;
export const COLS = 20;
export const CELL_SIZE = 40;
export const NUM_REGIONS = 45;
export const COLORS = ['#377eb8', '#4daf4a', '#984ea3', '#ffff33'];
export const EXPERIMENT2_ROUND_SEEDS = [2021, 2037, 2053, 2069, 2081, 2099, 2111, 2137, 2153, 2179];

export interface Region {
  id: number;
  cells: [number, number][];
}

export interface MapData {
  grid: number[][];
  numRegions: number;
  regions: Region[];
  adjacency: Record<number, Set<number>>;
}

export interface Experiment2Round {
  mapData: MapData;
  initialColors: number[];
  solvedColors: number[];
  conflictEdges: [number, number][];
}

export function mulberry32(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateMapData(numRegions: number, random: () => number): MapData {
  const grid = Array(ROWS).fill(null).map(() => Array(COLS).fill(-1));
  const regions: Region[] = [];

  for (let i = 0; i < numRegions; i++) {
    let r;
    let c;
    do {
      r = Math.floor(random() * ROWS);
      c = Math.floor(random() * COLS);
    } while (grid[r][c] !== -1);
    grid[r][c] = i;
    regions.push({ id: i, cells: [[r, c]] });
  }

  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 0; i < numRegions; i++) {
      const neighbors: [number, number][] = [];
      for (const [r, c] of regions[i].cells) {
        for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
          const nr = r + dr;
          const nc = c + dc;
          if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && grid[nr][nc] === -1) {
            neighbors.push([nr, nc]);
          }
        }
      }
      if (neighbors.length > 0) {
        const [nr, nc] = neighbors[Math.floor(random() * neighbors.length)];
        if (grid[nr][nc] === -1) {
          grid[nr][nc] = i;
          regions[i].cells.push([nr, nc]);
          changed = true;
        }
      }
    }
  }

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      if (grid[r][c] !== -1) {
        continue;
      }
      for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
        const nr = r + dr;
        const nc = c + dc;
        if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && grid[nr][nc] !== -1) {
          grid[r][c] = grid[nr][nc];
          regions[grid[nr][nc]].cells.push([r, c]);
          break;
        }
      }
    }
  }

  const adjacency: Record<number, Set<number>> = {};
  for (let i = 0; i < numRegions; i++) {
    adjacency[i] = new Set();
  }

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const region1 = grid[r][c];
      for (const [dr, dc] of [[0, 1], [1, 0]]) {
        const nr = r + dr;
        const nc = c + dc;
        if (nr < ROWS && nc < COLS) {
          const region2 = grid[nr][nc];
          if (region1 !== region2) {
            adjacency[region1].add(region2);
            adjacency[region2].add(region1);
          }
        }
      }
    }
  }

  return { grid, numRegions, regions, adjacency };
}

function shuffle<T>(items: T[], random: () => number): T[] {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function isLegalColor(regionId: number, color: number, adjacency: Record<number, Set<number>>, colors: number[]) {
  for (const neighbor of adjacency[regionId]) {
    if (colors[neighbor] === color) {
      return false;
    }
  }
  return true;
}

function solveColoring(mapData: MapData, random: () => number): number[] {
  const order = Array.from({ length: mapData.numRegions }, (_, idx) => idx).sort(
    (a, b) => mapData.adjacency[b].size - mapData.adjacency[a].size,
  );
  const colors = Array(mapData.numRegions).fill(-1);

  const backtrack = (index: number): boolean => {
    if (index === order.length) {
      return true;
    }
    const regionId = order[index];
    for (const color of shuffle([0, 1, 2, 3], random)) {
      if (!isLegalColor(regionId, color, mapData.adjacency, colors)) {
        continue;
      }
      colors[regionId] = color;
      if (backtrack(index + 1)) {
        return true;
      }
      colors[regionId] = -1;
    }
    return false;
  };

  if (!backtrack(0)) {
    throw new Error('Failed to construct a legal 4-coloring for experiment 2.');
  }
  return colors;
}

export function getConflictEdges(adjacency: Record<number, Set<number>>, colors: number[]): [number, number][] {
  const conflicts: [number, number][] = [];
  for (let region = 0; region < colors.length; region++) {
    for (const neighbor of adjacency[region]) {
      if (region < neighbor && colors[region] === colors[neighbor]) {
        conflicts.push([region, neighbor]);
      }
    }
  }
  return conflicts;
}

function canFixInOneMove(mapData: MapData, colors: number[]): boolean {
  for (let region = 0; region < colors.length; region++) {
    const currentColor = colors[region];
    for (let color = 0; color < COLORS.length; color++) {
      if (color === currentColor) {
        continue;
      }
      const next = [...colors];
      next[region] = color;
      if (getConflictEdges(mapData.adjacency, next).length === 0) {
        return true;
      }
    }
  }
  return false;
}

function buildConflictStartState(mapData: MapData, solvedColors: number[], random: () => number): number[] {
  const regionIds = Array.from({ length: mapData.numRegions }, (_, idx) => idx);

  for (let attempt = 0; attempt < 4000; attempt++) {
    const candidate = [...solvedColors];
    const changeCount = 2 + Math.floor(random() * 4);
    const chosenRegions = shuffle(regionIds, random).slice(0, changeCount);

    for (const region of chosenRegions) {
      const neighborColors = [...mapData.adjacency[region]].map((neighbor) => candidate[neighbor]);
      const preferredConflictingColors = shuffle(
        [...new Set(neighborColors.filter((color) => color >= 0 && color !== candidate[region]))],
        random,
      );
      const fallbackColors = shuffle(
        [0, 1, 2, 3].filter((color) => color !== candidate[region]),
        random,
      );
      const nextColor = preferredConflictingColors[0] ?? fallbackColors[0];
      candidate[region] = nextColor;
    }

    const conflictEdges = getConflictEdges(mapData.adjacency, candidate);
    if (conflictEdges.length < 2) {
      continue;
    }
    if (canFixInOneMove(mapData, candidate)) {
      continue;
    }
    return candidate;
  }

  throw new Error('Failed to build a planning-heavy initial state for experiment 2.');
}

export function getAdjacencyPairs(adjacency: Record<number, Set<number>>): [number, number][] {
  const pairs: [number, number][] = [];
  for (let region = 0; region < NUM_REGIONS; region++) {
    for (const neighbor of adjacency[region]) {
      if (region < neighbor) {
        pairs.push([region, neighbor]);
      }
    }
  }
  return pairs;
}

export function buildExperiment2Rounds(): Experiment2Round[] {
  return EXPERIMENT2_ROUND_SEEDS.map((seed) => {
    const random = mulberry32(seed);
    const mapData = generateMapData(NUM_REGIONS, random);
    const solvedColors = solveColoring(mapData, random);
    const initialColors = buildConflictStartState(mapData, solvedColors, random);
    const conflictEdges = getConflictEdges(mapData.adjacency, initialColors);
    return {
      mapData,
      initialColors,
      solvedColors,
      conflictEdges,
    };
  });
}
