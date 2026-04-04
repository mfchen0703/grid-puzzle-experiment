export const ROWS = 12;
export const COLS = 20;
export const CELL_SIZE = 40;
export const NUM_REGIONS = 45;
export const COLORS = ['#377eb8', '#4daf4a', '#984ea3', '#ffff33'];

export interface MapData {
  grid: number[][];
  numRegions: number;
  adjacencyPairs: [number, number][];
}

export interface Experiment2Round {
  mapData: MapData;
  initialColors: number[];
  solvedColors: number[];
  conflictEdges: [number, number][];
}

export interface Experiment2Materials {
  rows: number;
  cols: number;
  numRegions: number;
  colors: string[];
  rounds: Experiment2Round[];
}

export function buildAdjacencyMap(mapData: MapData): Record<number, Set<number>> {
  const adjacency: Record<number, Set<number>> = {};
  for (let i = 0; i < mapData.numRegions; i++) {
    adjacency[i] = new Set();
  }
  for (const [a, b] of mapData.adjacencyPairs) {
    adjacency[a].add(b);
    adjacency[b].add(a);
  }
  return adjacency;
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
