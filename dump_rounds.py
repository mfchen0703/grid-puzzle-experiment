"""
输出每个 round 的区域坐标和邻接关系。
"""
import ctypes
from collections import defaultdict

ROWS, COLS = 12, 20
ROUND_SIZES = [20, 23, 26, 28, 31, 34, 37, 39, 42, 45]
ROUND_SEEDS = [42, 137, 256, 389, 512, 647, 783, 891, 1024, 1157]

def _i32(x):
    return ctypes.c_int32(x & 0xFFFFFFFF).value

def _u32(x):
    return x & 0xFFFFFFFF

def _imul(a, b):
    return _i32(a * b)

def mulberry32(seed):
    s = _i32(seed)
    def rand():
        nonlocal s
        s = _i32(s + 0x6D2B79F5)
        t = _imul(s ^ _u32(_u32(s) >> 15), _i32(1 | s))
        t = _i32(t + _imul(t ^ _u32(_u32(t) >> 7), _i32(61 | t))) ^ t
        return _u32(t ^ _u32(_u32(t) >> 14)) / 4294967296
    return rand

def generate_map(num_regions, rng):
    grid = [[-1] * COLS for _ in range(ROWS)]
    regions = []

    for i in range(num_regions):
        while True:
            r = int(rng() * ROWS)
            c = int(rng() * COLS)
            if grid[r][c] == -1:
                break
        grid[r][c] = i
        regions.append([(r, c)])

    changed = True
    while changed:
        changed = False
        for i in range(num_regions):
            neighbors = []
            for (r, c) in regions[i]:
                for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] == -1:
                        neighbors.append((nr, nc))
            if neighbors:
                nr, nc = neighbors[int(rng() * len(neighbors))]
                if grid[nr][nc] == -1:
                    grid[nr][nc] = i
                    regions[i].append((nr, nc))
                    changed = True

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == -1:
                for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] != -1:
                        grid[r][c] = grid[nr][nc]
                        regions[grid[nr][nc]].append((r, c))
                        break

    adjacency = defaultdict(set)
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in [(0,1),(1,0)]:
                nr, nc = r+dr, c+dc
                if nr < ROWS and nc < COLS:
                    if grid[r][c] != grid[nr][nc]:
                        adjacency[grid[r][c]].add(grid[nr][nc])
                        adjacency[grid[nr][nc]].add(grid[r][c])

    return grid, regions, adjacency


if __name__ == '__main__':
    for round_idx in range(len(ROUND_SIZES)):
        num_regions = ROUND_SIZES[round_idx]
        rng = mulberry32(ROUND_SEEDS[round_idx])
        grid, regions, adjacency = generate_map(num_regions, rng)

        print(f'===== Round {round_idx} ({num_regions} regions) =====')
        print()

        print('--- Regions (id: cells) ---')
        for i, cells in enumerate(regions):
            coords = ', '.join(f'({r},{c})' for r, c in sorted(cells))
            print(f'  Region {i:2d}: [{coords}]')
        print()

        print('--- Adjacency ---')
        for i in range(num_regions):
            nbrs = sorted(adjacency[i])
            print(f'  Region {i:2d}: {nbrs}')
        print()
        print()
