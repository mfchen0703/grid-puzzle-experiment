# Grid Puzz - Exported CSV Documentation

## Overview

After completing all 10 rounds, the exported CSV file (`data_<ID>.csv`) contains two sections: **[Actions]** and **[Adjacency]**.

---

## [Actions] Section

Records every action the participant performed across all 10 rounds.

### Columns

| Column | Description |
|---|---|
| **SessionID** | The participant's ID entered at the start |
| **Round** | Round number (1–10) |
| **NumRegions** | Number of regions in this round's map (20, 23, 26, 28, 31, 34, 37, 39, 42, 45) |
| **Step** | Global step index across all rounds (0-based, sequential) |
| **Action** | Description of the action taken |
| **TimeTaken(s)** | Time in seconds since the previous action |

### Action Types

| Action | Meaning |
|---|---|
| `Game Started (N regions)` | A new round began with N regions. Always the first action of each round. TimeTaken is 0.0 |
| `Colored Region X with Color Y` | The participant colored region X with color Y (1–4) |
| `Colored Region X with Eraser` | The participant erased the color from region X |

### Example

```
[Actions]
SessionID,Round,NumRegions,Step,Action,TimeTaken(s)
42,1,20,0,"Game Started (20 regions)",0.0
42,1,20,1,"Colored Region 3 with Color 1",2.5
42,1,20,2,"Colored Region 7 with Color 2",1.3
42,1,20,3,"Colored Region 7 with Eraser",0.8
42,2,23,4,"Game Started (23 regions)",0.0
...
```

---

## [Adjacency] Section

Lists all pairs of adjacent (neighboring) regions for each round's map. Two regions are adjacent if they share a border edge on the grid.

### Columns

| Column | Description |
|---|---|
| **Round** | Round number (1–10) |
| **NumRegions** | Number of regions in this round's map |
| **Region_A** | First region in the adjacent pair (1-based) |
| **Region_B** | Second region in the adjacent pair (1-based, always > Region_A) |

### Notes

- Each pair appears only once (Region_A < Region_B), so `3,7` means regions 3 and 7 are adjacent — `7,3` will not appear separately.
- The puzzle requires that no two adjacent regions share the same color (graph coloring problem with 4 colors).
- All participants share the same maps (fixed random seeds), so the adjacency data is identical across all exported files.

### Example

```
[Adjacency]
Round,NumRegions,Region_A,Region_B
1,20,1,2
1,20,1,5
1,20,2,3
1,20,2,6
2,23,1,4
...
```

---

## Round Sequence

All participants play the same fixed sequence of 10 rounds with increasing complexity:

| Round | Number of Regions |
|---|---|
| 1 | 20 |
| 2 | 23 |
| 3 | 26 |
| 4 | 28 |
| 5 | 31 |
| 6 | 34 |
| 7 | 37 |
| 8 | 39 |
| 9 | 42 |
| 10 | 45 |

Maps are generated using fixed random seeds, ensuring every participant sees identical maps.
