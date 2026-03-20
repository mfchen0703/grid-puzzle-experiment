import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Grid, Check, Eraser, Download, PlaySquare, ListOrdered } from 'lucide-react';

const ROWS = 12;
const COLS = 20;
const CELL_SIZE = 40;

const COLORS = [
  '#377eb8', // Blue (Set1)
  '#4daf4a', // Green (Set1)
  '#984ea3', // Purple (Set1)
  '#ffff33', // Yellow (Set1)
];

// Fixed sequence: 10 rounds, region count from 20 to 45 (arithmetic)
const ROUND_SIZES = [20, 23, 26, 28, 31, 34, 37, 39, 42, 45];

// Seeded PRNG (mulberry32) - ensures all participants get identical maps
function mulberry32(seed: number) {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Fixed seed per round so every participant gets the same maps
const ROUND_SEEDS = [42, 137, 256, 389, 512, 647, 783, 891, 1024, 1157];

// Practice rounds: 2 rounds of 10 regions each
const PRACTICE_SIZES = [10, 10];
const PRACTICE_SEEDS = [9999, 8888];

interface Region {
  id: number;
  cells: [number, number][];
}

interface MapData {
  grid: number[][];
  numRegions: number;
  regions: Region[];
  adjacency: Record<number, Set<number>>;
}

type HistoryEntry = {
  regionColors: (string | null)[];
  moveDescription: string;
  timeTakenMs?: number;
};

type GlobalHistoryEntry = HistoryEntry & { round: string; numRegions: number };

type RoundAdjacency = { round: string; numRegions: number; adjacency: [number, number][] };

const generateMapData = (numRegions: number, random: () => number): MapData => {
  const grid = Array(ROWS).fill(null).map(() => Array(COLS).fill(-1));
  const regions: Region[] = [];

  for (let i = 0; i < numRegions; i++) {
    let r, c;
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
      const regionCells = regions[i].cells;
      const neighbors: [number, number][] = [];
      for (const [r, c] of regionCells) {
        const dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
        for (const [dr, dc] of dirs) {
          const nr = r + dr, nc = c + dc;
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
      if (grid[r][c] === -1) {
        const dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]];
        for (const [dr, dc] of dirs) {
          const nr = r + dr, nc = c + dc;
          if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && grid[nr][nc] !== -1) {
            grid[r][c] = grid[nr][nc];
            regions[grid[nr][nc]].cells.push([r, c]);
            break;
          }
        }
      }
    }
  }

  const adjacency: Record<number, Set<number>> = {};
  for (let i = 0; i < numRegions; i++) adjacency[i] = new Set();

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const region1 = grid[r][c];
      const dirs = [[0, 1], [1, 0]];
      for (const [dr, dc] of dirs) {
        const nr = r + dr, nc = c + dc;
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
};

// Extract adjacency pairs (sorted, deduplicated)
function getAdjacencyPairs(adjacency: Record<number, Set<number>>): [number, number][] {
  const pairs: [number, number][] = [];
  for (const [region, neighbors] of Object.entries(adjacency)) {
    const r = parseInt(region);
    for (const n of neighbors) {
      if (r < n) pairs.push([r, n]);
    }
  }
  return pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

export default function App() {
  const [mapData, setMapData] = useState<MapData | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [selectedColor, setSelectedColor] = useState<string | null>(COLORS[0]);
  const [hoveredRegion, setHoveredRegion] = useState<number | null>(null);
  const lastActionTime = useRef<number>(Date.now());

  const [sequenceIndex, setSequenceIndex] = useState<number>(0);
  const [practiceIndex, setPracticeIndex] = useState<number>(0);
  const [globalHistory, setGlobalHistory] = useState<GlobalHistoryEntry[]>([]);
  const [allAdjacencies, setAllAdjacencies] = useState<RoundAdjacency[]>([]);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [gamePhase, setGamePhase] = useState<'input' | 'instruction' | 'practice' | 'playing'>('input');
  const [inputId, setInputId] = useState('');

  const initGame = useCallback((roundIndex: number, mode: 'practice' | 'formal') => {
    const sizes = mode === 'practice' ? PRACTICE_SIZES : ROUND_SIZES;
    const seeds = mode === 'practice' ? PRACTICE_SEEDS : ROUND_SEEDS;
    const numRegions = sizes[roundIndex];
    const random = mulberry32(seeds[roundIndex]);
    const newMap = generateMapData(numRegions, random);
    const initialColors = Array(numRegions).fill(null);

    setMapData(newMap);
    lastActionTime.current = Date.now();
    setHistory([{ regionColors: initialColors, moveDescription: `Game Started (${numRegions} regions)`, timeTakenMs: 0 }]);
    setHistoryIndex(0);

    // Store adjacency for this round
    const roundLabel = mode === 'practice' ? `P${roundIndex + 1}` : `${roundIndex + 1}`;
    const pairs = getAdjacencyPairs(newMap.adjacency);
    setAllAdjacencies(prev => [...prev, { round: roundLabel, numRegions, adjacency: pairs }]);
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get('id');
    if (id) {
      setSessionId(id);
      setGamePhase('instruction');
    }
  }, []);

  const currentColors = historyIndex >= 0 ? history[historyIndex].regionColors : [];

  const errors = useMemo(() => {
    const errs = new Set<number>();
    if (!mapData || currentColors.length === 0) return errs;

    for (let i = 0; i < mapData.numRegions; i++) {
      if (currentColors[i] === null) continue;
      for (const neighbor of mapData.adjacency[i]) {
        if (currentColors[i] === currentColors[neighbor]) {
          errs.add(i);
          errs.add(neighbor);
        }
      }
    }
    return errs;
  }, [mapData, currentColors]);

  const isSolved = useMemo(() => {
    if (!mapData || currentColors.length === 0) return false;
    if (currentColors.includes(null)) return false;
    return errors.size === 0;
  }, [mapData, currentColors, errors]);

  const isSequenceComplete = gamePhase === 'playing' && sequenceIndex === ROUND_SIZES.length - 1 && isSolved;
  const isPracticeRoundLast = gamePhase === 'practice' && practiceIndex === PRACTICE_SIZES.length - 1;

  const handleRegionClick = (regionId: number) => {
    if (selectedColor === undefined || !mapData || isSolved) return;

    if (currentColors[regionId] === selectedColor) return;

    const newColors = [...currentColors];
    newColors[regionId] = selectedColor;

    const colorName = selectedColor === null ? 'Eraser' : `Color ${COLORS.indexOf(selectedColor) + 1}`;
    const moveDesc = `Colored Region ${regionId + 1} with ${colorName}`;

    const now = Date.now();
    const timeTakenMs = now - lastActionTime.current;
    lastActionTime.current = now;

    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({ regionColors: newColors, moveDescription: moveDesc, timeTakenMs });

    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const handleIdSubmit = () => {
    const id = inputId.trim();
    if (!id) return;
    setSessionId(id);
    setGlobalHistory([]);
    setAllAdjacencies([]);
    setGamePhase('instruction');
  };

  const handleStartPractice = () => {
    setPracticeIndex(0);
    setGamePhase('practice');
    initGame(0, 'practice');
  };

  const handleNextRound = () => {
    if (sequenceIndex >= ROUND_SIZES.length - 1) return;

    const currentRoundHistory = history.slice(0, historyIndex + 1).map(h => ({
      ...h,
      round: `${sequenceIndex + 1}`,
      numRegions: ROUND_SIZES[sequenceIndex],
    }));

    setGlobalHistory(prev => [...prev, ...currentRoundHistory]);

    const nextIndex = sequenceIndex + 1;
    setSequenceIndex(nextIndex);
    initGame(nextIndex, 'formal');
  };

  const handleNextPracticeRound = () => {
    const currentRoundHistory = history.slice(0, historyIndex + 1).map(h => ({
      ...h,
      round: `P${practiceIndex + 1}`,
      numRegions: PRACTICE_SIZES[practiceIndex],
    }));
    setGlobalHistory(prev => [...prev, ...currentRoundHistory]);

    const nextIndex = practiceIndex + 1;
    setPracticeIndex(nextIndex);
    initGame(nextIndex, 'practice');
  };

  const handleStartFormal = () => {
    const currentRoundHistory = history.slice(0, historyIndex + 1).map(h => ({
      ...h,
      round: `P${practiceIndex + 1}`,
      numRegions: PRACTICE_SIZES[practiceIndex],
    }));
    setGlobalHistory(prev => [...prev, ...currentRoundHistory]);

    setGamePhase('playing');
    setSequenceIndex(0);
    initGame(0, 'formal');
  };

  const handleExportCSV = () => {
    const currentRoundHistory = history.slice(0, historyIndex + 1);
    const idPrefix = sessionId ? `${sessionId},` : "";
    const headerIdPrefix = sessionId ? "SessionID," : "";

    // --- Actions sheet ---
    let csvContent = `[Actions]\n`;
    csvContent += `${headerIdPrefix}Round,NumRegions,Step,Action,TimeTaken(s)\n`;
    const currentMapped = currentRoundHistory.map(h => ({
      ...h,
      round: `${sequenceIndex + 1}`,
      numRegions: ROUND_SIZES[sequenceIndex],
    }));
    const allHistory = [...globalHistory, ...currentMapped];

    allHistory.forEach((entry, idx) => {
      const time = entry.timeTakenMs ? (entry.timeTakenMs / 1000).toFixed(1) : "0.0";
      const desc = `"${entry.moveDescription.replace(/"/g, '""')}"`;
      csvContent += `${idPrefix}${entry.round},${entry.numRegions},${idx},${desc},${time}\n`;
    });

    // --- Adjacency sheet ---
    csvContent += `\n[Adjacency]\n`;
    csvContent += `Round,NumRegions,Region_A,Region_B\n`;
    for (const ra of allAdjacencies) {
      for (const [a, b] of ra.adjacency) {
        csvContent += `${ra.round},${ra.numRegions},${a + 1},${b + 1}\n`;
      }
    }

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', sessionId ? `data_${sessionId}.csv` : `grid-puzz-data-${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (gamePhase === 'input') {
    return (
      <div className="min-h-screen bg-[#2a2a2a] font-sans flex flex-col items-center justify-center selection:bg-blue-500/30">
        <div className="flex items-center gap-3 text-3xl font-bold text-white mb-12">
          <Grid size={36} />
          Grid Puzz
        </div>
        <div className="bg-[#3a3a3a] p-10 rounded-2xl shadow-2xl border border-white/10 w-full max-w-md">
          <h2 className="text-2xl font-bold text-white mb-2 text-center">输入你的编号</h2>
          <p className="text-gray-400 text-sm mb-8 text-center">输入你的编号以开始实验。</p>
          <input
            type="text"
            value={inputId}
            onChange={e => setInputId(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleIdSubmit()}
            placeholder="例如 1, 2, 3……或你的名字"
            className="w-full px-5 py-3 rounded-lg bg-[#2a2a2a] text-white text-lg border border-white/20 focus:border-blue-500 focus:outline-none mb-4"
            autoFocus
          />
          <button
            onClick={handleIdSubmit}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white text-lg font-semibold rounded-lg transition-colors"
          >
            开始
          </button>
        </div>
      </div>
    );
  }

  if (gamePhase === 'instruction') {
    return (
      <div className="min-h-screen bg-[#2a2a2a] font-sans flex flex-col items-center justify-center selection:bg-blue-500/30 p-6">
        <div className="flex items-center gap-3 text-3xl font-bold text-white mb-12">
          <Grid size={36} />
          Grid Puzz
        </div>
        <div className="bg-[#3a3a3a] p-10 rounded-2xl shadow-2xl border border-white/10 w-full max-w-2xl">
          <h2 className="text-3xl font-bold text-white mb-6 text-center">实验说明</h2>
          <div className="text-gray-200 text-lg leading-relaxed space-y-4 mb-8">
            <p><strong>目标：</strong>给地图中的每个区域涂上颜色，使得相邻的区域颜色不同。</p>
            <p><strong>操作方法：</strong></p>
            <ul className="list-disc list-inside ml-4 space-y-1">
              <li>点击下方的色块选择颜色</li>
              <li>点击地图上的区域进行涂色</li>
              <li>点击橡皮擦可以擦除已涂的颜色</li>
            </ul>
            <p>共有 <strong>4</strong> 种可用颜色。如果两个相邻区域被涂成了相同的颜色，那么它们会变成红色，提示填色错误。</p>
            <p><strong>实验流程：</strong></p>
            <ul className="list-disc list-inside ml-4 space-y-1">
              <li>先进行 2 轮练习（每轮 10 个区域）</li>
              <li>再进行 10 轮正式实验（区域数从 20 到 45 递增）</li>
            </ul>
            <p className="text-yellow-300">注意：您的所有操作和用时都会被记录。</p>
          </div>
          <button
            onClick={handleStartPractice}
            className="w-full py-3 bg-green-600 hover:bg-green-700 text-white text-lg font-semibold rounded-lg transition-colors"
          >
            开始练习
          </button>
        </div>
      </div>
    );
  }

  if (!mapData) return null;

  return (
    <div className="min-h-screen bg-[#2a2a2a] font-sans selection:bg-blue-500/30 pb-12">
      {/* Header */}
      <header className="flex justify-between items-center p-4 text-white border-b border-white/10">
        <div className="flex items-center gap-3 text-xl font-bold">
          <Grid size={28} />
          Grid Puzz
        </div>
      </header>

      {/* Main Content - single column centered */}
      <main className="flex flex-col items-center mt-8 p-6">
        <h1 className="text-4xl font-bold text-white mb-4 tracking-tight">地图涂色</h1>
        <p className="text-xl text-gray-300 mb-8 leading-relaxed text-center">
          给地图涂色，使得相邻区域颜色不同。
        </p>

        {/* Round info */}
        <div className="bg-indigo-900/80 text-indigo-100 px-5 py-3 rounded-xl flex items-center justify-between shadow-md border border-indigo-500/30 mb-8 w-full" style={{ maxWidth: `${COLS * CELL_SIZE + 40}px` }}>
          <div className="flex items-center gap-2 font-bold">
            <ListOrdered size={18} />
            {gamePhase === 'practice'
              ? `练习 ${practiceIndex + 1} / ${PRACTICE_SIZES.length}`
              : `第 ${sequenceIndex + 1} 轮 / 共 ${ROUND_SIZES.length} 轮`}
            {sessionId && <span className="ml-2 px-2 py-0.5 bg-indigo-800 rounded text-xs">ID: {sessionId}</span>}
          </div>
          <div className="text-sm font-medium opacity-90 bg-indigo-950/50 px-3 py-1 rounded-md">
            {gamePhase === 'practice' ? PRACTICE_SIZES[practiceIndex] : ROUND_SIZES[sequenceIndex]} 个区域
          </div>
        </div>

        {/* Game Board */}
        <div className="p-5 bg-[#e0e0e0] rounded-sm shadow-2xl ring-2 ring-blue-500 ring-offset-4 ring-offset-[#2a2a2a] inline-block relative">
          <div
            className="grid bg-[#1a1a1a] border border-[#1a1a1a]"
            style={{
              gridTemplateColumns: `repeat(${COLS}, ${CELL_SIZE}px)`,
              gridTemplateRows: `repeat(${ROWS}, ${CELL_SIZE}px)`
            }}
            onMouseLeave={() => setHoveredRegion(null)}
          >
            {Array.from({ length: ROWS }).map((_, r) =>
              Array.from({ length: COLS }).map((_, c) => {
                const regionId = mapData.grid[r][c];
                const color = currentColors[regionId];
                const isError = errors.has(regionId);
                const isHovered = hoveredRegion === regionId;
                const cursor = isSolved ? 'default' : 'pointer';

                return (
                  <div
                    key={`${r}-${c}`}
                    className="box-border transition-colors duration-150 relative"
                    style={{
                      width: `${CELL_SIZE}px`,
                      height: `${CELL_SIZE}px`,
                      backgroundColor: color || '#d1d1d1',
                      borderTop: r === 0 || mapData.grid[r - 1][c] !== regionId ? '2px solid #1a1a1a' : 'none',
                      borderBottom: r === ROWS - 1 || mapData.grid[r + 1][c] !== regionId ? '2px solid #1a1a1a' : 'none',
                      borderLeft: c === 0 || mapData.grid[r][c - 1] !== regionId ? '2px solid #1a1a1a' : 'none',
                      borderRight: c === COLS - 1 || mapData.grid[r][c + 1] !== regionId ? '2px solid #1a1a1a' : 'none',
                      cursor,
                    }}
                    onMouseEnter={() => setHoveredRegion(regionId)}
                    onClick={() => handleRegionClick(regionId)}
                  >
                    {isHovered && !isSolved && <div className="absolute inset-0 bg-white/25 pointer-events-none" />}
                    {isError && <div className="absolute inset-0 pointer-events-none" style={{ backgroundColor: '#FF0000' }} />}
                  </div>
                );
              })
            )}
          </div>

          {isSolved && (
            <div className="absolute inset-0 bg-black/10 flex flex-col items-center justify-center backdrop-blur-[2px] z-10 gap-5">
              <div className="bg-white px-8 py-4 rounded-2xl shadow-2xl text-3xl font-bold text-green-600 flex items-center gap-3 animate-bounce">
                <Check size={40} strokeWidth={3} /> {isSequenceComplete ? "实验完成！" : "完成！"}
              </div>

              {gamePhase === 'practice' && !isPracticeRoundLast && (
                <button
                  onClick={handleNextPracticeRound}
                  className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-full text-lg font-semibold hover:bg-green-700 transition-colors shadow-lg"
                >
                  <PlaySquare size={20} /> 下一轮练习 ({practiceIndex + 2}/{PRACTICE_SIZES.length})
                </button>
              )}
              {gamePhase === 'practice' && isPracticeRoundLast && (
                <button
                  onClick={handleStartFormal}
                  className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-full text-lg font-semibold hover:bg-blue-700 transition-colors shadow-lg"
                >
                  <PlaySquare size={20} /> 开始正式实验
                </button>
              )}
              {gamePhase === 'playing' && !isSequenceComplete && (
                <button
                  onClick={handleNextRound}
                  className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-full text-lg font-semibold hover:bg-green-700 transition-colors shadow-lg"
                >
                  <PlaySquare size={20} /> 下一轮 ({sequenceIndex + 2}/{ROUND_SIZES.length})
                </button>
              )}
              {isSequenceComplete && (
                <>
                  <button
                    onClick={handleExportCSV}
                    className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-full text-lg font-semibold hover:bg-blue-700 transition-colors shadow-lg"
                  >
                    <Download size={20} /> 下载数据 (CSV)
                  </button>
                  <p className="text-white text-sm mt-2">请将下载好的 CSV 文件发送到邮箱：mfchen0703@gmail.com</p>
                </>
              )}
            </div>
          )}
        </div>

        {/* Color Palette */}
        <div className="flex gap-5 justify-center mt-10">
          {COLORS.map((color, idx) => (
            <button
              key={color}
              className={`w-14 h-14 rounded-full border-4 shadow-lg transition-all duration-200 ${selectedColor === color ? 'border-white scale-110 ring-4 ring-white/20' : 'border-[#2a2a2a] hover:scale-105'}`}
              style={{ backgroundColor: color }}
              onClick={() => setSelectedColor(color)}
              title={`颜色 ${idx + 1}`}
            />
          ))}
          <button
            className={`w-14 h-14 rounded-full border-4 shadow-lg flex items-center justify-center bg-[#e0e0e0] text-gray-800 transition-all duration-200 ${selectedColor === null ? 'border-white scale-110 ring-4 ring-white/20' : 'border-[#2a2a2a] hover:scale-105'}`}
            onClick={() => setSelectedColor(null)}
            title="橡皮擦"
          >
            <Eraser size={28} />
          </button>
        </div>
      </main>
    </div>
  );
}
