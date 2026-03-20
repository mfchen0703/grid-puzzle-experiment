import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Grid, Lightbulb, Globe, ChevronDown, Star, RotateCcw, Undo, Redo, Check, Share2, Eraser, Download, PlaySquare, ListOrdered } from 'lucide-react';

const ROWS = 12;
const COLS = 20;

const COLORS = [
  '#b58b72', // Tan
  '#8a9a65', // Green
  '#d4c473', // Yellow
  '#9c7c64', // Brown
];

type Difficulty = 'Tutorial' | 'Beginner' | 'Easy' | 'Medium' | 'Hard' | 'Expert' | 'Master' | 'Grandmaster';

type SequenceRound = { difficulty: Difficulty; prefill: number };
type GlobalHistoryEntry = HistoryEntry & { round: number; difficulty: string; prefill: number };

const DIFFICULTY_SETTINGS: Record<Difficulty, number> = {
  'Tutorial': 8,
  'Beginner': 12,
  'Easy': 15,
  'Medium': 30,
  'Hard': 50,
  'Expert': 80,
  'Master': 120,
  'Grandmaster': 160,
};

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

const generateMapData = (numRegions: number): MapData => {
  const grid = Array(ROWS).fill(null).map(() => Array(COLS).fill(-1));
  const regions: Region[] = [];
  
  // Initialize seeds
  for (let i = 0; i < numRegions; i++) {
    let r, c;
    do {
      r = Math.floor(Math.random() * ROWS);
      c = Math.floor(Math.random() * COLS);
    } while (grid[r][c] !== -1);
    grid[r][c] = i;
    regions.push({ id: i, cells: [[r, c]] });
  }
  
  // Grow regions
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
        const [nr, nc] = neighbors[Math.floor(Math.random() * neighbors.length)];
        if (grid[nr][nc] === -1) {
            grid[nr][nc] = i;
            regions[i].cells.push([nr, nc]);
            changed = true;
        }
      }
    }
  }
  
  // Fill any remaining isolated empty cells
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

  // Build adjacency list
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

const solveMap = (mapData: MapData, initialColors: (string | null)[] = []) => {
  const colors = Array(mapData.numRegions).fill(null);
  for (let i = 0; i < initialColors.length; i++) {
    if (initialColors[i]) colors[i] = initialColors[i];
  }
  
  const isValid = (region: number, color: string) => {
    for (const neighbor of mapData.adjacency[region]) {
      if (colors[neighbor] === color) return false;
    }
    return true;
  };
  
  const backtrack = (region: number): boolean => {
    if (region === mapData.numRegions) return true;
    
    if (colors[region] !== null) {
        return backtrack(region + 1);
    }
    
    const shuffledColors = [...COLORS].sort(() => Math.random() - 0.5);
    
    for (const c of shuffledColors) {
      if (isValid(region, c)) {
        colors[region] = c;
        if (backtrack(region + 1)) return true;
        colors[region] = null;
      }
    }
    return false;
  };
  
  backtrack(0);
  return colors;
};

export default function App() {
  const [difficulty, setDifficulty] = useState<Difficulty>('Medium');
  const [prefillCount, setPrefillCount] = useState<number>(3);
  const [mapData, setMapData] = useState<MapData | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [selectedColor, setSelectedColor] = useState<string | null>(COLORS[0]);
  const [hoveredRegion, setHoveredRegion] = useState<number | null>(null);
  const historyEndRef = useRef<HTMLDivElement>(null);
  const lastActionTime = useRef<number>(Date.now());
  
  const [sequence, setSequence] = useState<SequenceRound[] | null>(null);
  const [sequenceIndex, setSequenceIndex] = useState<number>(0);
  const [globalHistory, setGlobalHistory] = useState<GlobalHistoryEntry[]>([]);

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [gamePhase, setGamePhase] = useState<'input' | 'playing'>('input');
  const [inputId, setInputId] = useState('');
  const [loadError, setLoadError] = useState<string | null>(null);

  const initGame = useCallback((diff: Difficulty = difficulty, prefill: number = prefillCount) => {
    const numRegions = DIFFICULTY_SETTINGS[diff];
    const newMap = generateMapData(numRegions);
    
    const solution = solveMap(newMap);
    const initialColors = Array(numRegions).fill(null);
    
    if (solution && prefill > 0) {
      const indices = Array.from({length: numRegions}, (_, i) => i);
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      
      const actualPrefill = Math.min(prefill, numRegions);
      for (let i = 0; i < actualPrefill; i++) {
        initialColors[indices[i]] = solution[indices[i]];
      }
    }
    
    setMapData(newMap);
    lastActionTime.current = Date.now();
    setHistory([{ regionColors: initialColors, moveDescription: `Game Started (${diff}, ${prefill} pre-filled)`, timeTakenMs: 0 }]);
    setHistoryIndex(0);
  }, [difficulty, prefillCount]);

  useEffect(() => {
    if (gamePhase === 'playing' && !sequence) {
      initGame();
    }
  }, [initGame, gamePhase, sequence]);

  useEffect(() => {
    historyEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [historyIndex]);

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

  const isSequenceComplete = sequence !== null && sequenceIndex === sequence.length - 1 && isSolved;

  const allDisplayHistory = useMemo(() => {
    const items: { round: number; difficulty: string; prefill: number; moveDescription: string; timeTakenMs?: number }[] = [];
    for (const entry of globalHistory) {
      items.push(entry);
    }
    const currentRound = sequence ? sequenceIndex + 1 : 1;
    const currentDiff = sequence ? sequence[sequenceIndex].difficulty : difficulty;
    const currentPre = sequence ? sequence[sequenceIndex].prefill : prefillCount;
    for (const entry of history.slice(0, historyIndex + 1)) {
      items.push({ round: currentRound, difficulty: currentDiff, prefill: currentPre, moveDescription: entry.moveDescription, timeTakenMs: entry.timeTakenMs });
    }
    return items;
  }, [globalHistory, history, historyIndex, sequence, sequenceIndex, difficulty, prefillCount]);

  const handleRegionClick = (regionId: number) => {
    if (selectedColor === undefined || !mapData || isSolved) return;
    
    const isLocked = history.length > 0 && history[0].regionColors[regionId] !== null;
    if (isLocked) return;
    
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

  const handleNew = () => {
    setSequence(null);
    setGlobalHistory([]);
    initGame(difficulty, prefillCount);
  };
  
  const handleRestart = () => {
    if (historyIndex === 0) return;
    
    const now = Date.now();
    const timeTakenMs = now - lastActionTime.current;
    lastActionTime.current = now;
    
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({ regionColors: history[0].regionColors, moveDescription: 'Restarted Game', timeTakenMs });
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const handleUndo = () => {
    if (historyIndex > 0) {
      const now = Date.now();
      const timeTakenMs = now - lastActionTime.current;
      lastActionTime.current = now;

      const prevColors = history[historyIndex - 1].regionColors;
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push({ regionColors: prevColors, moveDescription: 'Undo', timeTakenMs });
      setHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
    }
  };

  const handleRedo = () => {
    if (historyIndex >= 2 && history[historyIndex].moveDescription === 'Undo') {
      const now = Date.now();
      const timeTakenMs = now - lastActionTime.current;
      lastActionTime.current = now;

      // Find the state before the last undo
      const redoColors = history[historyIndex - 1].regionColors;
      const newHistory = history.slice(0, historyIndex + 1);
      newHistory.push({ regionColors: redoColors, moveDescription: 'Redo', timeTakenMs });
      setHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
    }
  };

  const handleSolve = () => {
    if (!mapData || isSolved) return;
    const solution = solveMap(mapData, history[0].regionColors);
    
    const now = Date.now();
    const timeTakenMs = now - lastActionTime.current;
    lastActionTime.current = now;
    
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({ 
      regionColors: solution, 
      moveDescription: 'Auto-solved the puzzle',
      timeTakenMs
    });
    
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const parseSequenceCSV = useCallback((text: string) => {
    const lines = text.split('\n').map(l => l.trim()).filter(l => l);
    const parsedSequence: SequenceRound[] = [];
    
    let startIndex = 0;
    if (lines[0].toLowerCase().includes('difficulty') || lines[0].toLowerCase().includes('prefill')) {
      startIndex = 1;
    }
    
    for (let i = startIndex; i < lines.length; i++) {
      const parts = lines[i].split(',');
      if (parts.length >= 2) {
        const diffStr = parts[0].trim();
        const prefillStr = parts[1].trim();
        
        const diffKey = (Object.keys(DIFFICULTY_SETTINGS) as Difficulty[]).find(
          k => k.toLowerCase() === diffStr.toLowerCase()
        );
        
        const prefill = parseInt(prefillStr, 10);
        
        if (diffKey && !isNaN(prefill)) {
          parsedSequence.push({ difficulty: diffKey, prefill });
        }
      }
    }
    
    if (parsedSequence.length > 0) {
      setSequence(parsedSequence);
      setSequenceIndex(0);
      setGlobalHistory([]);
      setDifficulty(parsedSequence[0].difficulty);
      setPrefillCount(parsedSequence[0].prefill);
      initGame(parsedSequence[0].difficulty, parsedSequence[0].prefill);
    } else {
      alert("Invalid CSV format. Expected: Difficulty,Prefill (e.g., Medium,3)");
    }
  }, [initGame]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get('id');
    if (id) {
      setSessionId(id);
      setGamePhase('playing');
      fetch(`${import.meta.env.BASE_URL}sequences/${id}.csv`)
        .then(res => {
          if (!res.ok) throw new Error('Sequence file not found');
          return res.text();
        })
        .then(text => {
          parseSequenceCSV(text);
        })
        .catch(err => {
          console.error("Could not load sequence from URL:", err);
        });
    }
  }, [parseSequenceCSV]);

  const handleIdSubmit = () => {
    const id = inputId.trim();
    if (!id) return;
    setLoadError(null);
    fetch(`${import.meta.env.BASE_URL}sequences/${id}.csv`)
      .then(res => {
        if (!res.ok) throw new Error('not found');
        return res.text();
      })
      .then(text => {
        setSessionId(id);
        setGamePhase('playing');
        parseSequenceCSV(text);
      })
      .catch(() => {
        setLoadError(`Sequence "${id}" not found. Please check your ID.`);
      });
  };

  const handleNextRound = () => {
    if (!sequence || sequenceIndex >= sequence.length - 1) return;
    
    const currentRoundHistory = history.slice(0, historyIndex + 1).map(h => ({
      ...h, 
      round: sequenceIndex + 1, 
      difficulty: sequence[sequenceIndex].difficulty, 
      prefill: sequence[sequenceIndex].prefill
    }));
    
    setGlobalHistory(prev => [...prev, ...currentRoundHistory]);
    
    const nextIndex = sequenceIndex + 1;
    setSequenceIndex(nextIndex);
    const nextDiff = sequence[nextIndex].difficulty;
    const nextPrefill = sequence[nextIndex].prefill;
    
    setDifficulty(nextDiff);
    setPrefillCount(nextPrefill);
    initGame(nextDiff, nextPrefill);
  };

  const handleExportCSV = () => {
    let csvContent = "";
    const currentRoundHistory = history.slice(0, historyIndex + 1);
    const idPrefix = sessionId ? `${sessionId},` : "";
    const headerIdPrefix = sessionId ? "SessionID," : "";
    
    if (sequence) {
      csvContent = `${headerIdPrefix}Round,Difficulty,Prefill,Step,Action,TimeTaken(s)\n`;
      const currentMapped = currentRoundHistory.map(h => ({
        ...h, 
        round: sequenceIndex + 1, 
        difficulty: sequence[sequenceIndex].difficulty, 
        prefill: sequence[sequenceIndex].prefill
      }));
      const allHistory = [...globalHistory, ...currentMapped];
      
      allHistory.forEach((entry, idx) => {
        const time = entry.timeTakenMs ? (entry.timeTakenMs / 1000).toFixed(1) : "0.0";
        const desc = `"${entry.moveDescription.replace(/"/g, '""')}"`;
        csvContent += `${idPrefix}${entry.round},${entry.difficulty},${entry.prefill},${idx},${desc},${time}\n`;
      });
    } else {
      csvContent = `${headerIdPrefix}Step,Action,TimeTaken(s)\n`;
      currentRoundHistory.forEach((entry, idx) => {
        const time = entry.timeTakenMs ? (entry.timeTakenMs / 1000).toFixed(1) : "0.0";
        const desc = `"${entry.moveDescription.replace(/"/g, '""')}"`;
        csvContent += `${idPrefix}${idx},${desc},${time}\n`;
      });
    }

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', sessionId ? `action_${sessionId}.csv` : `grid-puzz-history-${Date.now()}.csv`);
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
          <h2 className="text-2xl font-bold text-white mb-2 text-center">Enter Your ID</h2>
          <p className="text-gray-400 text-sm mb-8 text-center">Input your assigned number to start the experiment.</p>
          <input
            type="text"
            value={inputId}
            onChange={e => { setInputId(e.target.value); setLoadError(null); }}
            onKeyDown={e => e.key === 'Enter' && handleIdSubmit()}
            placeholder="e.g. 1, 2, 3..."
            className="w-full px-5 py-3 rounded-lg bg-[#2a2a2a] text-white text-lg border border-white/20 focus:border-blue-500 focus:outline-none mb-4"
            autoFocus
          />
          {loadError && (
            <p className="text-red-400 text-sm mb-4 text-center">{loadError}</p>
          )}
          <button
            onClick={handleIdSubmit}
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white text-lg font-semibold rounded-lg transition-colors"
          >
            Start
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
        <div className="flex items-center gap-6 text-sm font-medium">
          <button className="flex items-center gap-1.5 hover:text-gray-300 transition-colors">
            <Lightbulb size={16} /> How To
          </button>
          <button className="flex items-center gap-1.5 hover:text-gray-300 transition-colors">
            <Globe size={16} /> English <ChevronDown size={14} />
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto mt-12 grid grid-cols-1 lg:grid-cols-[1fr_auto] gap-16 p-6">
        
        {/* Left Column */}
        <div className="flex flex-col">
          <h1 className="text-6xl font-bold text-white mb-6 tracking-tight">Map</h1>
          <p className="text-2xl text-gray-300 mb-10 leading-relaxed">
            Color the map so that no two adjacent regions share the same color.
          </p>
          
          {/* Action History */}
          <div className="bg-[#3a3a3a] p-5 rounded-xl text-gray-300 h-80 overflow-y-auto shadow-inner border border-white/5 flex-grow">
            <h3 className="font-bold mb-4 text-white sticky top-0 bg-[#3a3a3a] pb-3 border-b border-gray-600 text-lg flex items-center justify-between">
              <div className="flex items-center gap-2"><RotateCcw size={18} /> Action History</div>
              {isSequenceComplete && (
                <button onClick={handleExportCSV} className="text-sm bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md flex items-center gap-1.5 transition-colors shadow-sm">
                  <Download size={14} /> CSV
                </button>
              )}
            </h3>
            <div className="flex flex-col gap-1.5">
              {allDisplayHistory.map((entry, idx) => {
                const prevRound = idx > 0 ? allDisplayHistory[idx - 1].round : 0;
                const showRoundHeader = sequence && entry.round !== prevRound;
                return (
                  <React.Fragment key={idx}>
                    {showRoundHeader && (
                      <div className="text-xs font-bold text-indigo-300 mt-3 mb-1 px-3 py-1 bg-indigo-900/40 rounded">
                        Round {entry.round}: {entry.difficulty} ({entry.prefill} pre-filled)
                      </div>
                    )}
                    <div
                      className={`text-sm py-2 px-3 rounded-md transition-colors flex justify-between items-center ${idx === allDisplayHistory.length - 1 ? 'bg-blue-500/20 text-blue-200 font-medium' : 'hover:bg-[#4a4a4a]'}`}
                    >
                      <span>
                        <span className="text-gray-500 mr-3 w-6 inline-block text-right">{idx}.</span>
                        {entry.moveDescription}
                      </span>
                      {entry.timeTakenMs !== undefined && entry.timeTakenMs > 0 && (
                        <span className="text-xs opacity-60 font-mono">
                          {(entry.timeTakenMs / 1000).toFixed(1)}s
                        </span>
                      )}
                    </div>
                  </React.Fragment>
                );
              })}
              <div ref={historyEndRef} />
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="flex flex-col items-center">
          
          {/* Controls */}
          <div className="flex flex-col gap-4 mb-10">
            {sequence && (
              <div className="bg-indigo-900/80 text-indigo-100 px-5 py-3 rounded-xl flex items-center justify-between shadow-md border border-indigo-500/30">
                <div className="flex items-center gap-2 font-bold">
                  <ListOrdered size={18} />
                  Sequence Active: Round {sequenceIndex + 1} of {sequence.length}
                  {sessionId && <span className="ml-2 px-2 py-0.5 bg-indigo-800 rounded text-xs">ID: {sessionId}</span>}
                </div>
                <div className="text-sm font-medium opacity-90 bg-indigo-950/50 px-3 py-1 rounded-md">
                  Current: {sequence[sequenceIndex].difficulty} ({sequence[sequenceIndex].prefill} pre-filled)
                </div>
              </div>
            )}
            
            <div className="flex justify-center gap-4">
              <div className="flex items-center gap-2 px-5 py-2.5 bg-[#e0e0e0] text-gray-800 rounded-full text-sm font-semibold shadow-sm">
                <Grid size={16} /> Size: {difficulty}
              </div>
              <div className="flex items-center gap-2 px-5 py-2.5 bg-[#e0e0e0] text-gray-800 rounded-full text-sm font-semibold shadow-sm">
                <Star size={16} /> Pre-filled: {prefillCount}
              </div>
            </div>
            <div className="flex justify-center gap-3">
              <button onClick={handleNew} className="flex items-center gap-2 px-5 py-2.5 bg-[#e0e0e0] text-gray-800 rounded-full text-sm font-semibold hover:bg-white transition-colors shadow-sm">
                <Star size={16} /> New
              </button>
              <button onClick={handleRestart} className="flex items-center gap-2 px-5 py-2.5 bg-[#e0e0e0] text-gray-800 rounded-full text-sm font-semibold hover:bg-white transition-colors shadow-sm">
                <RotateCcw size={16} /> Restart
              </button>
              <button onClick={handleUndo} disabled={historyIndex <= 0} className="flex items-center gap-2 px-5 py-2.5 bg-[#e0e0e0] text-gray-800 rounded-full text-sm font-semibold hover:bg-white transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed">
                <Undo size={16} /> Undo
              </button>
              <button onClick={handleRedo} disabled={historyIndex >= history.length - 1} className="flex items-center gap-2 px-5 py-2.5 bg-[#e0e0e0] text-gray-800 rounded-full text-sm font-semibold hover:bg-white transition-colors shadow-sm disabled:opacity-50 disabled:cursor-not-allowed">
                <Redo size={16} /> Redo
              </button>
              <button onClick={handleSolve} className="flex items-center gap-2 px-5 py-2.5 bg-[#e0e0e0] text-gray-800 rounded-full text-sm font-semibold hover:bg-white transition-colors shadow-sm">
                <Check size={16} /> Solve
              </button>
            </div>
          </div>

          {/* Game Board Container */}
          <div className="p-5 bg-[#e0e0e0] rounded-sm shadow-2xl ring-2 ring-blue-500 ring-offset-4 ring-offset-[#2a2a2a] inline-block relative">
             {/* Map Grid */}
             <div 
               className="grid bg-[#1a1a1a] border border-[#1a1a1a]" 
               style={{ 
                 gridTemplateColumns: `repeat(${COLS}, 32px)`,
                 gridTemplateRows: `repeat(${ROWS}, 32px)`
               }}
               onMouseLeave={() => setHoveredRegion(null)}
             >
               {Array.from({ length: ROWS }).map((_, r) =>
                  Array.from({ length: COLS }).map((_, c) => {
                    const regionId = mapData.grid[r][c];
                    const color = currentColors[regionId];
                    const isError = errors.has(regionId);
                    const isHovered = hoveredRegion === regionId;
                    const isLocked = history.length > 0 && history[0].regionColors[regionId] !== null;
                    const cursor = isSolved || isLocked ? 'default' : 'pointer';
                    
                    return (
                      <div
                        key={`${r}-${c}`}
                        className="box-border transition-colors duration-150 relative"
                        style={{
                          width: '32px',
                          height: '32px',
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
                        {isHovered && !isSolved && !isLocked && <div className="absolute inset-0 bg-white/25 pointer-events-none" />}
                        {isError && <div className="absolute inset-0 bg-red-500/60 pointer-events-none" />}
                        {isLocked && <div className="absolute inset-0 flex items-center justify-center pointer-events-none"><div className="w-1.5 h-1.5 rounded-full bg-black/20" /></div>}
                      </div>
                    );
                  })
                )}
             </div>
             
             {isSolved && (
               <div className="absolute inset-0 bg-black/10 flex flex-col items-center justify-center backdrop-blur-[2px] z-10 gap-5">
                 <div className="bg-white px-8 py-4 rounded-2xl shadow-2xl text-3xl font-bold text-green-600 flex items-center gap-3 animate-bounce">
                   <Check size={40} strokeWidth={3} /> {sequence && sequenceIndex === sequence.length - 1 ? "Sequence Complete!" : "Solved!"}
                 </div>
                 
                 {sequence && sequenceIndex < sequence.length - 1 ? (
                   <button
                     onClick={handleNextRound}
                     className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-full text-lg font-semibold hover:bg-green-700 transition-colors shadow-lg"
                   >
                     <PlaySquare size={20} /> Next Round ({sequenceIndex + 2}/{sequence.length})
                   </button>
                 ) : (
                   <button
                     onClick={handleExportCSV}
                     className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-full text-lg font-semibold hover:bg-blue-700 transition-colors shadow-lg"
                   >
                     <Download size={20} /> Export All History (CSV)
                   </button>
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
                title={`Color ${idx + 1}`}
              />
            ))}
            <button
              className={`w-14 h-14 rounded-full border-4 shadow-lg flex items-center justify-center bg-[#e0e0e0] text-gray-800 transition-all duration-200 ${selectedColor === null ? 'border-white scale-110 ring-4 ring-white/20' : 'border-[#2a2a2a] hover:scale-105'}`}
              onClick={() => setSelectedColor(null)}
              title="Eraser"
            >
              <Eraser size={28} />
            </button>
          </div>

          {/* Footer Links */}
          <div className="flex items-center gap-5 mt-14 text-gray-300">
            <span className="font-medium">Link to this puzzle by:</span>
            <button className="flex items-center gap-2 px-5 py-2.5 bg-white text-gray-800 rounded-full text-sm font-semibold hover:bg-gray-200 transition-colors shadow-sm">
              <Share2 size={16} /> Game ID
            </button>
            <button className="flex items-center gap-2 px-5 py-2.5 bg-white text-gray-800 rounded-full text-sm font-semibold hover:bg-gray-200 transition-colors shadow-sm">
              <Share2 size={16} /> Random Seed
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
