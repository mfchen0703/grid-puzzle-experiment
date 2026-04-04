import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Check, Download, Grid, ListOrdered, PlaySquare } from 'lucide-react';
import {
  buildAdjacencyMap,
  CELL_SIZE,
  COLORS,
  COLS,
  Experiment2Materials,
  ROWS,
  getConflictEdges,
} from './gameLogic';

type HistoryEntry = {
  regionColors: number[];
  moveDescription: string;
  timeTakenMs?: number;
};

export default function Experiment2Game({ sessionId }: { sessionId: string }) {
  const [materials, setMaterials] = useState<Experiment2Materials | null>(null);
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [phase, setPhase] = useState<'instruction' | 'playing' | 'finished'>('instruction');
  const [roundIndex, setRoundIndex] = useState(0);
  const [selectedColor, setSelectedColor] = useState(0);
  const [hoveredRegion, setHoveredRegion] = useState<number | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyIndex, setHistoryIndex] = useState(0);
  const [globalHistory, setGlobalHistory] = useState<Array<HistoryEntry & { round: string }>>([]);
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');
  const lastActionTime = useRef(Date.now());

  useEffect(() => {
    let cancelled = false;
    async function loadMaterials() {
      setLoadState('loading');
      try {
        const res = await fetch(`${import.meta.env.BASE_URL}experiment2/rounds.json`);
        if (!res.ok) {
          throw new Error(`Failed to fetch rounds.json: ${res.status}`);
        }
        const data: Experiment2Materials = await res.json();
        if (!cancelled) {
          setMaterials(data);
          setLoadState('ready');
        }
      } catch {
        if (!cancelled) {
          setLoadState('error');
        }
      }
    }
    loadMaterials();
    return () => {
      cancelled = true;
    };
  }, []);

  const rounds = materials?.rounds ?? [];
  const round = rounds[roundIndex] ?? null;
  const adjacency = useMemo(
    () => (round ? buildAdjacencyMap(round.mapData) : null),
    [round],
  );

  if (loadState === 'loading') {
    return (
      <div className="min-h-screen bg-[#0f172a] text-white flex items-center justify-center p-6">
        <div className="w-full max-w-2xl rounded-3xl border border-cyan-500/20 bg-slate-900/80 p-10 shadow-2xl text-center">
          <div className="mb-4 text-3xl font-bold">实验 2 材料加载中</div>
          <p className="text-slate-300">正在读取预生成的地图与初始颜色，请稍候。</p>
        </div>
      </div>
    );
  }

  if (loadState === 'error' || !materials || !round || !adjacency) {
    return (
      <div className="min-h-screen bg-[#0f172a] text-white flex items-center justify-center p-6">
        <div className="w-full max-w-2xl rounded-3xl border border-red-500/20 bg-slate-900/80 p-10 shadow-2xl text-center">
          <div className="mb-4 text-3xl font-bold">实验 2 加载失败</div>
          <p className="text-slate-300">没有成功读取预生成材料。请刷新页面，或检查 `experiment1/public/experiment2/rounds.json` 是否存在。</p>
        </div>
      </div>
    );
  }

  const currentColors = history[historyIndex]?.regionColors ?? round.initialColors;
  const conflictEdges = useMemo(() => getConflictEdges(adjacency, currentColors), [adjacency, currentColors]);
  const errorRegions = useMemo(() => {
    const set = new Set<number>();
    for (const [a, b] of conflictEdges) {
      set.add(a);
      set.add(b);
    }
    return set;
  }, [conflictEdges]);
  const isSolved = conflictEdges.length === 0;

  const startExperiment = () => {
    setPhase('playing');
    setRoundIndex(0);
    lastActionTime.current = Date.now();
    setHistory([{ regionColors: [...rounds[0].initialColors], moveDescription: `Game Started (45 regions, ${rounds[0].conflictEdges.length} conflicts)`, timeTakenMs: 0 }]);
    setHistoryIndex(0);
    setGlobalHistory([]);
  };

  const handleRegionClick = (regionId: number) => {
    if (phase !== 'playing' || isSolved || currentColors[regionId] === selectedColor) {
      return;
    }
    const nextColors = [...currentColors];
    nextColors[regionId] = selectedColor;
    const now = Date.now();
    const timeTakenMs = now - lastActionTime.current;
    lastActionTime.current = now;
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push({
      regionColors: nextColors,
      moveDescription: `Recolored Region ${regionId + 1} to Color ${selectedColor + 1}`,
      timeTakenMs,
    });
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const commitCurrentRound = () => {
    const roundHistory = history.slice(0, historyIndex + 1).map((entry) => ({
      ...entry,
      round: `${roundIndex + 1}`,
    }));
    setGlobalHistory((prev) => [...prev, ...roundHistory]);
  };

  const handleNextRound = () => {
    commitCurrentRound();
    const nextRoundIndex = roundIndex + 1;
    if (nextRoundIndex >= rounds.length) {
      setPhase('finished');
      return;
    }
    const nextRound = rounds[nextRoundIndex];
    setRoundIndex(nextRoundIndex);
    setHistory([{ regionColors: [...nextRound.initialColors], moveDescription: `Game Started (45 regions, ${nextRound.conflictEdges.length} conflicts)`, timeTakenMs: 0 }]);
    setHistoryIndex(0);
    lastActionTime.current = Date.now();
  };

  const handleExportCSV = async () => {
    const currentRoundHistory = history.slice(0, historyIndex + 1).map((entry) => ({
      ...entry,
      round: `${roundIndex + 1}`,
    }));
    const allHistory = [...globalHistory, ...currentRoundHistory];

    let csvContent = '[Actions]\n';
    csvContent += 'SessionID,Experiment,Round,NumRegions,Step,Action,TimeTaken(s)\n';
    allHistory.forEach((entry, idx) => {
      const time = entry.timeTakenMs ? (entry.timeTakenMs / 1000).toFixed(1) : '0.0';
      const desc = `"${entry.moveDescription.replace(/"/g, '""')}"`;
      csvContent += `${sessionId},experiment2,${entry.round},45,${idx},${desc},${time}\n`;
    });

    csvContent += '\n[InitialState]\n';
    csvContent += 'Round,Region,InitialColor\n';
    rounds.forEach((roundData, idx) => {
      roundData.initialColors.forEach((color, region) => {
        csvContent += `${idx + 1},${region + 1},${color + 1}\n`;
      });
    });

    csvContent += '\n[Adjacency]\n';
    csvContent += 'Round,NumRegions,Region_A,Region_B\n';
    rounds.forEach((roundData, idx) => {
      for (const [a, b] of roundData.mapData.adjacencyPairs) {
        csvContent += `${idx + 1},45,${a + 1},${b + 1}\n`;
      }
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `data_experiment2_${sessionId}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    setUploadStatus('uploading');
    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId: `experiment2_${sessionId}`, csv: csvContent }),
      });
      setUploadStatus(res.ok ? 'success' : 'error');
    } catch {
      setUploadStatus('error');
    }
  };

  if (phase === 'instruction') {
    return (
      <div className="min-h-screen bg-[#1e293b] text-white flex items-center justify-center p-6">
        <div className="w-full max-w-3xl rounded-3xl border border-slate-500/30 bg-slate-900/80 p-10 shadow-2xl">
          <div className="mb-8 flex items-center gap-3 text-3xl font-bold">
            <Grid size={36} />
            实验 2
          </div>
          <div className="space-y-4 text-lg leading-relaxed text-slate-200">
            <p><strong>目标：</strong>地图一开始已经全部着色，但存在若干颜色冲突。你的任务是修改颜色，使整个地图最终没有任何相邻同色。</p>
            <p><strong>关键点：</strong>不只是当前冲突的区域可能需要修改，一些当前看起来没冲突的区域也可能必须改色，才能让整张图恢复合法。</p>
            <p><strong>操作方法：</strong>先选择一种颜色，再点击地图中的区域，将该区域改成所选颜色。</p>
            <p><strong>实验流程：</strong>共 {rounds.length} 轮正式实验，每轮都是 45 个区域。每轮初始地图都含有需要规划的冲突结构。</p>
            <p className="text-amber-300">注意：你的所有修改步骤和用时都会被记录。</p>
          </div>
          <button
            onClick={startExperiment}
            className="mt-8 w-full rounded-xl bg-emerald-600 px-6 py-3 text-lg font-semibold hover:bg-emerald-700 transition-colors"
          >
            开始实验 2
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0f172a] font-sans selection:bg-cyan-500/30 pb-12">
      <header className="flex justify-between items-center p-4 text-white border-b border-white/10">
        <div className="flex items-center gap-3 text-xl font-bold">
          <Grid size={28} />
          实验 2
        </div>
      </header>

      <main className="flex flex-col items-center mt-8 p-6">
        <h1 className="text-4xl font-bold text-white mb-4 tracking-tight">冲突修复</h1>
        <p className="text-xl text-slate-300 mb-8 leading-relaxed text-center">修改区域颜色，使整张地图恢复为无冲突状态。</p>

        <div className="bg-cyan-950/80 text-cyan-100 px-5 py-3 rounded-xl flex items-center justify-between shadow-md border border-cyan-500/30 mb-8 w-full" style={{ maxWidth: `${COLS * CELL_SIZE + 40}px` }}>
          <div className="flex items-center gap-2 font-bold">
            <ListOrdered size={18} />
            第 {roundIndex + 1} 轮 / 共 {rounds.length} 轮
            <span className="ml-2 px-2 py-0.5 bg-cyan-900 rounded text-xs">ID: {sessionId}</span>
          </div>
          <div className="text-sm font-medium opacity-90 bg-cyan-950/50 px-3 py-1 rounded-md">
            当前冲突边数: {conflictEdges.length}
          </div>
        </div>

        <div className="p-5 bg-[#e2e8f0] rounded-sm shadow-2xl ring-2 ring-cyan-500 ring-offset-4 ring-offset-[#0f172a] inline-block relative">
          <div
            className="grid bg-[#0f172a] border border-[#0f172a]"
            style={{
              gridTemplateColumns: `repeat(${COLS}, ${CELL_SIZE}px)`,
              gridTemplateRows: `repeat(${ROWS}, ${CELL_SIZE}px)`,
            }}
            onMouseLeave={() => setHoveredRegion(null)}
          >
            {Array.from({ length: ROWS }).map((_, r) =>
              Array.from({ length: COLS }).map((_, c) => {
                const regionId = round.mapData.grid[r][c];
                const color = COLORS[currentColors[regionId]];
                const isError = errorRegions.has(regionId);
                const isHovered = hoveredRegion === regionId;

                return (
                  <div
                    key={`${r}-${c}`}
                    className="box-border transition-colors duration-150 relative"
                    style={{
                      width: `${CELL_SIZE}px`,
                      height: `${CELL_SIZE}px`,
                      backgroundColor: color,
                      borderTop: r === 0 || round.mapData.grid[r - 1][c] !== regionId ? '2px solid #0f172a' : 'none',
                      borderBottom: r === ROWS - 1 || round.mapData.grid[r + 1][c] !== regionId ? '2px solid #0f172a' : 'none',
                      borderLeft: c === 0 || round.mapData.grid[r][c - 1] !== regionId ? '2px solid #0f172a' : 'none',
                      borderRight: c === COLS - 1 || round.mapData.grid[r][c + 1] !== regionId ? '2px solid #0f172a' : 'none',
                      cursor: isSolved ? 'default' : 'pointer',
                    }}
                    onMouseEnter={() => setHoveredRegion(regionId)}
                    onClick={() => handleRegionClick(regionId)}
                  >
                    {isHovered && !isSolved && <div className="absolute inset-0 bg-white/20 pointer-events-none" />}
                    {isError && (
                      <div
                        className="absolute inset-0 pointer-events-none box-border"
                        style={{
                          borderTop:
                            r === 0 || round.mapData.grid[r - 1][c] !== regionId
                              ? '3px solid #ef4444'
                              : 'none',
                          borderBottom:
                            r === ROWS - 1 || round.mapData.grid[r + 1][c] !== regionId
                              ? '3px solid #ef4444'
                              : 'none',
                          borderLeft:
                            c === 0 || round.mapData.grid[r][c - 1] !== regionId
                              ? '3px solid #ef4444'
                              : 'none',
                          borderRight:
                            c === COLS - 1 || round.mapData.grid[r][c + 1] !== regionId
                              ? '3px solid #ef4444'
                              : 'none',
                        }}
                      />
                    )}
                  </div>
                );
              }),
            )}
          </div>

          {isSolved && (
            <div className="absolute inset-0 bg-black/15 flex flex-col items-center justify-center backdrop-blur-[2px] z-10 gap-5">
              <div className="bg-white px-8 py-4 rounded-2xl shadow-2xl text-3xl font-bold text-emerald-600 flex items-center gap-3">
                <Check size={40} strokeWidth={3} /> {roundIndex === rounds.length - 1 ? '实验完成！' : '本轮完成！'}
              </div>
              {roundIndex < rounds.length - 1 ? (
                <button onClick={handleNextRound} className="flex items-center gap-2 px-6 py-3 bg-emerald-600 text-white rounded-full text-lg font-semibold hover:bg-emerald-700 transition-colors shadow-lg">
                  <PlaySquare size={20} /> 下一轮 ({roundIndex + 2}/{rounds.length})
                </button>
              ) : (
                <>
                  <button onClick={handleExportCSV} className="flex items-center gap-2 px-6 py-3 bg-cyan-600 text-white rounded-full text-lg font-semibold hover:bg-cyan-700 transition-colors shadow-lg">
                    <Download size={20} /> 下载数据 (CSV)
                  </button>
                  {uploadStatus === 'uploading' && <p className="text-amber-300 text-sm mt-2">正在上传数据...</p>}
                  {uploadStatus === 'success' && <p className="text-emerald-300 text-sm mt-2">数据已成功上传！</p>}
                  {uploadStatus === 'error' && <p className="text-red-300 text-sm mt-2">上传失败，请保留下载的 CSV 文件。</p>}
                </>
              )}
            </div>
          )}
        </div>

        <div className="flex gap-5 justify-center mt-10">
          {COLORS.map((color, idx) => (
            <button
              key={color}
              className={`w-14 h-14 rounded-full border-4 shadow-lg transition-all duration-200 ${selectedColor === idx ? 'border-white scale-110 ring-4 ring-white/20' : 'border-[#0f172a] hover:scale-105'}`}
              style={{ backgroundColor: color }}
              onClick={() => setSelectedColor(idx)}
              title={`颜色 ${idx + 1}`}
            />
          ))}
        </div>
      </main>
    </div>
  );
}
