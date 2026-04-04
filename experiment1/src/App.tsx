import { Grid } from 'lucide-react';
import { useEffect, useState } from 'react';
import Experiment1Game from './Experiment1Game';
import Experiment2Game from './experiment2/Experiment2Game';

type ExperimentChoice = 'experiment1' | 'experiment2';
type Route = 'home' | ExperimentChoice;

function getRouteFromPath(pathname: string): Route {
  if (pathname === '/experiment1') {
    return 'experiment1';
  }
  if (pathname === '/experiment2') {
    return 'experiment2';
  }
  return 'home';
}

function getPathFromRoute(route: Route): string {
  if (route === 'experiment1') {
    return '/experiment1';
  }
  if (route === 'experiment2') {
    return '/experiment2';
  }
  return '/';
}

export default function App() {
  const [inputId, setInputId] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [route, setRoute] = useState<Route>(() => getRouteFromPath(window.location.pathname));

  useEffect(() => {
    const syncFromLocation = () => {
      const params = new URLSearchParams(window.location.search);
      const id = params.get('id');
      setRoute(getRouteFromPath(window.location.pathname));
      if (id) {
        setInputId(id);
        setSessionId(id);
      }
    };

    syncFromLocation();
    window.addEventListener('popstate', syncFromLocation);
    return () => window.removeEventListener('popstate', syncFromLocation);
  }, []);

  const handleSubmitId = () => {
    const trimmed = inputId.trim();
    if (!trimmed) {
      return;
    }
    setSessionId(trimmed);
    const params = new URLSearchParams(window.location.search);
    params.set('id', trimmed);
    window.history.replaceState({}, '', `${getPathFromRoute(route)}?${params.toString()}`);
  };

  const navigateTo = (nextRoute: Route) => {
    const params = new URLSearchParams();
    if (sessionId) {
      params.set('id', sessionId);
    }
    const nextUrl = params.toString()
      ? `${getPathFromRoute(nextRoute)}?${params.toString()}`
      : getPathFromRoute(nextRoute);
    window.history.pushState({}, '', nextUrl);
    setRoute(nextRoute);
  };

  if (sessionId && route === 'experiment1') {
    return <Experiment1Game sessionId={sessionId} />;
  }

  if (sessionId && route === 'experiment2') {
    return <Experiment2Game sessionId={sessionId} />;
  }

  return (
    <div className="min-h-screen bg-[#111827] font-sans flex flex-col items-center justify-center selection:bg-cyan-500/30 p-6">
      <div className="flex items-center gap-3 text-3xl font-bold text-white mb-12">
        <Grid size={36} />
        Grid Puzzle Experiments
      </div>

      <div className="bg-[#1f2937] p-10 rounded-2xl shadow-2xl border border-white/10 w-full max-w-2xl">
        <h2 className="text-3xl font-bold text-white mb-3 text-center">输入编号并选择实验</h2>
        <p className="text-gray-400 text-sm mb-8 text-center">先输入你的编号，然后选择进行实验 1 或实验 2。</p>

        <input
          type="text"
          value={inputId}
          onChange={(e) => setInputId(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSubmitId()}
          placeholder="例如 1, 2, 3……或你的名字"
          className="w-full px-5 py-3 rounded-lg bg-[#111827] text-white text-lg border border-white/20 focus:border-cyan-500 focus:outline-none mb-6"
          autoFocus
        />

        {!sessionId ? (
          <button
            onClick={handleSubmitId}
            className="w-full py-3 bg-cyan-600 hover:bg-cyan-700 text-white text-lg font-semibold rounded-lg transition-colors"
          >
            确认编号
          </button>
        ) : (
          <div className="space-y-4">
            <div className="rounded-xl border border-cyan-500/30 bg-cyan-950/40 p-4 text-cyan-100">
              当前编号：<strong>{sessionId}</strong>
            </div>

            <button
              onClick={() => navigateTo('experiment1')}
              className="w-full rounded-xl border border-white/10 bg-slate-800 px-6 py-5 text-left hover:bg-slate-700 transition-colors"
            >
              <div className="text-xl font-semibold text-white">实验 1: 从空白地图开始涂色</div>
              <div className="mt-1 text-sm text-slate-300">先做 2 轮练习，再完成 10 轮正式实验，目标是给所有区域填色并避免相邻同色。</div>
            </button>

            <button
              onClick={() => navigateTo('experiment2')}
              className="w-full rounded-xl border border-white/10 bg-slate-800 px-6 py-5 text-left hover:bg-slate-700 transition-colors"
            >
              <div className="text-xl font-semibold text-white">实验 2: 修复已有颜色冲突</div>
              <div className="mt-1 text-sm text-slate-300">共 10 轮，每轮 45 个区域。地图初始已着色，但含有需要规划的冲突，你需要通过改色恢复无冲突状态。</div>
            </button>

            <button
              onClick={() => {
                setSessionId(null);
                navigateTo('home');
              }}
              className="w-full py-3 bg-transparent hover:bg-white/5 text-gray-300 text-sm font-medium rounded-lg transition-colors border border-white/10"
            >
              重新输入编号
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
