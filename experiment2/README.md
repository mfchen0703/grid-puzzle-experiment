## Experiment 2

这个目录现在主要存放实验 2 的设计说明与独立材料。

当前可运行的实验 2 前端代码已经移动到：

- `experiment1/src/experiment2/Experiment2Game.tsx`
- `experiment1/src/experiment2/gameLogic.ts`
- `experiment1/public/experiment2/rounds.json`

这样做的原因是 GitHub Pages 的构建只会在 `experiment1/` 下安装依赖；如果运行时代码放在仓库根目录的 `experiment2/`，CI 无法稳定解析 `react` 依赖。

为了避免实验 2 在浏览器里现场生成 10 轮 45-region 材料，现在使用静态预生成文件：

- `experiment2/generate_rounds_json.py` 负责生成材料
- 页面运行时只读取 `experiment1/public/experiment2/rounds.json`
