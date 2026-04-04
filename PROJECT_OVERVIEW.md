# 项目总览

## 1. 这个项目在研究什么

这个项目研究的是：人在解决一种带有图结构约束的地图涂色任务时，是如何一步一步做选择的。

被试看见的是一个由不规则区域组成的地图。每个区域都可以看成图中的一个节点；如果两个区域共享边界，它们就是相邻节点。任务目标是让相邻区域的颜色不同，因此本质上是一个带有空间外观的图着色问题。

项目目前包含两个实验：

- **实验 1**：从空白地图开始，为所有区域上色，使相邻区域颜色不同。
- **实验 2**：从一个已经填色但存在冲突的地图开始，通过改色把整个地图修复为无冲突状态。

除了网页实验本身，仓库还包含：

- 行为数据导出逻辑
- 地图与邻接关系的离线重建代码
- 被试选择过程的可视化脚本
- 对实验 1 行为进行建模和拟合的代码


## 2. 项目的核心科学问题

这个项目想回答的问题并不是“人能不能完成图着色”，而是更细一点的问题：

- 人是如何决定下一步去哪个区域的？
- 人是倾向于在局部连续推进，还是会跳到较远的地方？
- 人会不会优先处理那些约束更强、已经被周围已填色区域包围的部分？
- 在冲突修复任务里，人是否会为了全局可解而提前修改当前并不冲突的区域？

因此，这个项目同时关心三类过程：

- **局部约束处理**：某个区域周围已经有多少已着色邻居
- **全局空间组织**：下一步是留在附近，还是跳去别的地方
- **规划（planning）**：当前一步是否是在为之后的几步做准备


## 3. 仓库结构

仓库里最重要的目录如下：

- [`experiment1/`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1)：当前部署使用的前端项目和 API 路由
- [`experiment1/src/Experiment1Game.tsx`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/src/Experiment1Game.tsx)：实验 1 的前端逻辑
- [`experiment1/src/experiment2/Experiment2Game.tsx`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/src/experiment2/Experiment2Game.tsx)：实验 2 的前端逻辑
- [`experiment1/public/experiment2/rounds.json`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/public/experiment2/rounds.json)：实验 2 的静态题目材料
- [`experiment2/`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment2)：实验 2 材料生成脚本和说明
- [`data/`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/data)：被试导出的 CSV 数据
- [`model code/fitting/`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting)：拟合脚本、分析脚本、notebook
- [`CSV_FORMAT.md`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/CSV_FORMAT.md)：实验 1 数据格式说明


## 4. 实验 1：从空白地图开始涂色

### 4.1 任务设计

在实验 1 中，被试看见的是一张尚未上色的地图，以及 4 种可选颜色。目标是给所有区域着色，并满足：

- 任意两个相邻区域颜色不同

实验流程包括：

- 2 轮练习，每轮 10 个区域
- 10 轮正式实验，区域数量逐步增加

正式实验 10 轮的区域数依次是：

- 20
- 23
- 26
- 28
- 31
- 34
- 37
- 39
- 42
- 45

所有被试看到的是同一套正式地图，因为这些地图是由固定随机种子生成的。

### 4.2 被试在网页上能做什么

被试可以：

- 选择 4 种颜色中的一种
- 点击某个区域，把它涂成所选颜色
- 使用橡皮擦，把一个区域的颜色擦掉

如果两个相邻区域颜色相同，界面会将其标记为冲突。只有当：

- 所有区域都已着色
- 且没有任何相邻同色

该轮才算完成。

### 4.3 实验 1 的输出数据

实验 1 导出的 CSV 包含两部分：

- `[Actions]`：记录被试做过的所有动作
- `[Adjacency]`：记录每轮地图中哪些区域彼此相邻

具体字段说明见：

- [`CSV_FORMAT.md`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/CSV_FORMAT.md)

需要特别注意的是，原始导出数据记录的是**所有动作**，包括：

- 游戏开始
- 合法上色
- 非法尝试后的后续动作
- 橡皮擦动作

而后续建模并不会直接把“所有点击”都当成等价 trial，而是先回放状态，再提取适合拟合的那部分有效步骤。


## 5. 实验 2：从冲突地图开始修复

### 5.1 任务设计

实验 2 的地图在开始时已经全部上色，但颜色配置是有问题的。被试一开始看到的是：

- 45 个 region 都已经有颜色
- 其中有一些相邻 region 颜色相同，因而存在冲突

被试的任务是通过修改颜色，把整张地图修复到无冲突状态。

### 5.2 实验 2 想操纵的认知过程

实验 2 的目标不是简单地让被试看见一个冲突，然后改掉冲突中的某一块。更核心的设计意图是：

- 让被试不能只靠“哪里冲突就改哪里”完成任务
- 而需要考虑更远的连锁影响
- 甚至需要修改一些当前并不冲突的区域，才能让整体恢复为合法状态

因此，实验 2 想强调的是：

- repair（修复）
- global coordination（全局协调）
- planning（规划）

### 5.3 当前实验 2 的实现状态

实验 2 当前已经在网页端实现，并且使用预生成的静态材料。网页端直接读取：

- [`rounds.json`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/public/experiment2/rounds.json)

这份材料由脚本生成：

- [`generate_rounds_json.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment2/generate_rounds_json.py)

目前的生成器会筛掉过于简单的题目，并加入更强的 planning 约束。

### 5.4 当前实验 2 生成器的关键约束

当前版本至少要求：

- 不能通过一次改色就修好整张图
- 在冲突相关区域及其邻域中，3 步以内不能修好
- 更重要的是：**不存在任何一种合法解，只修改初始冲突区域就能完成修复**

最后这一条是硬约束。也就是说，当前保留下来的题目都要求：

- 至少有一个初始时并不冲突的 region 必须被修改

这比之前“我选中的参考解里改到了非冲突 region”更强，因为它排除了“虽然参考解需要 planning，但其实还有更简单局部解”的情况。

不过仍然要谨慎理解：

- 这已经明显接近 planning 任务
- 但它仍然不是对“所有可能最优解结构”的完整理论刻画

更准确的说法是：

- 当前生成器实现了一个很强的、面向 planning 的实验材料筛选标准

### 5.5 实验 2 的辅助检查文件

为了人工检查实验 2 题目，目前有两个重要文件：

- [`experiment2_rounds_info.txt`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment2_rounds_info.txt)
- [`experiment1_rounds_info.txt`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1_rounds_info.txt)

其中 `experiment2_rounds_info.txt` 包含：

- 每轮每个区域包含哪些网格单元
- 邻接关系
- 每个区域的初始颜色和目标合法颜色
- 初始冲突边


## 6. 网页实验的组织方式

### 6.1 网页入口

网页端使用的是一个统一的前端项目，入口文件是：

- [`experiment1/src/App.tsx`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/src/App.tsx)

被试进入网页后，先输入编号，然后选择进入：

- 实验 1
- 实验 2

对应的路径是：

- `/`
- `/experiment1`
- `/experiment2`

### 6.2 上传逻辑

网页前端支持把导出的 CSV 上传到后端接口：

- [`experiment1/api/upload.ts`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/api/upload.ts)
- [`experiment1/api/list.ts`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/api/list.ts)

需要注意的是，这里并不是数据库，而是：

- **Vercel Blob 对象存储**

所以：

- 如果部署在支持这些 API 的平台上，自动上传可以工作
- 如果只是部署到 GitHub Pages 这样的纯静态平台，网页本身能打开，但这些 API 不会工作


## 7. 为什么可以离线重建地图

项目分析代码有一个非常重要的特点：地图不是通过截图保存的，而是通过固定随机种子生成的。

这意味着分析代码只要知道：

- 第几轮
- 该轮的固定 seed

就可以在离线环境里重新生成出与实验网页完全一致的地图、region 结构和邻接关系。

这部分逻辑主要在：

- [`fit_softmax.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/fit_softmax.py)

正因为可以离线重建地图，分析代码才能进一步恢复：

- 每一步当时有哪些 region 可选
- 每个候选 region 有多少已上色邻居
- 当前选择与上一步选择之间的图距离
- 各 region 的几何中心和欧氏距离


## 8. 什么叫做“用于建模的步骤”

这个项目里，“被试做了多少步”和“模型拿多少步来拟合”并不是完全一样的。

原始导出数据记录了所有动作，但用于拟合时，通常只保留那些更有意义的“有效构造步骤”。在实验 1 中，这通常要求：

- 该动作是上色，而不是橡皮擦
- 被点击的 region 之前是未着色状态
- 该颜色在当下是合法的

因此，后面模型里所谓的 step、trial，更准确地说是：

- **有效构造选择**

而不是：

- 被试所有的鼠标点击

这一点很重要，因为它决定了模型解释的是“有效决策策略”，而不是所有表层操作噪声。


## 9. 描述性分析部分

在正式拟合模型之前，项目先对行为做了一些描述性统计。

### 9.1 全局策略统计

- [`global_stats.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/global_stats.py)

这个脚本主要分析：

- 每轮第一个选择落在地图哪里
- 连续两步之间的空间距离有多大
- 连续两步是不是相邻 region

典型指标包括：

- 起点到地图中心的距离
- 连续步骤之间质心距离的均值
- 相邻转移比例

### 9.2 图结构分析

- [`graph_analysis.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/graph_analysis.py)

这个脚本分析的是地图图结构本身，包括：

- 节点数和边数
- 度分布
- 图距离
- 聚类结构
- 着色相关指标

这部分的作用是帮助理解：

- 为什么某些 round 客观上更难
- 被试策略是否会受到图结构差异影响

### 9.3 高复杂度轮次的可视化

- [`visualize_high_region_choices.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/visualize_high_region_choices.py)

这个脚本会把大图（尤其是 region 数量大于 35 的轮次）中被试的选择过程画出来，便于人工检查策略。


## 10. 实验 1 的建模思路

实验 1 的核心建模问题是：

> 在某一步里，被试为什么会选择这个 region，而不是其他 region？

当前建模主要从两个层面来理解这个过程：

- **移动尺度**：下一步离上一步有多远
- **局部目标选择**：在这一尺度下，为何选中某个具体 region

也就是说，项目不是把“下一步选哪个 region”看成一个完全扁平的单层选择，而是把它拆成：

- 先决定“走多远”
- 再决定“在这个距离上挑哪一个”


## 11. Softmax 区域选择模型

- [`fit_softmax.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/fit_softmax.py)

这是一个相对基础的选择模型。对每个尚未着色的 region，计算一个 utility：

- 已着色邻居越多，utility 越高
- 离地图中心越近，utility 越高（等价于负距离）

然后对所有候选 region 做 softmax，得到该步选择概率。

这个模型的意义在于回答：

- 被试是否偏好处理约束更强的区域
- 被试是否偏好先从中心区域开始

它可以作为一个直观、低复杂度的基线模型。


## 12. 步长模型

- [`step_length_models.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/step_length_models.py)

这一部分分析的是：从上一个被选 region 到当前被选 region，人的“步长”分布是什么样的。

这里考虑了几类候选模型：

- random
- neighbor
- jump
- levy

步长可以用两种方式定义：

- **graph distance**：在 region 邻接图上的最短路长度
- **Euclidean distance**：两个 region 质心之间的欧氏距离

项目后续更强调 graph distance，因为这个任务本质上是图约束任务，而不是连续平面上的移动任务。


## 13. Levy + neighbor 联合模型

当前最重要的模型实现是：

- [`fit_levy_neighbor_joint.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/fit_levy_neighbor_joint.py)

这个模型把每一步有效选择拆成三部分：

1. 选择步长 `L`
2. 在这个步长上选择具体 region
3. 为该 region 选择颜色

形式上写成：

`p(action | state, b, beta) = p(L | b) * p(piece | state, L, beta) * p(color | piece, state)`

其中：

- `b`：控制步长分布的 Levy 指数
- `beta`：控制在固定步长下，对“已着色邻居数更多”的 region 的偏好强度

### 13.1 这个分解为什么合理

这个分解反映了一个自然的认知想法：

- 一个人可能先大致决定“下一步留在附近还是跳远一点”
- 然后在那个尺度上，再根据局部约束挑一个具体目标

所以这个模型对应的是：

- **全局移动尺度决策**
- **局部目标选择决策**

两个层次的结合。

### 13.2 颜色项怎么处理

颜色项没有直接把 4 个颜色标签当成完全独立的原始类别，而是使用 canonical effective colors 的思路，避免把本质上等价的颜色重命名差异误当成行为差异。


## 14. 这里的“trial-by-trial”到底是什么意思

这个项目里经常会说“逐 trial 拟合”，但严格来说，当前实验 1 的拟合并不是“一轮一个 trial”，而是：

- **逐 step**
- **逐有效选择**
- **逐被试**

也就是说：

- 每个被试有一串有效选择步骤
- 被试的总 likelihood 是这些步骤 likelihood 的乘积
- 参数是对每个被试单独估计的

因此更准确的表述应该是：

- **step-by-step effective choice fitting**

而不是：

- 每轮一个观测
- 或者每次鼠标点击都直接进模型


## 15. 为什么 Levy 部分可能会拟合得比较好

这个项目里，Levy 型步长分布之所以可能表现好，直观上是因为人的行为常常呈现：

- 大量短步
- 少量长跳

这在地图涂色任务中很自然。被试可能会：

- 在某一块局部区域连续推进几步
- 但偶尔又跳到另一块更值得处理的区域

而 Levy 型分布正适合描述这种“以短步为主，但带有长尾跳转”的模式。

需要强调的是，这并不意味着：

- 人的大脑真的显式实现了 Levy process

更稳妥的说法是：

- Levy-like 模型很好地描述了人在这个图结构任务中的多尺度搜索行为


## 16. Notebooks 和结果分析

仓库里有多份 notebook 用于结果整理和可视化，例如：

- [`levy_neighbor_joint_fit.ipynb`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/levy_neighbor_joint_fit.ipynb)
- [`model_comparison.ipynb`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/model_comparison.ipynb)
- [`step_length_model_comparison.ipynb`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/step_length_model_comparison.ipynb)
- [`graph_global_strategy_analysis.ipynb`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/graph_global_strategy_analysis.ipynb)

这些 notebook 通常做的事情包括：

- 构建用于拟合的 step 表
- 对每个被试分别拟合模型
- 用 AIC/BIC 或 log-likelihood 比较模型
- 可视化参数分布和策略差异


## 17. 当前项目的局限与后续方向

### 17.1 实验 1 和实验 2 的成熟度不同

实验 1 是当前项目中更成熟的部分：

- 任务设计稳定
- 数据格式清楚
- 分析流程完整
- 拟合代码已经成型

实验 2 是更近一步的扩展：

- 网页任务已经实现
- 静态材料生成器已经建立
- planning 约束在逐步加强
- 但还没有像实验 1 那样形成完整、稳定的正式建模框架

### 17.2 上传后端不是数据库

当前上传逻辑用的是 Vercel Blob，而不是数据库。因此：

- 自动上传能力依赖部署平台
- 静态托管环境无法运行上传 API

### 17.3 当前建模主要针对实验 1

目前仓库中的正式模型基本都围绕实验 1 展开。实验 2 虽然网页与材料已经有了，但尚未形成对应的 repair/planning 模型分析管线。


## 18. 给新合作者的推荐阅读顺序

如果一个新合作者完全不了解这个项目，推荐按下面的顺序阅读：

1. [`PROJECT_OVERVIEW.md`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/PROJECT_OVERVIEW.md)
2. [`CSV_FORMAT.md`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/CSV_FORMAT.md)
3. [`experiment1/src/Experiment1Game.tsx`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/src/Experiment1Game.tsx)
4. [`experiment1/src/experiment2/Experiment2Game.tsx`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/experiment1/src/experiment2/Experiment2Game.tsx)
5. [`fit_softmax.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/fit_softmax.py)
6. [`step_length_models.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/step_length_models.py)
7. [`fit_levy_neighbor_joint.py`](/Users/akira1/Documents/26春/codes/grid-puzzle-experiment-main/model%20code/fitting/fit_levy_neighbor_joint.py)
8. 最后再看各个 notebook 和图表输出


## 19. 一句话总结

这个项目通过两个网页图着色实验研究人类在空间约束问题中的逐步选择行为，并用 step-level 概率模型来解释实验 1 中“下一步选哪里”的决策机制，同时逐步把实验 2 发展成一个更强调 planning 的冲突修复任务。
