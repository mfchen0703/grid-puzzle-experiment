# 实验2 Agent 参数化方案

## 1. 目的

这份文档整理当前实验2搜索 agent 的结构，并说明如何在此基础上引入类似 `fourinarow` 的参数，用于后续拟合被试行为。

当前目标不是立刻完成拟合代码，而是先明确：

- 现有 agent 的哪些部分是固定机制
- 哪些部分可以参数化
- `pruning_thresh`、`gamma`、`lapse_rate` 在实验2中分别可以对应什么


## 2. 当前 agent 的固定框架

当前实验2 agent 现在可以概括为：

1. 节点表示整张地图的当前颜色状态
2. 从当前冲突 region 出发，按图距离逐层生成候选改色动作
3. 通过 heuristic valuation 给每个候选动作赋值
4. 根据 heuristic value 和 `pruning_thresh` 剪枝
5. 在有限深度内维护并扩展搜索树
6. 对子树信息做回传
7. 根节点根据 `best_child` 选择当前执行动作

当前默认实现的关键设置是：

- `max_depth = 4`
- `n_iterations = 20`
- deterministic tie-break
- 当前默认动作启发式分数：
  $$
  H(a) = -\,\mathrm{conflicts\_after}(a) + \mathbf{1}\!\left[\mathrm{next\_depth}(a)=0\right]
  $$

这套机制已经能解开当前全部 10 个 round。


## 3. 为什么需要参数化

如果要用这个 agent 去拟合实验2被试行为，仅有“能解出题目”还不够。

因为被试之间很可能在以下维度上存在差异：

- 是否只保留最优动作，还是会同时考虑一批接近最优的动作
- 每一步愿意投入多少内部搜索预算
- 是否会偶尔偏离当前最优动作，表现出试错或 lapses

这些差异不能只靠一个固定 agent 表示，因此需要引入参数。


## 4. 建议引入的三个参数

### 4.1 `pruning_thresh`

#### 作用

控制候选动作剪枝的宽度。

当前代码已经实现为连续阈值版本：

- 对每个合法动作先计算一个数值启发式分数 $H(a)$
- 再保留所有与最优动作分差不超过阈值的动作

#### 在实验2中的建议定义

记每个合法动作 `a` 的启发式分数为 `H(a)`。

则保留集合定义为：

$$
\mathcal{A}_{\mathrm{keep}}(s_t)
=
\left\{
a \in \mathcal{A}_{\mathrm{legal}}(s_t)
:
H^*(s_t) - H(a) \le \tau_{\mathrm{prune}}
\right\}
$$

其中：

- $H^*(s_t) = \max_{a \in \mathcal{A}_{\mathrm{legal}}(s_t)} H(a)$
- $\tau_{\mathrm{prune}}$ 就是 `pruning_thresh`

#### 认知解释

- `pruning_thresh` 小：被试只考虑极少数最优动作，搜索更窄、更贪心
- `pruning_thresh` 大：被试会保留更多接近最优的动作，搜索更宽


### 4.2 `gamma`

#### 作用

控制每一步决策前的内部搜索预算。

当前实现里这个预算是固定的：

- `n_iterations = 20`

但如果想拟合被试，搜索预算最好不要固定为常数，而应该允许被试之间不同。

#### 在实验2中的建议定义

当前代码里，`gamma` 已实现成“每一步的 tree-search iteration budget”参数。

例如可定义：

$$
N_t = \left\lfloor \frac{1}{\gamma_i} \right\rfloor + 1
$$

其中：

- $N_t$ 是第 $t$ 步决策前执行多少轮 `select -> expand -> backpropagate`
- $\gamma_i$ 是被试 $i$ 的搜索预算参数

更接近 `fourinarow` 的随机版本也可行，但当前代码还没有实现。

#### 认知解释

- `gamma` 小：内部搜索预算更大，更愿意继续想
- `gamma` 大：预算更小，更快停止并执行当前主观最佳动作


### 4.3 `lapse_rate`

#### 作用

控制 agent 偶尔偏离当前最优动作的概率。

这对应被试中的：

- 误点击
- 注意力波动
- 临时试错
- 不完全按当前最优分支行动

#### 在实验2中的建议定义

设当前根节点下已保留下来的 child 动作为 $\mathcal{A}_{\mathrm{child}}(s_t)$。

可以定义：

$$
p(a_t \mid s_t)
=
(1-\lambda_i)\, p_{\mathrm{tree}}(a_t \mid s_t)
+
\lambda_i\, p_{\mathrm{lapse}}(a_t \mid s_t)
$$

其中：

- $\lambda_i$ 是被试 $i$ 的 `lapse_rate`
- $p_{\mathrm{tree}}$ 是当前树搜索给出的动作分布
- $p_{\mathrm{lapse}}$ 是随机动作分布

当前代码里，lapse 是在根节点最终执行动作时触发的，最简单随机分布可写成：

$$
p_{\mathrm{lapse}}(a_t \mid s_t)
=
\frac{1}{|\mathcal{A}_{\mathrm{child}}(s_t)|}
$$

#### 认知解释

- `lapse_rate` 小：被试更稳定地执行当前最优搜索结果
- `lapse_rate` 大：被试更容易偏离、试错、产生噪声


## 5. 启发式分数的定义位置

为了使用 `pruning_thresh`，需要先定义动作启发式分数 `H(a)`。

当前代码已经把 heuristic 抽象成一组 action feature 与权重。

当前实现位置：

- `action_heuristic_features(...)`
- `action_heuristic_score(...)`
- `prune_tree_actions(...)`

当前核心 feature 包括：

- `repair`: 标准化后的冲突减少比例越高越好
- `opportunity`: 下一步冲突区本身是否可改
- `region_preserve`: 是否直接处理当前冲突 region，不包含局部冲突减少奖励
- `color_preserve`: 是否先修改冲突邻域环境以释放颜色空间

代码仍会计算 `spatial`、`neighbor` 和局部 `color` 作为诊断 feature，但它们不再进入当前核心模型的权重集合。

当前默认权重为：

$$
\begin{aligned}
w_{\mathrm{repair}} = 2,\quad
w_{\mathrm{opp}} = 1,\quad
w_{\mathrm{region}} = 0,\quad
w_{\mathrm{color}} = 0
\end{aligned}
$$

其中 $w_{\mathrm{region}}$ 对应代码里的 `region_preserve`，$w_{\mathrm{color}}$ 对应代码里的 `color_preserve`。

因此默认版本的启发式排序为：

1. 改完后冲突边更少
2. 若下一步冲突区本身可改，则额外加分

当前代码已经把它写成一个数值分数：

$$
H(a)
=
\frac{
\mathrm{conflicts\_before}
-
\mathrm{conflicts\_after}(a)
}{
\max(1,\mathrm{conflicts\_before})
}
+
\mathbf{1}\!\left[\mathrm{next\_depth}(a)=0\right]
$$

也就是说，当前第一版里：

- 启发式 feature 接口已经统一
- 默认只打开 `repair` 和 `opportunity`
- 拟合参数先集中在 `pruning_thresh`、`gamma`、`lapse_rate`

当前与实验1建模对齐的 heuristic valuation 层参数主要是：

- `w_region_preserve`
- `w_color_preserve`

`pruning_thresh`、`gamma`、`lapse_rate` 则属于 planning/search 层的参数。

其中：

- `w_region_preserve` 对应区域保持 / 持续修改当前冲突 region 的倾向。
- `w_color_preserve` 对应颜色保持 / 环境颜色重排倾向。

当前代码中，`color` 和 `color_preserve` 相关 feature 已做归一化：

- `color` 使用局部非邻接同色匹配比例，而不是原始计数。
- `color_preserve` 由四个 `0-1` 或归一化子项平均组成，整体大致限制在 `0-1`。

这样做是为了避免颜色配置项因为计数尺度过大而压过目标推进项。

当前代码进一步加入了两个可选入口，让同一个 heuristic score 不只影响剪枝：

- `heuristic_eval_weight`: 控制 $H(a)$ 是否进入叶节点 / 子树评分。
- `heuristic_frontier_weight`: 控制 $H(a)$ 是否影响 frontier 中优先扩展哪个叶子。

叶节点评分现在可以写成：

$$
\begin{aligned}
V_{\mathrm{node}}(s,a)
=\;&
\Big(
\mathrm{solved}(s),
-\,\mathrm{conflicts}(s),
\mathbf{1}[\mathrm{next\_depth}(s)=0],
-\,\mathrm{next\_depth}(s),
\eta_{\mathrm{eval}} H(a),
-\,d_{\mathrm{best}}
\Big)
\end{aligned}
$$

其中 $\eta_{\mathrm{eval}}$ 对应 `heuristic_eval_weight`。代码中这个评分是按 tuple 字典序比较的，所以前四项仍然优先表示“是否解开、冲突数是否更少、是否打开直接修复机会、下一步距离是否更浅”；启发式项只在这些核心目标相同或接近时进一步区分分支。

frontier selection 现在可以写成：

$$
\begin{aligned}
K_{\mathrm{frontier}}(s,a)
=\;&
\Big(
\mathrm{self\_score}(s),
\eta_{\mathrm{frontier}} H(a),
-\,\mathrm{depth}(s)
\Big)
\end{aligned}
$$

其中 $\eta_{\mathrm{frontier}}$ 对应 `heuristic_frontier_weight`。它控制搜索预算优先投向哪些叶子，而不是直接决定最终动作。

默认情况下：

$$
\begin{aligned}
\eta_{\mathrm{eval}} = 0,\quad
\eta_{\mathrm{frontier}} = 0
\end{aligned}
$$

因此默认版本仍然保持旧行为；只有显式设置这两个参数时，heuristic score 才会进入 leaf evaluation / frontier priority。


## 6. 当前适合的最小参数化版本

如果要尽快做出一版可拟合模型，建议先用下面这个最小版本：

### 固定机制

- 状态空间：整图颜色状态
- 搜索深度：固定 `max_depth = 4`
- frontier selection：沿用当前实现
- subtree backprop：沿用当前实现
- 启发式分数：沿用当前版本

### 自由参数

1. `pruning_thresh_i`
2. `gamma_i`
3. `lapse_rate_i`
4. `heuristic_eval_weight_i`
5. `heuristic_frontier_weight_i`

其中：

- `pruning_thresh_i` 控制动作保留宽度
- `gamma_i` 控制内部搜索预算
- `lapse_rate_i` 控制随机偏离程度
- `heuristic_eval_weight_i` 控制 heuristic score 对子树评分的影响
- `heuristic_frontier_weight_i` 控制 heuristic score 对叶节点扩展顺序的影响

前三个参数已经足以刻画：

- 搜索宽度
- 搜索深度/预算
- 执行噪声

后两个参数进一步刻画 heuristic valuation 如何进入 planning search 内部。


## 7. 一个可能的概率模型写法

对被试 $i$ 在状态 $s_{it}$ 的动作 $a_{it}$，可写：

$$
p(a_{it} \mid s_{it}, \theta_i)
=
(1-\lambda_i)\, p_{\mathrm{tree}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
+
\lambda_i\, p_{\mathrm{lapse}}(a_{it} \mid s_{it})
$$

其中：

- $\theta_i = (\tau_i, \gamma_i, \lambda_i)$
- $\tau_i$ 对应 `pruning_thresh`
- $\gamma_i$ 控制搜索预算
- $\lambda_i$ 对应 `lapse_rate`

如果第一版只想做最大似然估计，可以把：

- `p_tree`

定义成当前 tree search 生成的离散动作分布；
或者更简单地，把 tree search 的最终选中动作视为概率 1 的主动作，再把其他保留动作按极小概率平滑。


## 8. likelihood 方程

下面给出一版更正式的 likelihood 写法。

### 8.1 记号

对被试 $i$、第 $t$ 步：

- $s_{it}$：当前整图状态
- $a_{it}$：被试真实执行的改色动作
- $\mathcal{A}_{\mathrm{legal}}(s_{it})$：状态 $s_{it}$ 下全部合法动作集合
- $\mathcal{A}_{\mathrm{keep}}(s_{it}; \tau_i)$：在 `pruning_thresh = \tau_i` 下保留下来的候选动作集合
- $\theta_i = (\tau_i, \gamma_i, \lambda_i)$

其中：

- $\tau_i$：`pruning_thresh`
- $\gamma_i$：搜索预算参数
- $\lambda_i$：`lapse_rate`


### 8.2 剪枝后候选集合

设每个合法动作 $a$ 的启发式分数为 $H(a; s_{it})$，则保留集合定义为：

$$
\begin{aligned}
\mathcal{A}_{\mathrm{keep}}(s_{it}; \tau_i)
=\;&
\left\{
a \in \mathcal{A}_{\mathrm{legal}}(s_{it})
:
H^*(s_{it}) - H(a; s_{it}) \le \tau_i
\right\}
\end{aligned}
$$

其中

$$
\begin{aligned}
H^*(s_{it})
=\;&
\max_{a \in \mathcal{A}_{\mathrm{legal}}(s_{it})} H(a; s_{it})
\end{aligned}
$$


### 8.3 当前代码中的 tree policy 概率

在当前代码里，tree search 可以视为：

- 在参数 $(\tau_i, \gamma_i)$ 下
- 对当前状态 $s_{it}$ 进行一次有限预算搜索
- 返回一个候选动作集合 $\mathcal{A}_{\mathrm{best}}(s_{it}; \tau_i, \gamma_i)$

其中 $\mathcal{A}_{\mathrm{best}}$ 表示当前 root 下、`subtree_score` 并列最优的 child 动作集合。

第一版最简单的 `p_tree` 可以定义成该集合上的均匀分布：

$$
\begin{aligned}
p_{\mathrm{tree}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
=\;&
\begin{cases}
\dfrac{1}{\left| \mathcal{A}_{\mathrm{best}}(s_{it}; \tau_i, \gamma_i) \right|}
&
\text{if } a_{it} \in \mathcal{A}_{\mathrm{best}}(s_{it}; \tau_i, \gamma_i)
\\[8pt]
0
&
\text{otherwise}
\end{cases}
\end{aligned}
$$

如果后续担心出现零概率，可以加入一个很小的平滑常数 $\varepsilon$，改成：

$$
\begin{aligned}
p_{\mathrm{tree}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
=\;&
\frac{
\mathbf{1}\!\left[a_{it} \in \mathcal{A}_{\mathrm{best}}(s_{it}; \tau_i, \gamma_i)\right]
\;+\; \varepsilon
}{
\left| \mathcal{A}_{\mathrm{best}}(s_{it}; \tau_i, \gamma_i) \right|
\;+\;
\varepsilon \cdot \left| \mathcal{A}_{\mathrm{legal}}(s_{it}) \right|
}
\end{aligned}
$$


### 8.4 当前代码中的 lapse policy 概率

当前代码里，lapse policy 不是在全部合法动作上随机，而是在当前 root 已保留的 child 动作上均匀随机：

$$
\begin{aligned}
p_{\mathrm{lapse}}(a_{it} \mid s_{it})
=\;&
\frac{1}{\left| \mathcal{A}_{\mathrm{child}}(s_{it}) \right|}
\qquad
\text{for } a_{it} \in \mathcal{A}_{\mathrm{child}}(s_{it})
\end{aligned}
$$


### 8.5 单步动作概率

于是单步动作概率为：

$$
\begin{aligned}
p(a_{it} \mid s_{it}, \theta_i)
=\;&
(1-\lambda_i)\,
p_{\mathrm{tree}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
\\
&+
\lambda_i\,
p_{\mathrm{lapse}}(a_{it} \mid s_{it})
\end{aligned}
$$


### 8.6 单个被试的 likelihood

对被试 $i$ 的全部动作序列，其 likelihood 为：

$$
\begin{aligned}
\mathcal{L}_i(\tau_i, \gamma_i, \lambda_i)
=\;&
\prod_t
p(a_{it} \mid s_{it}, \tau_i, \gamma_i, \lambda_i)
\end{aligned}
$$

对应 log-likelihood 为：

$$
\begin{aligned}
\log \mathcal{L}_i(\tau_i, \gamma_i, \lambda_i)
=\;&
\sum_t
\log p(a_{it} \mid s_{it}, \tau_i, \gamma_i, \lambda_i)
\end{aligned}
$$

最大似然估计为：

$$
\begin{aligned}
(\hat{\tau}_i, \hat{\gamma}_i, \hat{\lambda}_i)
=\;&
\arg\max_{\tau_i, \gamma_i, \lambda_i}
\log \mathcal{L}_i(\tau_i, \gamma_i, \lambda_i)
\end{aligned}
$$


### 8.7 当前代码中的 `gamma` 映射

当前代码采用确定性搜索预算：

$$
\begin{aligned}
N_i
=\;&
\left\lfloor \frac{1}{\gamma_i} \right\rfloor + 1
\end{aligned}
$$

然后把 tree policy 写成：

$$
\begin{aligned}
p_{\mathrm{tree}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
=\;&
p_{\mathrm{tree}}(a_{it} \mid s_{it}; \tau_i, N_i)
\end{aligned}
$$

也就是说，$\gamma_i$ 并不直接出现在 softmax 里，而是通过控制内部搜索预算，间接影响动作分布。


### 8.8 当前实现版的单被试 likelihood

把当前实现合起来，可以写成：

$$
\begin{aligned}
\mathcal{L}_i(\tau_i, \gamma_i, \lambda_i)
=\;&
\prod_t
\Big[
(1-\lambda_i)\,
p_{\mathrm{tree}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
\\
&\qquad\qquad
+
\lambda_i\,
p_{\mathrm{lapse}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
\Big]
\end{aligned}
$$

其中：

- $\tau_i$ 通过动作剪枝宽度影响 $\mathcal{A}_{\mathrm{keep}}$ 与最终 root children
- $\gamma_i$ 通过搜索轮数 $N_i$ 影响 tree search 的展开程度
- $\lambda_i$ 控制最终执行动作时偏离 `best_child` 的概率

对应 log-likelihood 为：

$$
\begin{aligned}
\log \mathcal{L}_i(\tau_i, \gamma_i, \lambda_i)
=\;&
\sum_t
\log
\Big[
(1-\lambda_i)\,
p_{\mathrm{tree}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
\\
&\qquad\qquad
+
\lambda_i\,
p_{\mathrm{lapse}}(a_{it} \mid s_{it}; \tau_i, \gamma_i)
\Big]
\end{aligned}
$$

最大似然估计为：

$$
\begin{aligned}
(\hat{\tau}_i, \hat{\gamma}_i, \hat{\lambda}_i)
=\;&
\arg\max_{\tau_i, \gamma_i, \lambda_i}
\log \mathcal{L}_i(\tau_i, \gamma_i, \lambda_i)
\end{aligned}
$$


## 9. 后续建议

建议按下面顺序推进：

1. 先固定现有搜索框架
2. 只加 `lapse_rate`
3. 再把硬剪枝改成 `pruning_thresh`
4. 最后把固定 `n_iterations` 参数化成 `gamma`

这样做的原因是：

- `lapse_rate` 最容易解释，也最稳定
- `pruning_thresh` 会直接改变搜索宽度
- `gamma` 会影响搜索预算与运行时间，改动最大


## 10. 当前结论

对实验2最自然的 `fourinarow` 风格参数化方案是：

- `pruning_thresh`：控制保留多少接近最优的动作
- `gamma`：控制每一步的内部搜索预算
- `lapse_rate`：控制偏离当前最优动作的概率

在现阶段，最推荐的做法不是完全照抄 `fourinarow`，而是：

- 保留实验2现有的冲突中心搜索结构
- 只把这三个参数映射到“剪枝宽度、搜索预算、执行噪声”这三个层面

这样既保留了任务结构，也便于后续拟合。
