# 实验2 Agent：启发式赋值 + 有限规划搜索

这份文档整理当前实验2 agent 的正式结构。当前版本的核心思想是：

1. 先用启发式函数给每个合法改色动作赋值。
2. 再用这些启发式分数引导有限 planning search。
3. 最终从搜索树根节点的 best child 中选择当前动作。

实验2和实验1的关系可以理解为：

- 实验1主要是直接用 heuristic valuation 选择动作。
- 实验2在 heuristic valuation 之上增加了 bounded planning。


## 1. 状态、动作和目标

### 1.1 状态

实验2中，一个状态是整张图所有 region 的当前颜色：

$$
\begin{aligned}
s_t
=\;&
(c_{1t}, c_{2t}, \ldots, c_{nt})
\end{aligned}
$$

其中 $c_{jt}$ 是第 $j$ 个 region 在第 $t$ 步的颜色。

### 1.2 动作

每一步动作是一个 region-color 修改：

$$
\begin{aligned}
a_t
=\;&
(r_t, c_t^{\mathrm{new}})
\end{aligned}
$$

也就是把 region $r_t$ 的当前颜色改成 $c_t^{\mathrm{new}}$。

### 1.3 冲突边

如果两个相邻 region 颜色相同，则这条邻接边是冲突边：

$$
\begin{aligned}
E_{\mathrm{conflict}}(s)
=\;&
\left\{
(u,v) \in E:
c_u = c_v
\right\}
\end{aligned}
$$

agent 的最终目标是让冲突边数量变为 0。


## 2. 候选动作生成

当前 agent 不会在所有 region 上盲目搜索，而是从当前冲突区域出发，按图距离逐层生成候选动作。

### 2.1 冲突区域

当前冲突区域集合定义为：

$$
\begin{aligned}
R_{\mathrm{conflict}}(s)
=\;&
\left\{
r:
\exists q,\ (r,q)\in E_{\mathrm{conflict}}(s)
\right\}
\end{aligned}
$$

### 2.2 搜索层级

对每个 region $r$，定义它到当前冲突区域的最短图距离：

$$
\begin{aligned}
d_{\mathrm{conflict}}(r,s)
=\;&
\min_{q \in R_{\mathrm{conflict}}(s)}
d(r,q)
\end{aligned}
$$

候选 region 按这个距离分层：

- depth 0：当前冲突 region 本身。
- depth 1：冲突 region 的邻居。
- depth 2：邻居的邻居。
- 以此类推。

一个候选动作必须满足：改色后该 region 本身不和任何邻居同色。也就是说，agent 只考虑局部合法改色动作。


## 3. 启发式赋值

每个合法动作 $a$ 会得到一个启发式分数：

$$
\begin{aligned}
H(a \mid s)
=\;&
w_{\mathrm{repair}} f_{\mathrm{repair}}(a,s)
+ w_{\mathrm{opp}} f_{\mathrm{opportunity}}(a,s) \\
&+
w_{\mathrm{region}} f_{\mathrm{region\_preserve}}(a,s)
+ w_{\mathrm{color}} f_{\mathrm{color\_preserve}}(a,s)
\end{aligned}
$$

也就是说，当前正式模型只保留四类权重：两个实验2任务相关权重，以及两个和实验1策略解释相对应的保持倾向权重。

当前代码中的入口是：

- `action_heuristic_features(...)`
- `action_heuristic_score(...)`
- `prune_tree_actions(...)`

### 3.1 默认目标推进项

默认 agent 使用两个目标推进 feature。

第一个是即时修复价值：

$$
\begin{aligned}
f_{\mathrm{repair}}(a,s)
=\;&
\frac{
\left|E_{\mathrm{conflict}}(s)\right|
-
\left|E_{\mathrm{conflict}}(T(s,a))\right|
}{
\max\left(1,\left|E_{\mathrm{conflict}}(s)\right|\right)
}
\end{aligned}
$$

其中 $T(s,a)$ 表示在状态 $s$ 执行动作 $a$ 后的新状态。

注意，`repair` 不是二值的“这一步是否完全解决冲突”，而是标准化后的冲突减少比例。它的值为正表示动作减少冲突，值为 0 表示冲突数不变，值为负表示动作制造了更多冲突。代码同时保留 `repair_raw = -|E_conflict(T(s,a))|` 和 `repair_delta` 作为诊断字段。

第二个是下一步机会：

$$
\begin{aligned}
f_{\mathrm{opportunity}}(a,s)
=\;&
\mathbf{1}\!\left[
\mathrm{next\_depth}(T(s,a)) = 0
\right]
\end{aligned}
$$

其中 `next_depth = 0` 表示动作后，下一步已经可以直接修改冲突 region 本身。

因此，`opportunity` 衡量的是“这一步是否为下一步直接处理冲突核心创造机会”，而不是这一步本身是否已经完全解决问题。

默认权重是：

$$
\begin{aligned}
w_{\mathrm{repair}} = 2,\quad
w_{\mathrm{opp}} = 1
\end{aligned}
$$

因此默认启发式为：

$$
\begin{aligned}
H_{\mathrm{default}}(a \mid s)
=\;&
\frac{
\left|E_{\mathrm{conflict}}(s)\right|
-
\left|E_{\mathrm{conflict}}(T(s,a))\right|
}{
\max\left(1,\left|E_{\mathrm{conflict}}(s)\right|\right)
}
+
\mathbf{1}\!\left[
\mathrm{next\_depth}(T(s,a)) = 0
\right]
\end{aligned}
$$

### 3.2 诊断性空间、邻居、局部颜色项

当前代码还会计算三类额外 feature，用于检查动作结构和之后可能的扩展。需要注意的是，按照当前核心模型，这三类 feature 只作为诊断字段保留，不进入正式的 $H(a\mid s)$。

#### `spatial`

`spatial` 衡量动作 region 离当前冲突区域有多近：

$$
\begin{aligned}
f_{\mathrm{spatial}}(a,s)
=\;&
-\,d_{\mathrm{conflict}}(r(a),s)
\end{aligned}
$$

因此：

- 修改当前冲突 region：`spatial = 0`
- 修改冲突 region 的邻居：`spatial = -1`
- 修改邻居的邻居：`spatial = -2`

它表示围绕当前冲突局部连续搜索，而不是实验1中“相对上一步 region 的连续性”。

#### `neighbor`

`neighbor` 衡量动作 region 和当前冲突核心的邻接强度：

$$
\begin{aligned}
f_{\mathrm{neighbor}}(a,s)
=\;&
\left|
N(r(a)) \cap R_{\mathrm{conflict}}(s)
\right|
\end{aligned}
$$

也就是说，如果一个候选 region 邻接多个当前冲突 region，它的 `neighbor` 分数更高。

认知解释是：优先修改更贴近当前冲突结构、可能同时影响多个冲突点的位置。它不是实验1中“已填色邻居最多”的 neighbor feature，因为实验2中所有 region 本来都有颜色。

#### `color`

`color` 衡量动作的新颜色是否复用局部非邻接 region 的已有颜色配置。

直觉是：

- 相邻 region 不能同色。
- 非相邻但位于同一个局部冲突环境中的 region 可以同色。

当前代码使用局部非邻接同色匹配比例，而不是原始计数：

$$
\begin{aligned}
f_{\mathrm{color}}(a,s)
=\;&
\frac{
\left|
\left\{
q \in L(r(a),s):
q \notin N(r(a)),
c_q = c^{\mathrm{new}}(a)
\right\}
\right|
}{
\max\left(1,\left|
\left\{
q \in L(r(a),s):
q \notin N(r(a))
\right\}
\right|\right)
}
\end{aligned}
$$

其中 $L(r(a),s)$ 表示动作 region 附近、和当前冲突区域相关的局部 region 集合。

认知解释是：`color` 表示局部颜色复用 / 颜色组织倾向，而不是“保持上一动作颜色”。

这些 feature 可以帮助理解 agent 的候选动作结构，但当前正式模型不再估计 `w_spatial`、`w_neighbor`、`w_color`。

### 3.3 区域保持策略

区域保持策略表示：优先继续处理当前冲突核心，也就是不断尝试直接修改冲突 region。

当前 feature 为：

$$
\begin{aligned}
f_{\mathrm{region\_preserve}}(a,s)
=\;&
\mathbf{1}\!\left[
r(a)\in R_{\mathrm{conflict}}(s)
\right]
\end{aligned}
$$

这个 feature 只表示“是否选择当前冲突 region 本身”，不再奖励局部冲突边减少。

认知解释：

- 高 `region_preserve`：更像“先解决眼前冲突”的策略。
- 这种策略可能会制造新冲突，但会优先推动当前冲突区域发生改变。
- 动作结果好不好由 `repair` 负责评价，`region_preserve` 只负责表示区域保持倾向。

### 3.4 颜色保持策略

颜色保持策略表示：先不直接修改当前冲突 region，而是修改其周围环境，让冲突 region 之后有更多合法颜色空间。

当前 feature 已做归一化，避免计数项尺度过大：

$$
\begin{aligned}
f_{\mathrm{color\_preserve}}(a,s)
=\;&
\frac{1}{4}\mathrm{EnvironmentRegion}(a,s)
+
\frac{1}{4}\Delta\mathrm{LegalConflictColors}_{\mathrm{norm}}(a,s) \\
&+
\frac{1}{4}\Delta\mathrm{SameColorBlockersRemoved}_{\mathrm{norm}}(a,s)
+
\frac{1}{4}\mathrm{LocalNonNeighborColorMatch}_{\mathrm{norm}}(a,s)
\end{aligned}
$$

其中：

$$
\begin{aligned}
\mathrm{EnvironmentRegion}(a,s)
=\;&
\mathbf{1}\!\left[
r(a)\notin R_{\mathrm{conflict}}(s)
\right]
\mathbf{1}\!\left[
d_{\mathrm{conflict}}(r(a),s) \le 2
\right]
\end{aligned}
$$

$$
\begin{aligned}
\Delta\mathrm{LegalConflictColors}(a,s)
=\;&
\sum_{q\in R_{\mathrm{conflict}}(s)}
\left|
C_{\mathrm{legal}}(q,T(s,a))
\right| \\
&-
\sum_{q\in R_{\mathrm{conflict}}(s)}
\left|
C_{\mathrm{legal}}(q,s)
\right|
\end{aligned}
$$

其他两个子项的含义是：

- `SameColorBlockersRemoved`: 动作是否移除了冲突 region 周围与其同色的 blocker。
- `LocalNonNeighborColorMatch`: 动作新颜色是否匹配局部非邻接 region 的已有颜色。

认知解释：

- 高 `color_preserve`：更像“先改环境，保留或释放冲突核心颜色”的策略。
- 它不是简单保持上一动作颜色，而是围绕当前图中的颜色配置做环境重排。


## 4. 启发式分数进入 planning 的三个位置

当前 agent 中，$H(a\mid s)$ 可以进入 planning 的三个位置。

### 4.1 剪枝

对当前状态 $s$ 的所有合法动作，先计算最优启发式分数：

$$
\begin{aligned}
H^*(s)
=\;&
\max_{a \in \mathcal{A}_{\mathrm{legal}}(s)}
H(a\mid s)
\end{aligned}
$$

然后保留所有与最优动作分差不超过阈值的动作：

$$
\begin{aligned}
\mathcal{A}_{\mathrm{keep}}(s;\tau)
=\;&
\left\{
a \in \mathcal{A}_{\mathrm{legal}}(s):
H^*(s)-H(a\mid s) \le \tau
\right\}
\end{aligned}
$$

其中 $\tau$ 对应 `pruning_thresh`。

解释：

- `pruning_thresh` 小：只保留极少数启发式最优动作，搜索更贪心。
- `pruning_thresh` 大：保留更多接近最优的动作，搜索更宽。

### 4.2 frontier priority

每次 tree-search iteration 会先选择一个 frontier leaf 继续扩展。

当前 frontier priority 为：

$$
\begin{aligned}
K_{\mathrm{frontier}}(s,a)
=\;&
\Big(
\mathrm{self\_score}(s),
\eta_{\mathrm{frontier}} H(a\mid s_{\mathrm{parent}}),
-\,\mathrm{depth}(s)
\Big)
\end{aligned}
$$

其中 $\eta_{\mathrm{frontier}}$ 对应 `heuristic_frontier_weight`。

解释：

- `heuristic_frontier_weight = 0`：frontier 扩展不额外受 heuristic score 影响。
- `heuristic_frontier_weight > 0`：搜索预算更容易投向 heuristic score 高的叶子。

### 4.3 leaf / subtree evaluation

每个搜索节点都有一个状态评分：

$$
\begin{aligned}
\mathrm{self\_score}(s)
=\;&
\Big(
\mathrm{solved}(s),
-\,|E_{\mathrm{conflict}}(s)|,
\mathbf{1}[\mathrm{next\_depth}(s)=0],
-\,\mathrm{next\_depth}(s)
\Big)
\end{aligned}
$$

加入 heuristic leaf evaluation 后，节点评分写成：

$$
\begin{aligned}
V_{\mathrm{node}}(s,a)
=\;&
\Big(
\mathrm{solved}(s),
-\,|E_{\mathrm{conflict}}(s)|,
\mathbf{1}[\mathrm{next\_depth}(s)=0],
-\,\mathrm{next\_depth}(s),
\eta_{\mathrm{eval}} H(a\mid s_{\mathrm{parent}}),
-\,d_{\mathrm{best}}
\Big)
\end{aligned}
$$

其中：

- $\eta_{\mathrm{eval}}$ 对应 `heuristic_eval_weight`。
- $d_{\mathrm{best}}$ 是当前节点到其子树内最佳状态的距离。
- 代码按 tuple 字典序比较这些评分。

解释：

- 前四项仍然优先表示任务目标推进。
- `heuristic_eval_weight` 只在核心状态指标相同或接近时进一步区分分支。
- 如果这个权重太大，启发式偏差可能被放大。


## 5. Tree search 流程

当前 tree search 可以概括为：

1. root 是当前真实状态。
2. 选择当前最值得扩展的 frontier leaf。
3. 在该 leaf 上生成合法候选动作。
4. 用 $H(a\mid s)$ 给动作打分。
5. 根据 `pruning_thresh` 剪枝。
6. 为保留下来的动作创建 child node。
7. 从 child node 向 root 回传 subtree score。
8. root 根据 `best_child` 选择当前执行动作。
9. 执行动作后，把被选中的 child 作为新的 root，保留其已展开子树。

这套结构参考了 `fourinarow` 中“持续保留并扩展搜索树、子树值回传、根节点选择 best child”的思想，但实验2的状态和动作定义是地图改色任务自己的定义。


## 6. 当前 planning 参数

### 6.1 `max_depth`

`max_depth` 控制单次 planning search 最深扩展到几层。

当前常用值：

```text
max_depth = 4
```

### 6.2 `n_iterations` 和 `gamma`

`n_iterations` 控制每一步真实动作前执行多少轮：

```text
select -> expand -> backpropagate
```

也可以用 `gamma` 间接控制搜索预算：

$$
\begin{aligned}
N_i
=\;&
\left\lfloor \frac{1}{\gamma_i} \right\rfloor + 1
\end{aligned}
$$

其中 $N_i$ 是被试 $i$ 的 tree-search iteration budget。

解释：

- `gamma` 小：搜索预算大。
- `gamma` 大：搜索预算小。

### 6.3 `pruning_thresh`

`pruning_thresh` 控制每个节点扩展时保留多少候选动作。

解释：

- 小 `pruning_thresh`：更贪心，容易过早丢弃合法分支。
- 大 `pruning_thresh`：更宽，搜索更完整，但也更容易受错误启发式拖累。

### 6.4 `lapse_rate`

`lapse_rate` 控制 agent 偶尔偏离当前 tree policy 的概率。

当前实现中：

$$
\begin{aligned}
p(a_t\mid s_t)
=\;&
(1-\lambda)p_{\mathrm{tree}}(a_t\mid s_t)
+
\lambda p_{\mathrm{lapse}}(a_t\mid s_t)
\end{aligned}
$$

其中 $\lambda$ 对应 `lapse_rate`。

当前 `p_lapse` 是在 root 已保留下来的 child actions 上均匀随机。

### 6.5 `heuristic_eval_weight`

控制 heuristic score 对 leaf / subtree evaluation 的影响。

解释：

- 值为 0：不影响 leaf evaluation。
- 值越大：启发式分数越容易影响回传后的 best child。

### 6.6 `heuristic_frontier_weight`

控制 heuristic score 对 frontier priority 的影响。

解释：

- 值为 0：不影响 frontier 扩展顺序。
- 值越大：搜索预算越集中到启发式分数高的叶子。


## 7. 动作概率和 likelihood

对被试 $i$ 在第 $t$ 步执行的动作 $a_{it}$，模型概率可以写成：

$$
\begin{aligned}
p(a_{it}\mid s_{it},\Theta_i)
=\;&
(1-\lambda_i)
p_{\mathrm{tree}}(a_{it}\mid s_{it},\Theta_i)
+
\lambda_i
p_{\mathrm{lapse}}(a_{it}\mid s_{it})
\end{aligned}
$$

其中：

$$
\begin{aligned}
\Theta_i
=\;&
(
\tau_i,
\gamma_i,
\lambda_i,
\eta_{\mathrm{eval},i},
\eta_{\mathrm{frontier},i},
\mathbf{w}_i
)
\end{aligned}
$$

这些参数分别表示：

- $\tau_i$: `pruning_thresh`
- $\gamma_i$: 搜索预算
- $\lambda_i$: `lapse_rate`
- $\eta_{\mathrm{eval},i}$: leaf evaluation 中 heuristic score 的权重
- $\eta_{\mathrm{frontier},i}$: frontier priority 中 heuristic score 的权重
- $\mathbf{w}_i$: 核心 heuristic feature weights，即 `repair`、`opportunity`、`region_preserve`、`color_preserve`

当前最简单的 tree policy 可以定义为 root 下并列 best children 上的均匀分布：

$$
\begin{aligned}
p_{\mathrm{tree}}(a\mid s,\Theta)
=\;&
\begin{cases}
\dfrac{1}{|\mathcal{A}_{\mathrm{best}}(s,\Theta)|},
&
a\in \mathcal{A}_{\mathrm{best}}(s,\Theta)
\\[8pt]
0,
&
\text{otherwise}
\end{cases}
\end{aligned}
$$

为了避免拟合时出现零概率，可以加入平滑：

$$
\begin{aligned}
p_{\mathrm{tree}}^{\epsilon}(a\mid s,\Theta)
=\;&
\frac{
\mathbf{1}[a\in \mathcal{A}_{\mathrm{best}}(s,\Theta)]
+\epsilon
}{
|\mathcal{A}_{\mathrm{best}}(s,\Theta)|
+\epsilon|\mathcal{A}_{\mathrm{legal}}(s)|
}
\end{aligned}
$$

被试 $i$ 的负对数似然为：

$$
\begin{aligned}
\mathrm{NLL}_i(\Theta_i)
=\;&
-
\sum_t
\log
p(a_{it}\mid s_{it},\Theta_i)
\end{aligned}
$$


## 8. 当前默认版本

当前默认设置为：

```text
max_depth = 4
n_iterations = 20
pruning_thresh = 0
lapse_rate = 0
heuristic_eval_weight = 0
heuristic_frontier_weight = 0
```

默认 heuristic weights 为：

$$
\begin{aligned}
w_{\mathrm{repair}} = 2,\quad
w_{\mathrm{opp}} = 1,\quad
w_{\mathrm{region}} = 0,\quad
w_{\mathrm{color}} = 0
\end{aligned}
$$

在代码参数中，$w_{\mathrm{region}}$ 对应 `region_preserve`，$w_{\mathrm{color}}$ 对应 `color_preserve`。`spatial`、`neighbor` 和局部 `color` 仍可作为诊断 feature 查看，但不进入当前核心模型的权重集合。

在当前 10 张实验2地图上，默认版本可以全部解开：

```text
steps = [7, 4, 5, 4, 6, 11, 5, 6, 7, 7]
solved = 10 / 10
```


## 9. 当前行为检查结论

### 9.1 默认 heuristic 进入 eval/frontier

在默认 $H(a)$ 下，测试了：

```text
heuristic_eval_weight = 0.5
heuristic_frontier_weight = 0.5
```

结果：

- 10 张地图全部解开。
- 每一步动作序列都和 default 完全相同。

解释：

- 默认 $H(a)$ 本身已经和核心 tree score 高度一致。
- 它主要奖励冲突更少和下一步可直接修复。
- 因此把它放入 leaf evaluation / frontier priority 后，没有提供新的区分信息。

### 9.2 加入颜色保持偏置

固定：

```text
heuristic_weights = {"color_preserve": 0.1}
pruning_thresh = 0.5
```

测试结果：

| 条件 | solved | 轨迹变化 |
|---|---:|---|
| 不加 eval/frontier | 10/10 | 无变化 |
| `heuristic_eval_weight = 0.05` | 10/10 | 多数 round 变化 |
| `heuristic_frontier_weight = 0.05` | 10/10 | 部分 round 变化 |
| 两者都为 `0.05` | 7/10 | 多数 round 变化，部分 round 卡住 |

解释：

- 单独让 $H(a)$ 进入 leaf evaluation 或 frontier priority，确实会改变搜索路径。
- 同时让 $H(a)$ 进入两个位置，会放大颜色保持偏置。
- 如果启发式偏置本身不完全可靠，过强的双重影响会让 agent 更容易卡住。

### 9.3 硬剪枝的问题

当：

```text
heuristic_weights = {"color_preserve": 0.1}
pruning_thresh = 0
```

即使设置：

```text
heuristic_eval_weight = 0.5
heuristic_frontier_weight = 0.5
```

也只能解开 5/10 张地图。

解释：

- 如果 pruning 阶段已经把关键分支剪掉，后续 leaf evaluation 和 frontier priority 无法恢复这些分支。
- 因此 `pruning_thresh` 是当前模型里非常关键的参数。


## 10. 当前模型解释

当前实验2 agent 可以被理解为三层机制：

1. **Heuristic valuation**
   给每个 region-color action 一个主观价值 $H(a\mid s)$。

2. **Search control**
   用 `pruning_thresh`、`heuristic_frontier_weight`、`heuristic_eval_weight` 决定哪些分支被保留、哪些叶子被扩展、哪些子树被认为更好。

3. **Execution noise**
   用 `lapse_rate` 表示偶尔偏离当前 tree policy 的行为。

这个结构的优点是：

- 可以保留目前能解题的 planning agent。
- 可以把实验1中的空间、邻居、颜色倾向迁移成实验2中的 heuristic weights。
- 可以区分“被试看重什么信息”和“被试如何使用这些信息进行搜索”。

当前需要谨慎的地方是：

- heuristic score 进入的位置越多，偏置越容易被放大。
- `pruning_thresh` 太小会过早剪掉关键分支。
- `color_preserve` 这类策略项需要和较宽 pruning 配合，否则容易形成循环或卡住。
