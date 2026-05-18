# Levy + 邻居联合模型

对被试 $i$ 的第 $t$ 步，记观测动作为 $a_{it} = (piece_{it}, color_{it})$，当前状态为 $s_{it}$。

每个被试有两个独立拟合参数：$b_i \in (1,3)$，$\beta_i \in \mathbb{R}$。

- $b_i$：控制 Levy 步长分布
- $\beta_i$：控制固定步长下对已填色邻居数的 softmax 敏感性

## 1. 单步动作概率分解

单步概率定义为：$p(a_{it} \mid s_{it}, b_i, \beta_i) = p(L_{it} \mid b_i)\, p(piece_{it} \mid s_{it}, L_{it}, \beta_i)\, p(color_{it} \mid piece_{it}, s_{it})$。

等价写法：$p(action_{it} \mid state_{it}, b_i, \beta_i) = p(L_{it} \mid b_i)\, p(piece_{it} \mid state_{it}, L_{it}, \beta_i)\, p(color_{it} \mid piece_{it}, state_{it})$。

## 2. Levy 步长项

记 $\mathcal{L}$ 为全局图步长支持集，即所有 trial 的所有候选位置里实际出现过的正图步长。

则有：$p(L_{it} = \ell \mid b_i) = \frac{\ell^{-b_i}}{\sum_{\ell' \in \mathcal{L}} (\ell')^{-b_i}}$，其中 $\ell \in \mathcal{L}$。

## 3. 固定步长后的 piece 选择

给定这一步已经产生步长 $L_{it}$，定义候选位置集合：$\mathcal{C}_{it}(L_{it}) = \{ r : r \text{ 当前可选，且 } d_{graph}(r, prev_{it}) = L_{it} \}$。

其中：

- $r$：某个候选 region
- $prev_{it}$：上一步所在位置
- $d_{graph}(\cdot,\cdot)$：图最短路步长

对任意候选位置 $r \in \mathcal{C}_{it}(L_{it})$，记 $N_{it}(r)$ 为该位置当前已填色邻居的个数。

效用函数定义为：$U_{it}(r) = \beta_i N_{it}(r)$。

在固定步长条件下，piece 选择服从 softmax：$p(piece_{it} = r \mid s_{it}, L_{it}, \beta_i) = \frac{\exp(\beta_i N_{it}(r))}{\sum_{r' \in \mathcal{C}_{it}(L_{it})} \exp(\beta_i N_{it}(r'))}$。

## 4. 颜色项

颜色项采用 canonical effective colors。

记 $E_{it}(piece_{it})$ 为当前状态下所选位置的有效合法颜色数，则：$p(color_{it} \mid piece_{it}, s_{it}) = \frac{1}{E_{it}(piece_{it})}$。

## 5. 被试层面的 likelihood

对被试 $i$，全部 trial 的 likelihood 为：$\mathcal{L}_i(b_i,\beta_i) = \prod_t p(a_{it} \mid s_{it}, b_i, \beta_i)$。

对应的 log-likelihood 为：$\log \mathcal{L}_i(b_i,\beta_i) = \sum_t \log p(L_{it} \mid b_i) + \sum_t \log p(piece_{it} \mid s_{it}, L_{it}, \beta_i) + \sum_t \log p(color_{it} \mid piece_{it}, s_{it})$。

对每个被试分别做最大似然估计：$(\hat b_i, \hat \beta_i) = \arg\max_{b_i,\beta_i} \log \mathcal{L}_i(b_i,\beta_i)$。

## 6. 三个比较模型

联合模型：$Joint: \; p(L_{it} \mid b_i)\, p(piece_{it} \mid s_{it}, L_{it}, \beta_i)\, p(color_{it} \mid piece_{it}, s_{it})$。

仅 Levy 模型：$Levy\text{-}only: \; p(L_{it} \mid b_i)\, p_{unif}(piece_{it} \mid s_{it}, L_{it})\, p(color_{it} \mid piece_{it}, s_{it})$。

仅邻居模型：$Neighbor\text{-}only: \; p_{unif}(L_{it})\, p(piece_{it} \mid s_{it}, L_{it}, \beta_i)\, p(color_{it} \mid piece_{it}, s_{it})$。

## 7. PPT 使用建议

如果要复制到 Keynote / PPT，优先复制下面这 4 条：

- 单步概率分解
- Levy 步长项
- softmax 的 piece 选择项
- 总 log-likelihood
