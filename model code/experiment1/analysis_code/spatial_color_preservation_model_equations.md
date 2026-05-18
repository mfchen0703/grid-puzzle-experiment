# Spatial-Color Preservation Model

这份文档整理一个新的实验1行为模型。该模型将每一步选择分解为：

1. `region` 选择：是否倾向在空间上保持连续，优先选择与上一步区域更近的区域  
2. `color` 选择：是否倾向在颜色上保持连续，优先延续上一步使用的颜色

当前先考虑一个最简的两参数版本：

- $\theta_s$：空间保持（spatial preservation）
- $\phi_{\text{same\_color}}$：颜色保持（same-color preservation）

## 1. 记号

对被试 $i$，第 $t$ 个有效着色步骤定义为：

- $r_{it}$：第 $t$ 步选择的 region
- $c_{it}$：第 $t$ 步选择的颜色
- $s_{it}$：第 $t$ 步开始时的状态
- $r_{i,t-1}$：上一步选择的 region
- $c_{i,t-1}$：上一步选择的颜色

其中状态 $s_{it}$ 至少包含：

- 当前每个 region 的着色状态
- 哪些 region 仍未着色
- 对每个未着色 region，当前哪些颜色是合法颜色

定义：

- $U(s_{it})$：状态 $s_{it}$ 下所有未着色 region 的集合
- $C_{\text{legal}}(r, s_{it})$：在状态 $s_{it}$ 下，对 region $r$ 合法的颜色集合
- $d(r, r_{i,t-1})$：region $r$ 与上一步 region $r_{i,t-1}$ 的 graph distance

## 2. 总体分解

该模型把一步行为概率分解为：

$$
\begin{aligned}
p(r_{it}, c_{it} \mid s_{it}, r_{i,t-1}, c_{i,t-1})
=\;&
p(r_{it} \mid s_{it}, r_{i,t-1})
\cdot
p(c_{it} \mid r_{it}, s_{it}, c_{i,t-1})
\end{aligned}
$$

也就是说：

- 先决定下一步要去哪个 region
- 再在该 region 当前合法的颜色中选择颜色

## 3. Region 选择模型

先定义一个空间连续性的特征：

$$
\begin{aligned}
f_{\mathrm{spatial}}(r_{it}, r_{i,t-1})
=\;&
-d(r_{it}, r_{i,t-1})
\end{aligned}
$$

该特征越大，表示候选 region 越接近上一步区域。

于是 region 选择概率可写成：

$$
\begin{aligned}
p(r_{it} \mid s_{it}, r_{i,t-1}; \theta_{s,i})
=\;&
\frac{
\exp\big(\theta_{s,i} \, f_{\mathrm{spatial}}(r_{it}, r_{i,t-1})\big)
}{
\sum_{r' \in U(s_{it})}
\exp\big(\theta_{s,i} \, f_{\mathrm{spatial}}(r', r_{i,t-1})\big)
}
\end{aligned}
$$

其中：

- $\theta_{s,i} > 0$ 表示被试更偏好空间连续，即更愿意选择离上一步更近的 region
- $\theta_{s,i} \approx 0$ 表示空间位置对 region 选择影响较弱
- 如果需要，后续也可把 $f_{\mathrm{spatial}}$ 改成更离散的版本，例如：
  - 邻居 region 记为 $1$
  - 邻居的邻居记为 $0$
  - 更远区域记为负值

但当前推荐先使用 graph distance 的连续形式。

## 4. Color 选择模型

对已经选定的 region $r_{it}$，在其当前合法颜色集合 $C_{\text{legal}}(r_{it}, s_{it})$ 上做 softmax。

定义一个颜色保持特征：

$$
\begin{aligned}
f_{\mathrm{same}}(c_{it}, c_{i,t-1})
=\;&
\mathbf{1}\!\left[c_{it} = c_{i,t-1}\right]
\end{aligned}
$$

则颜色选择概率为：

$$
\begin{aligned}
p(c_{it} \mid r_{it}, s_{it}, c_{i,t-1}; \phi_{\mathrm{same},i})
=\;&
\frac{
\exp\big(\phi_{\mathrm{same},i} \, \mathbf{1}\!\left[c_{it} = c_{i,t-1}\right]\big)
}{
\sum_{c' \in C_{\mathrm{legal}}(r_{it}, s_{it})}
\exp\big(\phi_{\mathrm{same},i} \, \mathbf{1}\!\left[c' = c_{i,t-1}\right]\big)
}
\end{aligned}
$$

其中：

- $\phi_{\text{same\_color},i} > 0$ 表示更倾向延续上一步颜色
- $\phi_{\text{same\_color},i} = 0$ 表示对当前合法颜色近似均匀选择
- $\phi_{\text{same\_color},i} < 0$ 表示更倾向换色

## 5. 单步行为概率

把上面两部分合起来，一步行为概率为：

$$
\begin{aligned}
p(r_{it}, c_{it} \mid s_{it}, r_{i,t-1}, c_{i,t-1})
=\;&
\left[
\frac{
\exp\big(\theta_{s,i} \, (-d(r_{it}, r_{i,t-1}))\big)
}{
\sum_{r' \in U(s_{it})}
\exp\big(\theta_{s,i} \, (-d(r', r_{i,t-1}))\big)
}
\right] \\
&\cdot
\left[
\frac{
\exp\big(\phi_{\mathrm{same},i} \, \mathbf{1}\!\left[c_{it} = c_{i,t-1}\right]\big)
}{
\sum_{c' \in C_{\mathrm{legal}}(r_{it}, s_{it})}
\exp\big(\phi_{\mathrm{same},i} \, \mathbf{1}\!\left[c' = c_{i,t-1}\right]\big)
}
\right]
\end{aligned}
$$

## 6. 被试级 log-likelihood

对被试 $i$，全部有效步骤的 likelihood 为：

$$
\begin{aligned}
\mathcal{L}_i(\theta_{s,i}, \phi_{\mathrm{same},i})
=\;&
\prod_t
p(r_{it}, c_{it} \mid s_{it}, r_{i,t-1}, c_{i,t-1})
\end{aligned}
$$

对应的 log-likelihood 为：

$$
\begin{aligned}
\log \mathcal{L}_i(\theta_{s,i}, \phi_{\mathrm{same},i})
=\;&
\sum_t \log p(r_{it} \mid s_{it}, r_{i,t-1}; \theta_{s,i}) \\
&+
\sum_t \log p(c_{it} \mid r_{it}, s_{it}, c_{i,t-1}; \phi_{\mathrm{same},i})
\end{aligned}
$$

每个被试分别进行最大似然估计：

$$
\begin{aligned}
(\hat{\theta}_{s,i}, \hat{\phi}_{\mathrm{same},i})
=\;&
\arg\max_{\theta_{s,i},\, \phi_{\mathrm{same},i}}
\log \mathcal{L}_i(\theta_{s,i}, \phi_{\mathrm{same},i})
\end{aligned}
$$

## 7. 参数解释

这个模型的两个参数可以解释为两种不同的 preservation tendency：

- $\theta_s$
  - 衡量被试是否倾向保持空间上的连续性
  - 越大表示越偏好与上一步区域相邻、或更接近的 region

- $\phi_{\text{same\_color}}$
  - 衡量被试是否倾向保持颜色上的连续性
  - 越大表示越偏好继续使用上一步颜色

因此：

- 高 $\theta_s$、低 $\phi_{\text{same\_color}}$ 的被试：
  - 更像“沿着局部区域连续行动，但颜色可以灵活变化”

- 低 $\theta_s$、高 $\phi_{\text{same\_color}}$ 的被试：
  - 更像“尽量维持颜色不变，即使 region 选择上更跳跃”

## 8. 当前版本与下一步扩展

当前版本是最简的两参数模型，优点是：

- 结构清楚
- 容易拟合
- 参数解释直接

但它还没有直接刻画一种更强的交互策略：

- “为了继续使用上一步颜色，被试会主动跳去一个更远但仍可合法使用该颜色的 region”

如果后续需要，可以把这个交互直接加入 region 选择部分，例如再增加一个特征：

$$
\begin{aligned}
f_{\mathrm{color\_preserve}}(r_{it}, c_{i,t-1}, s_{it})
=\;&
\mathbf{1}\!\left[c_{i,t-1} \in C_{\mathrm{legal}}(r_{it}, s_{it})\right]
\end{aligned}
$$

从而形成一个更完整的三特征或三参数模型。

但当前建议先从本文档中的两参数版本开始拟合。
