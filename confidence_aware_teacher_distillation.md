# Confidence-Aware Teacher Distillation for Space2Act-VLA

## 1. 目的与问题定义

当前 S/F/P 教师特征是每个 anchor 的单次离线推理结果。即使特征来自较低噪声的中间 denoising step，随机 seed、输入扰动、相机条件或时间边界仍可能导致局部不稳定。如果学生对所有离线特征使用等权 alignment loss，低可靠 target 可能被当作确定事实蒸馏，导致学生 hidden 出现偏差，并进一步影响动作生成。

本设计增加一个训练期的教师可靠性估计：

1. 同一个 anchor 和 channel 执行多次语义保持的教师推理；
2. 使用多次输出的均值作为最终离线 target，降低单次 realization 噪声；
3. 用多次输出的一致性生成一个 anchor/channel 级 confidence scalar；
4. 用该 scalar 连续缩放对应的 alignment loss；
5. 推理时不读取教师特征、confidence 或 mapper，因此不增加部署依赖。

该机制称为 **confidence-aware teacher alignment**。第一版只影响教师对齐监督，不把 confidence 输入 AISER。

## 2. 设计原则

- **语义保持**：多次采样只能改变教师推理的不确定性来源，不能改变任务、anchor、目标 token 顺序或监督语义。
- **均值 target**：保存多次输出的均值，而不是任意一次输出，避免把随机 realization 直接写入监督。
- **标量控制**：最终每个 anchor/channel 只保存一个 confidence，控制该 channel 的整体监督强度；token-level disagreement 只作为计算中间量。
- **可配置降权**：默认允许 `confidence=0` 完全关闭该 target 的 alignment 梯度；可通过
  `min_weight` 调高下限，形成软降权并避免对单一阈值的依赖。
- **尺度稳定**：confidence 加权后按有效权重重新归一化，使不同 batch、channel 和 confidence 分布下的 loss 尺度可比较。
- **可复现与可回退**：缓存和 metadata 必须记录生成协议；关闭开关时严格复现现有等权 alignment。

## 3. 离线特征抽取

### 3.1 抽取单位

抽取单位仍然是现有的 `(episode_id, anchor_frame, channel)`。对于每个单位执行 `K=4` 次教师推理，记输出为：

$$
Z^{(k)}_{a,c} \in \mathbb{R}^{N_c \times D_c}, \quad k=1,\ldots,K.
$$

所有重复推理必须复用当前缓存协议中的 layer、denoising step、CFG branch、resolution、token order 和 anchor 映射。不能因为增加 confidence 估计而更换 teacher target 定义。

### 3.2 各 channel 的语义保持扰动

| Channel | 固定条件 | 允许改变的随机性 | 禁止改变的内容 |
|---|---|---|---|
| S / spatial | 当前图像、anchor、相机轨迹采样规则、目标层 | 教师随机 seed；可选轻微相机轨迹等价扰动 | 任务语义、相机轨迹的有效几何约束、token 数和顺序 |
| F / future | 当前图像、指令、17-frame rollout、CFG、block 20、step 19 | diffusion seed；可选与训练协议一致的生成噪声 | 指令、anchor、rollout horizon、positive branch 和输出布局 |
| P / plan | anchor、future window、proprio、Moto checkpoint | 教师 dropout/seed；可选不改变时间语义的输入增强 | 四段 delta-4 划分、尾部策略、checkpoint 和 pre-VQ 定义 |

默认只改变 seed；输入或相机扰动属于独立消融，不能与主方案混用。每次抽取记录确定的 seed 列表，保证中断后可以续跑并复算相同结果。

### 3.3 Target 与不确定性统计

首先计算均值 target：

$$
\bar Z_{a,c} = \frac{1}{K}\sum_{k=1}^{K}Z^{(k)}_{a,c}.
$$

为使统计与当前 cosine alignment 几何一致，对每次输出沿 feature dimension 做无参数 LayerNorm：

$$
\tilde Z^{(k)}_{a,c,n}=\operatorname{LN}(Z^{(k)}_{a,c,n}).
$$

每个 teacher token 的 pairwise cosine disagreement 定义为：

$$
d_{a,c,n}=1-\frac{2}{K(K-1)}\sum_{i<j}
\cos(\tilde Z^{(i)}_{a,c,n},\tilde Z^{(j)}_{a,c,n}).
$$

anchor/channel 级 raw disagreement 使用 token 维度的中位数作为主定义：

$$
u_{a,c}=\operatorname{median}_{n=1}^{N_c}d_{a,c,n}.
$$

中位数比均值更不容易被少量异常 token 主导。实现可额外记录 `mean_disagreement`、`p90_disagreement` 供诊断，但训练主路径只使用 `u`。

### 3.3.1 存储受限下的流式统计

由于单个 teacher feature 已经很大，抽取过程不能保存同一 anchor 的 `K` 份完整输出。confidence 必须在特征抽取侧计算，而不是由训练 loader 根据磁盘内容重新估计。对每个 `(anchor, channel)`，抽取器按顺序运行 K 次推理；第 k 次输出处理完后立即释放，只保留以下临时统计量：

- `feature_sum`：原始输出的 float32 累加，用于生成均值 target；
- `normalized_sum`：逐 token LayerNorm 后、再做 L2 normalization 的特征累加；
- `normalized_sqnorm_sum`：上述归一化特征的逐 token 平方范数累加，用于数值稳定地计算 pairwise cosine；
- `sample_count`：已经成功完成的推理次数。

不需要保存 `Z^(1),...,Z^(K)`。第 k 次输出 `Z^(k)` 到达后，抽取器在 token 或 feature chunk 上完成：

```text
feature_sum += Z^(k)
x^(k) = L2Normalize(LayerNorm(Z^(k), last_dim), last_dim)
normalized_sum += x^(k)
normalized_sqnorm_sum += ||x^(k)||_2^2
sample_count += 1
```

其中累加器使用 float32（显存允许时可使用 float64），最终均值 target 才转换为既有缓存 dtype `bfloat16`。如果特征维度过大，chunk 只改变临时峰值内存，不改变统计结果。

K 次推理完成后，写盘前计算：

$$
\bar Z=\frac{\texttt{feature\_sum}}{K},
\qquad
\overline{\cos}_{a,c,n}=
\frac{\lVert\texttt{normalized\_sum}_{a,c,n}\rVert_2^2-
\texttt{normalized\_sqnorm\_sum}_{a,c,n}}
{K(K-1)}.
$$

这里的恒等式来自
\(\sum_{i<j}x_i^\top x_j=(\lVert\sum_i x_i\rVert_2^2-\sum_i\lVert x_i\rVert_2^2)/2\)。由于 `x` 已做 L2 normalization，`normalized_sqnorm_sum` 理论上接近 K；显式累加它可以减少混合精度和 LayerNorm 数值误差。随后令 `d=1-overline{cos}`，对 token 取中位数得到 `u`，再执行下一节的 channel-specific quantile calibration。

因此，抽取阶段的磁盘写入仅发生一次，并且只写入均值 `features`、confidence 和可选 raw disagreement。磁盘占用不会乘以 K；内存峰值最多包含当前一次输出和必要的累加器，而不是 K 份完整 feature。

抽取器必须在 `sample_count == K` 且所有统计量 finite、shape 一致时，才把当前 anchor 写入最终文件。若进程在 K 次推理中断，当前 anchor 的部分累加器不能标记为 complete，恢复时必须从该 anchor 的第 1 次推理重新开始；不能把不完整均值或 confidence 混入缓存。

### 3.4 Confidence 校准

不能直接把不同 channel 的 raw disagreement 当作同一尺度的权重。使用训练集、按 channel 独立计算的分位数边界 `q_low,c` 和 `q_high,c` 做校准：

$$
r_{a,c}=\operatorname{clip}\left(
\frac{u_{a,c}-q_{low,c}}{q_{high,c}-q_{low,c}+\epsilon},0,1\right),
\qquad
c_{a,c}=1-r_{a,c}.
$$

默认使用 `q_low=0.05`、`q_high=0.95`。分位数只能由训练 split 统计，不能使用验证/测试数据。校准参数写入抽取 manifest 或训练配置，不能隐式依赖当前机器上的动态统计。

建议同时保留 `raw_disagreement`，这样后续可以重新校准而不必重新运行昂贵的教师推理。

## 4. 缓存格式与版本契约

### 4.1 Safetensors episode 文件

在现有字段基础上新增：

```text
features       [A, N, D]   bfloat16   # K 次输出的均值 target
anchor_frames  [A]         int64
confidence     [A]         float32   # 抽取侧计算并校准后的 [0, 1] scalar
disagreement   [A]         float32   # 可选，抽取侧计算的 raw u，便于重新校准
```

`confidence[a]` 与 `features[a]`、`anchor_frames[a]` 的第一维必须完全一致。confidence 不保存为 `[A,N]`，以控制缓存增长并保持 loader 接口简单。训练 loader 只读取已经计算好的 confidence，不重新运行教师、不保留 K 次结果，也不根据单次 `features[a]` 反推 confidence。

### 4.2 必需 metadata

新增 metadata 至少包括：

```text
confidence_aware: true
confidence_version: "v1"
num_teacher_samples: "4"
target_aggregation: "mean"
uncertainty_metric: "pairwise_cosine_disagreement"
feature_normalization: "layernorm_last_dim"
uncertainty_aggregation: "median_token"
calibration: "train_quantile_05_95"
seed_policy: "explicit_per_anchor"
```

原有 `complete`、episode、suite、layer、stride、block、step、branch、shape 和 checkpoint metadata 继续保留。文件缺少 confidence 或 metadata 不得被标记为 confidence-aware cache；loader 应明确报错，而不是静默退化。

### 4.3 旧缓存兼容

旧的仅含 `features`/`anchor_frames` 的缓存仍可在 `confidence_aware_alignment.enabled=false` 时使用。开启 confidence-aware alignment 时必须拒绝旧缓存，避免用户误以为已经启用了可靠性调控。若需要对照实验，使用显式的 legacy 模式并在日志中记录 `confidence_aware=false`。

### 4.4 Packed cache

后续执行 `pack_teacher_latents.py` 时，packed manifest 应增加 confidence 的 dtype、offset 和 shape 信息；`confidence.bin` 与 `features.bin` 分开存储，避免改变现有 features 的 mmap 布局。packed loader 必须同时校验 episode 的 confidence count、offset 和 anchor count。

## 5. 训练期 loss 调控

令现有 aligner 对单个样本/anchor 返回未加权 loss：

$$
\ell_{a,c}=\frac{1}{N_c}\sum_n
\left[1-\cos(\operatorname{LN}(\hat Z_{a,c,n}),
\operatorname{LN}(\bar Z_{a,c,n}))
+\alpha_c\operatorname{MSE}(\cdot)\right].
$$

由 confidence 生成带下限的监督权重：

$$
w_{a,c}=w_{min}+(1-w_{min})c_{a,c},
\qquad w_{min}=0.0\ \text{(default, configurable)}.
$$

对 batch 中同一 channel 的 alignment loss 做有效权重归一化：

$$
\mathcal L_c^{conf}=
\frac{\sum_a w_{a,c}\ell_{a,c}}
{\sum_a w_{a,c}+\epsilon}.
$$

这样高不确定 target 仍保留至少 10% 的监督，且一个 batch 的 confidence 分布不会改变该 channel loss 的数量级。`loss_weight`、progressive/DWA/GradNorm 的 channel 系数仍位于外层：

$$
\mathcal L=\mathcal L_{act}
+\sum_{c\in\{S,F,P\}}\lambda_c\mathcal L_c^{conf}
+\mathcal L_{route}+\cdots.
$$

confidence 只作用于 alignment，不改变 action loss、`L_SF`/`L_FP` 的定义或 AISER 正则；若未来一致性损失使用 teacher feature，也应明确选择是否复用同一 confidence，主版本默认复用对应 future/plan 权重并单独记录。

### 5.1 配置开关

建议在 `latent_reasoning` 下增加：

```yaml
confidence_aware_alignment:
  enabled: false
  min_weight: 0.0
  require_cached_confidence: true
  use_mean_target: true
```

`enabled=false` 时保持现有等权路径和旧 checkpoint 行为。`enabled=true` 时：

- 只接受带 confidence metadata 的缓存；
- `use_mean_target` 必须与 cache metadata 一致；
- 训练日志输出 confidence、raw disagreement、effective weight 和有效监督比例；
- checkpoint 保存该配置和 calibration 参数，恢复训练时不得改变校准协议。

## 6. 与 AISER 的边界

confidence 不输入 AISER 的 query、key、value、attention logits 或 gate。AISER 仍分别对 S/F/P hidden 执行三次独立 channel routing；不存在三个 channel 共享 attention softmax 的直接竞争。

但三路共享 Qwen backbone、Action-Intent hidden 和 action loss，因此 confidence 会通过改变某一路 alignment 梯度，间接影响后续的 hidden 与路由结果。这是多任务优化的正常耦合，不应描述为“完全互不影响”。论文应使用以下边界表述：

> Confidence-aware alignment changes only the reliability of teacher supervision. AISER remains confidence-agnostic and routes each channel from the learned student hidden states.

必须包含 `alignment-only`、`AISER-only` 和 `both` 三个设置，区分 confidence 的直接收益与共享 backbone 的间接收益。

## 7. 数据处理与实现顺序

1. 修改三类教师抽取脚本：加入显式 seed 循环、K 次输出检查、均值 target 和 disagreement 计算。
2. 在抽取侧以流式累加器计算均值 target、token disagreement 和 anchor/channel scalar，不保存 K 份完整输出。
3. 生成每个 channel 的训练集 disagreement 分布和 quantile calibration manifest。
4. 写入新版本 episode safetensors，并执行 shape、dtype、anchor、finite-value 和 metadata 审计。
5. 更新 packed cache 工具和 manifest；确认 mmap features 的旧布局不被破坏。
6. loader 增加 confidence 读取接口，开启模式下严格校验，关闭模式下保持旧行为。
7. aligner 改为返回 per-example alignment loss，再应用 confidence 权重和有效权重归一化。
8. 训练入口接入配置开关、日志和 checkpoint sidecar。
9. 最后运行 teacher-free inference，确认部署路径没有读取 confidence、teacher feature 或 mapper。

## 8. 验证与消融

### 数据与指标校验

- K 次推理输出 shape、dtype、有限值和 token 顺序完全一致；
- 流式公式与在小型合成张量上显式保存 K 次结果计算的 pairwise disagreement 在浮点容差内一致；
- 抽取器在 K 次未完成或任一统计量非 finite 时不写入 complete anchor；
- 均值 target 与逐次输出的距离不大于随机单次 target 的平均距离；
- confidence 全部位于 `[0,1]`，并按 channel 报告 p05/p50/p95；
- raw disagreement 与留出 seed 的 feature disagreement 呈正相关；
- 高 confidence 样本的留出误差显著低于低 confidence 样本；
- train-only quantile calibration 不读取 validation/test episode。

### 训练消融

- legacy 单次 target + 等权 loss；
- 单次 target + confidence weighting；
- 均值 target + 等权 loss；
- **均值 target + confidence weighting（主方案）**；
- K=2/4/8；
- mean/median/trimmed-mean token aggregation；
- soft weighting 与 hard masking；
- `w_min=0`、`0.1`、`0.2`；
- confidence-aware alignment only、AISER only、both。

除 success rate 外，报告 alignment loss、action loss、effective supervision ratio、confidence 分桶 success、梯度 cosine 和 AISER gate/attention 统计。

## 9. 接受标准

本设计可进入主实验的最低条件为：

- confidence-aware cache 可被三类 loader 完整读取并严格校验；
- 同一随机 seed 可复现 features、disagreement 和 confidence；
- 开关关闭时与现有 loss 数值和 checkpoint 加载行为兼容；
- 开关开启时高置信度分桶具有更低的留出教师误差；
- teacher-free inference 与原有接口、action shape 和有限值检查保持一致；
- 主方案在至少一个空间/未来/精细接触指标上优于均值-only 或等权对照，且不以显著增加推理延迟为代价。
