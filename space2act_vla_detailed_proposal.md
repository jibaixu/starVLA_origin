# Space2Act-VLA: 多世界模型教师蒸馏的三通道潜在推理方案

## 实现基线与当前范围（2026-07）

本文档中的工程基线以仓库当前实现为准：**Qwen3-VL-4B-Instruct + GR00T
FlowmatchingActionHead**，而不是 OpenVLA-OFT。Qwen 接收多视角图像和指令，输出完整末层
hidden sequence；GR00T ActionHead 以该完整序列作为 cross-attention KV，并额外接收 proprio state。

第一版采用以下经过显存与扩展性权衡后的设计：

- 三组 S/F/P token 从 Stage 1 起全部注册到 tokenizer（默认每组 32 个），确保各阶段词表和 checkpoint shape 稳定；只有 `token_enabled` 的组被插入序列。
- 序列沿用 Qwen 原生 causal attention，物理顺序为 `images -> S -> instruction -> F -> P`。自定义 block-wise mask 是后续消融项，不属于当前 MVP。
- 学生侧每个通道只有 32 个 compact token。训练期 mapper 使用与教师 token 数相同的 learned queries，通过 cross-attention 读取 compact hidden，再映射到教师维度；推理完全跳过 mapper。
- Stage 1 的 LingBot 监督是 `[1521, 5120]`，mapper 为 `32 x 2560 -> 1521 x 5120`，内部维度 1024、8 heads、query chunk 128。
- 对齐损失是逐 token LayerNorm 后的 cosine + `0.1 * MSE`；总损失为 GR00T action loss 加启用通道的加权 alignment loss。
- `token_enabled` 与 `alignment_enabled` 独立，支持同 token 数无 teacher loss 的 random-token control。
- Stage 2/3 的接口、开关和 shape 校验已经预留，但 teacher shape 与路径在提取完成前保持未定义，启用时必须显式填写。

对应实现入口为：

- `starVLA/model/framework/VLM4A/QwenSpace2ActGR00T.py`
- `starVLA/model/modules/latent_reasoning/space2act_alignment.py`
- `examples/simBenchmarks/LIBERO/train_files/space2act_libero_state.yaml`
- `examples/simBenchmarks/LIBERO/train_files/run_space2act_train_jbx.sh`

当前工程验证已经完成：四个 LIBERO suite 的 episode 0/1/2 可组成 2067-step pilot；8 卡
ZeRO-2 单步训练成功得到有限的 action/alignment/total loss；保存后的 checkpoint 在不提供任何
teacher tensor、且禁止 mapper 执行的条件下可输出形状为 `[1, 8, 7]` 的有限 action。因此
“训练期 teacher + mapper、推理期仅 Qwen typed tokens + 原 GR00T ActionHead”的闭环已实际跑通。

下文保留了完整研究愿景；凡涉及 block-wise mask、跨角色一致性、teacher dropout 或额外 resampler 的内容，均应视为后续研究选项，而非第一版实现承诺。

## 0. 方案定位

本方案把潜在推理拆成三类具有明确语义职责的可学习 token：

```text
[视觉 token | 空间位置感知 token | 文本指令 token | 未来状态推演 token | 潜在动作规划 token]
```

三类 latent token 分别回答三个问题：

- `S` token: 当前场景中“物体、相机视角、可达区域、接触区域”之间的 3D 空间关系是什么？
- `F` token: 为完成语言任务，未来若干步中场景应当朝什么状态变化？
- `P` token: 在上述空间结构和未来变化约束下，机械臂应执行什么高层运动逻辑？

核心论文主张可以写成：

> Text CoT is a low-density and weakly grounded supervision signal for robotic action generation. Instead of distilling verbal reasoning into VLA, this work distills the complementary capacities of spatial exploration world models, embodied future-prediction world models, and latent motion/action teachers into typed latent reasoning tokens, enabling teacher-free future-state imagination and action planning at inference time.

也就是说，本方案不是让 VLA 在内部“默写”语言推理，而是构造一个更适合具身控制的潜在推理语法：`where` -> `what will change` -> `how to move`。

## 1. 研究动机

### 1.1 文本 CoT 与机器人控制之间存在表示错配

现有 CoT-enhanced VLA 通常沿用语言模型的思路：先显式或隐式地产生一段文本推理，再基于推理结果预测动作。这个范式在任务分解和语义规划上有一定价值，但对机器人低层控制来说存在三个结构性问题。

第一，文本 CoT 的信息密度低。真实操作中决定动作成败的往往是目标物与夹爪的相对位姿、遮挡关系、接触方向、容器开口、抽屉滑轨方向、障碍物边界等连续视觉-几何因素。文本 CoT 很容易写出“抓住杯子并放到盘子上”这样的高层描述，却难以稳定编码“杯柄朝右、夹爪应从左上方绕开遮挡、接触点应避开边缘”这类 action-critical 信息。

第二，文本 CoT 会引入大量场景无关和任务无关内容。语言推理往往包含对物体类别、任务常识和步骤顺序的重复描述，但这些内容未必对当前动作 chunk 的生成有直接贡献。对 VLA 来说，这会造成监督信号稀疏、延迟增加，以及潜在推理 token 被训练成“语言摘要器”而不是真正的世界状态推演器。

第三，文本 CoT 对连续未来状态的约束偏弱。机器人动作的本质是改变物理世界状态：杯子被移动，抽屉被打开，按钮被按下，物体进入容器。相比语言描述，视觉未来状态、变化区域、运动方向和连续潜动作表征更接近 action decoder 需要的条件变量。

### 1.2 世界模型提供了更高密度的潜在监督

当前可用于 VLA 潜在推理蒸馏的监督源可以分成三类。

第一类是 LingBot-World / GeoWorld-VLM 类型的空间探索世界模型。这类模型接收当前相机视角和相机运动条件，生成或预测视角变化下的场景演化。它的中间层表征不是普通图像语义特征，而是被“跨视角一致性”和“视角运动预测”塑形过的空间结构特征，因此更适合监督 VLA 内部的 3D 空间理解。

第二类是 BRIDGE-WA 类型的具身未来预测世界模型。这类模型接收当前多视角图像、机器人状态和语言指令，预测未来某个时间偏移处的场景状态或变化结构。它比通用视频生成模型更贴近机器人交互，因为预训练目标包含物体移动、接触诱发变化、任务条件下的结果状态。

第三类是 Moto / IDM 类型的潜动作教师。它不直接描述未来图像，而是把视频片段或状态转移压缩成连续的 motion/action latent。相比离散动作 token 或文本步骤，它提供的是“如何移动”的高层连续运动语义，适合作为潜在动作规划 token 的教师信号。

因此，本方案的关键不是简单增加 latent token 数量，而是把 VLA 的内部推理空间结构化为三种互补的可学习变量，并分别从三个教师源获得监督：

```text
LingBot/GeoWorld-style teacher  -> spatial position-aware tokens
Wan/BRIDGE-WA-style teacher     -> future state rollout tokens
Moto/IDM-style teacher          -> latent action planning tokens
```

## 2. 方法论

### 2.1 问题定义与记号

给定机器人演示数据集：

$$
\mathcal{D} = \{(o_t, q_t, \ell, a^*_{t:t+H-1}, o_{t+\Delta})\},
$$

其中 $o_t$ 表示当前视觉观测，可以是单视角或多视角 RGB 图像；$q_t$ 表示机器人本体状态；$\ell$ 表示语言指令；$a^*_{t:t+H-1}$ 是专家 action chunk；$o_{t+\Delta}$ 是训练数据中可用的未来观测，只用于训练教师或构造监督信号，推理时不可见。

基模采用 Qwen3-VL-4B + GR00T FlowmatchingActionHead。Qwen 将当前多视角观测与语言指令编码为 hidden states，GR00T 动作头输出连续 action chunk。本文在 Qwen 输入序列中插入三组可学习 latent tokens：

- $S=\{s_i\}_{i=1}^{N_s}$: 空间位置感知 token；
- $F=\{f_j\}_{j=1}^{N_f}$: 未来状态推演 token；
- $P=\{p_k\}_{k=1}^{N_p}$: 潜在动作规划 token。

主输入序列为：

$$
X_t = [V(o_t, q_t) \;|\; S \;|\; L(\ell) \;|\; F \;|\; P],
$$

其中 $V(o_t)$ 表示 Qwen processor 生成的视觉 token；当前实现不把 $q_t$ 拼入 Qwen 序列，而是沿用 base contract 将 proprio state 独立输入 GR00T ActionHead。

模型输出三组对应 hidden states：

$$
H_S, H_F, H_P = f_\theta(X_t),
$$

动作预测由三类 latent reasoning token、视觉 token 和指令 token 共同条件化：

$$
\hat{a}_{t:t+H-1} = g_\theta(H_V, H_L, H_S, H_F, H_P, q_t).
$$

训练阶段使用多个冻结教师生成对齐目标；推理阶段只保留 VLA 本体、三类可学习 token 和动作头，不需要调用任何教师模型。

### 2.2 三通道 latent reasoning 架构

本方案可以称为 **Space2Act-VLA** 或 **TriWorld-Latent VLA**。其核心架构如下：

```text
Current visual observation + proprioception + instruction
        |
        v
Qwen3-VL-4B backbone with typed learnable latent tokens
        |
        |-- S tokens: spatial relation / 3D layout / reachable geometry
        |       teacher: LingBot/GeoWorld-style spatial world model
        |
        |-- F tokens: task-conditioned future state / world-change direction
        |       teacher: embodied Wan/BRIDGE-WA-style future world model
        |
        |-- P tokens: high-level latent motor program
                teacher: Moto/IDM continuous pre-quantization action latent
        |
        v
Action decoder / flow-matching action head
        |
        v
Continuous action chunk
```

三类 token 使用不同的 token type embedding 和不同的 projection head。这样做的目的不是人为增加参数，而是给 latent space 一个清晰的因果语义分工：

- $S$ token 被塑造成空间坐标系中的场景结构表示；
- $F$ token 被塑造成任务条件下的未来变化表示；
- $P$ token 被塑造成与动作 decoder 更近的运动计划表示。

这种设计比“所有 latent token 都共享同一种 `<st>` token”更适合 VLA，因为机器人控制天然包含空间、动态和运动三层结构。

### 2.3 第一版 causal routing 与后续 block-mask 消融

第一版不修改 Qwen3-VL 内部 attention 实现，直接使用其原生 causal mask。通过 token 的物理排布表达“当前感知 -> 空间理解 -> 未来推演 -> 动作计划”的顺序：

设 token block 顺序为：

$$
V \prec S \prec L \prec F \prec P \prec A,
$$

其中 $A$ 不属于 Qwen token 序列，而是 GR00T ActionHead 中读取完整 Qwen hidden states 的动作查询。原生 causal mask 定义为：

$$
M_{ij} =
\begin{cases}
1, & \text{if } rank(B_j) \le rank(B_i), \\
0, & \text{otherwise},
\end{cases}
$$

其中 $B_i$ 是第 $i$ 个 token 所属 block。$M_{ij}=1$ 表示第 $i$ 个 token 可以 attend 到第 $j$ 个 token。

具体可见性如下：

| Query block | 可见信息 | 设计目的 |
|---|---|---|
| $V$ | 先前视觉位置 | 保持 Qwen 原生多模态 causal 行为 |
| $S$ | $V, S$ | 学习任务无关或弱任务相关的 3D 空间结构 |
| $L$ | $V, S, L$ | 让语言理解可以读取空间结构，避免纯文本解释场景 |
| $F$ | $V, S, L, F$ | 基于当前视觉、空间关系和语言任务推演未来变化 |
| $P$ | $V, S, L, F, P$ | 基于空间与未来推演形成潜在动作计划 |
| $A$ | $V, S, L, F, P, A$ | 动作头读取完整潜在推理结果 |

同一 latent block 在当前实现中也是 causal 的，不额外改成双向 set attention。ActionHead 对有效 Qwen KV 使用 cross-attention，因此能读取视觉、指令和所有已启用 reasoning token。

自定义 block-wise mask（例如组内双向、组间单向）仍是有价值的后续消融，但需要修改 Qwen attention mask/position 处理并重新验证 flash attention 兼容性。MVP 先隔离“typed token + teacher alignment”本身的收益，避免把 routing 改动混入第一阶段结论。

### 2.4 空间位置感知 token 的世界模型对齐

空间教师记为 $T_s$，可以采用 LingBot-World 类型的空间探索世界模型，或者采用 GeoWorld-VLM 思想中的 camera-conditioned world model。教师输入当前观测和采样相机轨迹：

$$
G_t^{(\ell)} = T_s^{(\ell)}(o_t, \pi_{\text{cam}}),
$$

其中 $G_t^{(\ell)}$ 是空间世界模型第 $\ell$ 层的中间表征，$\pi_{\text{cam}}$ 是相机视角探索轨迹。使用中间层而不是最终生成视频，是因为中间层更接近“跨视角一致的空间结构”，而最终像素或 VAE latent 会混入纹理、光照和视频生成细节。

当前 LingBot extractor 已直接保存固定数量的空间目标 token，不再额外压缩：

$$
S_t^* = G_t^{(24)} \in \mathbb{R}^{1521 \times 5120}.
$$

学生端取 Qwen 中 32 个 $S$ token hidden states。训练期 mapper 用 1521 个 learned target queries 对这 32 个 hidden states 做 cross-attention，再投影到教师空间：

$$
\hat{S}_t = P_s(\operatorname{CrossAttn}(Q_s, H_S)) \in \mathbb{R}^{1521 \times 5120}.
$$

空间对齐损失为：

$$
\mathcal{L}_{S}
= \frac{1}{N_s}\sum_{i=1}^{N_s}
\left[
1 - \cos(\text{LN}(\hat{S}_{t,i}), \text{LN}(S^*_{t,i}))
+ \alpha_s \lVert \text{LN}(\hat{S}_{t,i}) - \text{LN}(S^*_{t,i}) \rVert_2^2
\right].
$$

这里使用 LN 后的 cosine + MSE，而不是直接 MSE，是为了降低教师和学生 feature scale 不一致带来的训练不稳定。教师特征使用 stop-gradient。

空间 token 的监督目标不是“预测未来”，而是为后续未来推演提供可操作的空间基底：目标物在哪里、障碍物如何分布、相机视角变化下物体关系是否稳定、夹爪从哪个方向接近更可行。

### 2.5 未来状态推演 token 的中间层世界动态对齐

未来教师记为 $T_f$，采用 Wan / BRIDGE-WA 类型的具身任务视频世界模型。教师需要先在机器人轨迹数据上预训练，使其输入当前机器人上下文时能够预测未来状态：

$$
W_t^{(\ell, \Delta)}
= T_f^{(\ell)}(o_t, q_t, \ell_{\text{text}}, \Delta),
$$

其中 $\Delta$ 表示未来时间偏移，$W_t^{(\ell, \Delta)}$ 是教师在预测未来状态过程中的第 $\ell$ 层中间表征。

本方案刻意不对齐最终层输出，而对齐中间层表征。理由如下：

- 最终层更接近像素重建或 VAE latent 重建，包含大量纹理、背景和渲染细节；
- 最终层往往已经被压成单一未来图像结果，可能丢失“变化方向”和“可行动因果因素”；
- 中间层更可能保留任务条件、物体交互、变化区域和运动趋势，是更适合 action generation 的 world-change representation；
- 如果直接使用最终层，容易让 VLA 学到教师的生成器偏差，而不是学到未来推演能力。

因此，未来目标 token 由中间层动态特征压缩得到：

$$
F_t^* = R_f(\{W_t^{(\ell, \Delta)}\}_{\ell \in \mathcal{L}_f, \Delta \in \mathcal{H}})
\in \mathbb{R}^{N_f \times d},
$$

其中 $\mathcal{H}$ 可以是多尺度未来窗口，例如 $\{4, 8, 16\}$ 或与 action chunk 长度解耦的固定 future index。多尺度 future 的好处是同时覆盖接触前准备、接触瞬间变化和动作结果状态。

学生端：

$$
\hat{F}_t = P_f(H_F).
$$

未来状态对齐损失为：

$$
\mathcal{L}_{F}
= \frac{1}{N_f}\sum_{j=1}^{N_f}
\left[
1 - \cos(\text{LN}(\hat{F}_{t,j}), \text{LN}(F^*_{t,j}))
+ \alpha_f \lVert \text{LN}(\hat{F}_{t,j}) - \text{LN}(F^*_{t,j}) \rVert_2^2
\right].
$$

如果教师同时能输出变化图或运动流图，可以额外构造空间-未来一致性损失。令 $A_{F \rightarrow S}$ 表示 future token 对 spatial token 的 cross-attention 权重，$M_\Delta^*$ 表示教师预测的任务相关变化区域，经 object/patch-to-token mapping 后映射到 $S$ token 空间：

$$
\mathcal{L}_{SF}
= \text{KL}\left(
\text{stopgrad}(R_{\text{map}}(M_\Delta^*))
\;\|\;
A_{F \rightarrow S}
\right).
$$

该损失强制未来推演不是无条件视频预测，而是集中在空间 token 标记的可操作区域上。

### 2.6 潜在动作规划 token 的连续动作表征对齐

动作规划教师记为 $T_p$，采用 Moto / IDM 类型的潜动作编码器。教师先在机器人视频或演示数据上预训练，学习从状态转移中抽取 motion/action latent：

$$
\tilde{z}^{m}_t = E_{\text{IDM}}(o_t, o_{t+\Delta}, q_t, \ell_{\text{text}}).
$$

关键点是使用量化前的连续潜动作表征 $\tilde{z}^{m}_t$，而不是 VQ 后的离散 token。原因是离散 token 更适合视频生成或动作语义聚类，但会损失小位移、接触时序、速度变化等连续控制细节。VLA 的 action decoder 需要连续动作条件，因此 pre-quantization latent 更合适。

将连续 motion latent 压缩成 $N_p$ 个动作规划目标 token：

$$
P_t^* = R_p(\tilde{z}^{m}_t) \in \mathbb{R}^{N_p \times d}.
$$

学生端：

$$
\hat{P}_t = P_p(H_P).
$$

动作规划对齐损失为：

$$
\mathcal{L}_{P}
= \frac{1}{N_p}\sum_{k=1}^{N_p}
\left[
1 - \cos(\text{LN}(\hat{P}_{t,k}), \text{LN}(P^*_{t,k}))
+ \alpha_p \lVert \text{LN}(\hat{P}_{t,k}) - \text{LN}(P^*_{t,k}) \rVert_2^2
\right].
$$

为了让 $P$ token 不只是模仿视频 motion latent，还真正服务于未来状态变化，可以加入 future-action consistency。用一个轻量预测头从空间 token 和动作规划 token 预测未来教师表征：

$$
\tilde{F}_t = T_{\eta}(H_S, H_P),
$$

并约束它接近 $F_t^*$：

$$
\mathcal{L}_{FP}
= \lVert \text{LN}(\tilde{F}_t) - \text{LN}(\text{stopgrad}(F_t^*)) \rVert_2^2.
$$

这会让动作规划 token 学到“这个高层运动计划应该导致什么未来变化”，避免 $P$ token 退化成 action head 的额外容量。

### 2.7 动作监督与总损失

当前 base 固定采用 GR00T flow-matching action head。若作为对照实验改用简单连续回归，动作损失可写为：

$$
\mathcal{L}_{act}
= \lVert \hat{a}_{t:t+H-1} - a^*_{t:t+H-1} \rVert_1
+ \beta \lVert \hat{a}_{t:t+H-1} - a^*_{t:t+H-1} \rVert_2^2.
$$

若沿用 flow matching，则可写为：

$$
\mathcal{L}_{act}
= \mathbb{E}_{\tau,\epsilon}
\left[
\lVert v_\theta(a_\tau, \tau, c_t) - v^*_\tau \rVert_2^2
\right],
$$

其中 $c_t$ 是由 $H_V, H_L, H_S, H_F, H_P$ 构成的条件表示。

总损失为：

$$
\mathcal{L}
= \mathcal{L}_{act}
+ \lambda_S \mathcal{L}_S
+ \lambda_F \mathcal{L}_F
+ \lambda_P \mathcal{L}_P
+ \lambda_{SF}\mathcal{L}_{SF}
+ \lambda_{FP}\mathcal{L}_{FP}
+ \lambda_{role}\mathcal{L}_{role}.
$$

$\mathcal{L}_{role}$ 用于防止三类 token 表征坍缩，可采用简单的 role disentanglement 正则：

$$
\mathcal{L}_{role}
= \sum_{U,V \in \{S,F,P\}, U \ne V}
\lVert \text{Cov}(\bar{H}_U, \bar{H}_V) \rVert_F^2,
$$

其中 $\bar{H}_U$ 是某一类 token 的 pooled representation。这个正则不应过强，否则会破坏 action loss 所需要的跨角色信息融合。

## 3. 训练策略

### 3.1 教师预训练与缓存

在训练 VLA student 前，需要准备三个教师源。

第一，空间世界模型教师。若 LingBot-World 类型模型已可用，可以直接冻结并抽取中间层特征；如果不可用，可以先使用 GeoWorld-VLM 风格的 camera-conditioned video world model 作为替代。教师输入当前图像和采样相机轨迹，输出中间层空间表征。该教师不需要理解机器人动作，只需提供视角一致的空间结构。

第二，Wan / BRIDGE-WA 类型未来教师。该教师需要在具身场景数据上预训练，输入当前多视角图像、本体状态和语言指令，预测未来状态。预训练目标可以包括未来图像 latent、变化区域、motion flow 或中间视频 token。预训练完成后冻结教师，并缓存每个训练样本在多个 future horizon 上的中间层表征。

第三，Moto / IDM 类型潜动作教师。该教师需要在机器人视频或演示轨迹上预训练，输入当前和未来观测，学习压缩状态转移的连续 motion/action latent。对 VLA student 训练时，取量化前连续 latent 作为 $P$ token 的监督目标。

所有教师都只在训练阶段使用。部署时不调用教师，也不生成视频或文本 CoT。

### 3.2 三阶段课程学习

可以采用联合训练，也可以采用更稳的课程学习。推荐先做三阶段课程学习，因为三个教师信号的尺度、语义和收敛速度不同。

**Stage 1: Spatial grounding warm-up**

只加入 $S$ token，训练 $\mathcal{L}_{act} + \lambda_S\mathcal{L}_S$。这一阶段的目标是让 VLA 学会把世界模型的空间结构吸收到 latent token 中，而不是直接让未来和动作规划 token 同时竞争表征空间。

可训练参数建议为：latent token embedding、role projector、LoRA adapter、action head；教师冻结。若资源有限，可以先冻结 VLM backbone 只训练 token 和 projector，验证空间 probe 后再放开 LoRA。

**Stage 2: Future rollout distillation**

在 Stage 1 基础上加入 $F$ token，训练：

$$
\mathcal{L}_{act}
+ \lambda_S\mathcal{L}_S
+ \lambda_F\mathcal{L}_F
+ \lambda_{SF}\mathcal{L}_{SF}.
$$

这一阶段应重点调 future horizon、教师中间层位置和 $F$ token 数量。训练初期可以固定 $S$ token projector 或使用较小学习率，避免未来教师信号破坏已学到的空间结构。

**Stage 3: Latent action planning alignment**

加入 $P$ token 和潜动作教师，训练完整损失：

$$
\mathcal{L}_{act}
+ \lambda_S\mathcal{L}_S
+ \lambda_F\mathcal{L}_F
+ \lambda_P\mathcal{L}_P
+ \lambda_{SF}\mathcal{L}_{SF}
+ \lambda_{FP}\mathcal{L}_{FP}.
$$

这一阶段让 $P$ token 学习高层运动逻辑，并通过 action loss 绑定到底层连续动作。推荐在后半程加入 teacher dropout：随机丢弃某一类 teacher alignment loss，只保留 action loss 和其他 token 的监督。这可以避免 student 过度依赖某一个教师信号，也更符合推理时 teacher-free 的设置。

### 3.3 联合训练替代方案

如果算力充足，也可以从一开始加入全部 token 并联合训练，但建议使用 loss ramp-up：

```text
early:   L_act + small L_S
middle:  L_act + L_S + ramp(L_F)
late:    L_act + L_S + L_F + ramp(L_P + L_consistency)
```

联合训练的优点是三类 token 可以更早协同；缺点是更容易出现 token role collapse，即所有 token 都变成 action head 的普通上下文容量。若采用联合训练，必须保留容量匹配 baseline、role probe 和 token masking 实验。

## 4. 实验方案

### 4.1 Benchmark 与任务选择

基模采用 Qwen3-VL-4B + GR00T，在 LIBERO 上完成第一阶段，再扩展到 RoboTwin。

LIBERO 用于标准化比较和快速消融，建议覆盖：

- LIBERO-Spatial: 检验相对空间关系、左右/前后/遮挡等能力；
- LIBERO-Object: 检验物体识别与操作泛化；
- LIBERO-Goal: 检验目标条件下的动作选择；
- LIBERO-Long: 检验长程任务和未来状态推演。

RoboTwin 用于检验更复杂的双臂、多物体、长程和物理交互任务，重点选择需要精细空间关系和未来状态变化的任务，例如放入容器、打开/关闭抽屉、按按钮、绕障碍抓取、多步骤整理等。

除标准 benchmark 外，建议构造 OOD evaluation split：

- 物体位置扰动；
- 相机视角扰动；
- 遮挡和 distractor 增加；
- 语言指令 paraphrase；
- 初始状态与训练集不同但目标状态相同；
- 需要先改变场景再完成目标的长程任务。

### 4.2 Baselines

为了证明改进不是来自“更多 token”或“更大容量”，baseline 需要包含容量控制和 teacher 类型控制。

| Baseline | 目的 |
|---|---|
| Qwen3-VL-4B + GR00T | 主基线，验证在同一 action head 和训练协议下的提升 |
| Base + random typed tokens | 控制 token 数量和参数量，排除单纯扩容 |
| DualCoT-VLA-style visual/linguistic query tokens | 对比文本/视觉 CoT 内化范式 |
| Spatial-only: Base + $S$ | 测试空间教师贡献 |
| Future-only: Base + $F$ | 测试未来教师贡献 |
| Action-plan-only: Base + $P$ | 测试潜动作教师贡献 |
| $S+F$ without $P$ | 测试空间到未来推演是否足够 |
| $F+P$ without $S$ | 测试没有空间 grounding 时未来与动作规划是否退化 |
| $S+P$ without $F$ | 测试动作规划是否需要显式未来状态中介 |
| Full Space2Act-VLA | 完整三通道方案 |

如果实现资源允许，可以加入更强对比：

- Spatial alignment with DINO/VGGT/Depth features instead of world-model features: 证明世界模型中间层优于普通几何/视觉特征；
- BRIDGE-WA final-layer target vs middle-layer target: 验证中间层动态表征优于最终重建状态；
- Moto quantized token target vs pre-quantization continuous latent: 验证连续潜动作表征更适合控制。

### 4.3 Ablation 设计

**Token 设计消融**

- $N_s, N_f, N_p$ 的数量扫描，例如 4/8/16/32；
- 三类 token 是否共享 embedding；
- 是否使用 role type embedding；
- latent token 放置顺序：`V-S-L-F-P` vs `V-L-S-F-P` vs 并列无顺序；
- block-wise causal mask vs full attention；
- 同 block 内双向 self-attention vs 严格 causal attention。

**教师特征消融**

- 空间教师层选择：浅层、中层、深层、多层融合；
- 未来教师层选择：BRIDGE-WA/Wan 最终层 vs 中间层；
- future horizon: 短期、中期、长期、多尺度；
- 使用教师 predicted future features vs 使用真实未来帧 encoder features；
- Moto teacher 使用量化后 token vs 量化前连续 latent；
- teacher feature resampler: mean pooling vs Perceiver Resampler vs query transformer。

**损失与训练策略消融**

- 去掉 $\mathcal{L}_{SF}$；
- 去掉 $\mathcal{L}_{FP}$；
- 去掉 $\mathcal{L}_{role}$；
- cosine-only vs MSE-only vs cosine+MSE；
- joint training vs three-stage curriculum；
- teacher dropout 的比例；
- 只训练 token/projector vs LoRA 微调 vs 全量微调。

### 4.4 评估指标

主指标：

- Success rate；
- Subtask completion rate；
- Contact/grasp success rate；
- Collision / constraint violation rate；
- Action smoothness；
- Average trajectory length；
- Inference latency；
- GPU memory / throughput。

泛化指标：

- OOD spatial generalization；
- OOD viewpoint generalization；
- Distractor robustness；
- Language paraphrase robustness；
- Long-horizon success degradation。

潜在推理诊断指标：

- Spatial relation probe: 用 $H_S$ 线性预测 left/right/behind/in-front-of、depth order、目标-夹爪距离等级；
- Future-change probe: 用 $H_F$ 预测变化区域、目标未来状态、是否成功接触；
- Action-latent probe: 用 $H_P$ 预测高层 motion class、waypoint、gripper open/close phase；
- Token masking: 分别 mask $S/F/P$ token，观察 success rate 下降；
- Counterfactual token swap: 交换不同任务的 $F$ 或 $P$ token，验证 token 是否确实携带未来/动作计划信息；
- Attention-to-change alignment: 检查 $F \rightarrow S$ attention 是否集中于教师变化区域。

### 4.5 关键假设与预期结果

**H1: 空间世界模型中间层监督优于普通视觉特征监督。**  
预期在 LIBERO-Spatial、视角扰动和遮挡任务上，$S$ token 对齐 LingBot/GeoWorld-style 中间层会优于 DINO/VGGT/普通视觉 encoder 特征。

**H2: 未来教师中间层优于最终层。**  
预期中间层对齐在长程任务、抽屉/容器/按钮等状态变化任务上更稳定；最终层可能在 future image probe 上更好，但 action success 不一定更高。

**H3: 连续潜动作表征优于离散 motion token。**  
预期量化前 Moto/IDM latent 对 action smoothness、接触成功率和小样本泛化更有帮助。

**H4: 三通道组合优于任意单通道。**  
空间 token 提供“哪里可操作”，未来 token 提供“会变成什么”，动作 token 提供“如何移动”。三者缺一时，性能提升应集中在部分任务；完整模型应在空间、长程和精细操作上都有提升。

**H5: 自定义 block-wise causal routing 可能优于原生 causal routing。**  
当前实现以原生 Qwen causal attention 为基线。只有后续 block-wise routing 在相同 token、teacher 和训练预算下稳定更好，才能把收益归因于额外 routing 归纳偏置。

**H6: teacher-free inference 能保持低延迟。**  
训练阶段读取离线教师特征并执行 mapper，但推理阶段只增加固定数量 latent token，mapper 分支完全跳过。相比需要外部 world model rollout 或显式 CoT 生成的方法，额外延迟只来自 Qwen 序列中的少量 token。

## 5. 论文贡献写法

可以将贡献组织为三点。

**Contribution 1: Typed latent reasoning for VLA.**  
提出一种面向具身控制的三通道潜在推理范式，将 VLA 内部 reasoning token 明确分解为空间位置感知、未来状态推演和潜在动作规划三种角色，避免文本 CoT 的低密度和弱 grounding 问题。

**Contribution 2: Multi-world-model distillation without inference-time teachers.**  
提出将空间探索世界模型、具身未来预测世界模型和潜动作教师的能力蒸馏进 VLA latent tokens，使模型在推理时不依赖外部 world model 或显式 CoT，也能自主形成未来状态推演。

**Contribution 3: Causal latent routing from space to future to action.**  
通过 typed token 的 causal 排布与独立训练期 expansion mapper，使空间、未来和动作计划监督能够在同一 Qwen hidden sequence 中逐阶段加入；block-wise mask 与跨角色一致性损失作为后续增强验证。

一句话 novelty 可以写成：

> We shift latent reasoning for VLA from language-chain imitation to world-dynamics internalization, using typed latent tokens as a teacher-free interface between spatial structure, imagined future state, and continuous motor planning.

## 6. 最强质疑与回应

| 质疑 | 回应 |
|---|---|
| 这是否只是增加了更多 learnable tokens？ | 设置同等 token 数量的 random latent baseline，并做 token masking、role probe、teacher source replacement。若完整模型只比 random token 略好，则说明方法失败；若三类 probe 分别有效且 mask 后出现对应能力下降，则说明 token 学到了角色语义。 |
| 训练时使用未来帧或未来教师是否信息泄漏？ | Student 推理时只接收当前观测、指令和本体状态。未来观测只用于教师预训练或训练阶段监督，等价于 auxiliary future prediction target。实验中应明确 teacher target 是训练标签，不是 inference input。 |
| 为什么不用世界模型在线 rollout？ | 在线 rollout 成本高、延迟大，且容易把规划能力外包给教师。本方案目标是把多个教师的能力内化为 VLA 的 latent computation，使部署时仍保持单次或近似单次前向推理。 |
| 为什么对齐中间层而不是最终层？ | 最终层更接近图像重建，包含渲染和纹理细节；中间层更可能保留任务条件下的变化方向、接触区域和物体交互关系。该观点需要通过 final-layer vs middle-layer 消融验证。 |
| 三个教师信号是否会冲突？ | 使用课程学习、独立 projector、loss ramp-up 和 teacher dropout 降低冲突。若冲突明显，可引入 uncertainty weighting 或只在不同训练阶段激活不同 teacher。 |
| Moto 的 motion latent 和机器人 action 是否不完全一致？ | 本方案不把 Moto latent 当成最终动作，而是作为高层运动计划监督。最终 action 仍由 GR00T flow-matching action loss 决定，Moto 只提供连续 motion prior。 |
| 世界模型教师可能 hallucinate future。 | 教师必须在具身轨迹上预训练，并通过 future-change probe 和 action success 验证。训练时可使用 teacher confidence、multi-horizon consistency 或真实未来帧 encoder feature 作为辅助锚点。 |

## 7. 最小可行实验路线

两周 pilot 可以这样做：

1. 复现 Qwen3-VL-4B + GR00T 在 LIBERO 上的训练和评估。
2. 注册三类 token、使用原生 causal 排布，但先不接教师，跑 random-token baseline。
3. 先做 $S$ token，对齐空间世界模型或 GeoWorld-style 中间层，评估 LIBERO-Spatial 与 spatial probe。
4. 预训练或准备 Wan/BRIDGE-WA-style future teacher，缓存中间层 future features，加入 $F$ token。
5. 预训练或准备 Moto/IDM continuous latent teacher，加入 $P$ token。
6. 跑最小矩阵：Qwen3-VL-4B + GR00T base、random latent、$S$ only、$F$ only、$P$ only、$S+F+P$。
7. 报告 success、latency、token masking、spatial/future/action probe，判断是否进入完整 LIBERO + RoboTwin 评估。

如果资源受限，优先顺序建议为：

```text
Qwen3-VL-4B + GR00T baseline
-> random same-token baseline
-> S token spatial teacher
-> S+F token future teacher
-> full S+F+P
```

因为空间和未来是论文主线，Moto/IDM 的动作规划 token 是增强 novelty 和控制相关性的关键，但实现成本更高。

## 8. 方案命名建议

推荐名称：

- **Space2Act-VLA**: 强调从空间推理到动作规划；
- **TriWorld-Latent VLA**: 强调三个世界/动作教师源；
- **World2Latent-VLA**: 强调世界模型能力内化；
- **SFA-VLA**: Spatial-Future-Action Latent Reasoning for VLA。

其中 **Space2Act-VLA** 最适合作为论文标题中的方法名，因为它直接表达了因果链条：空间结构不是最终目标，而是为了未来推演和动作生成服务。

## 9. 主要参考本地材料

- `Latent Reasoning/DualCoT-VLA Visual-Linguistic Chain of Thought via Parallel Reasoning for Vision-Language-Action Models.pdf`
- `Latent Reasoning/GeoWorld-VLM Geometry from World Models for Vision-Language Models.pdf`
- `Latent Reasoning/Bridge-WA Predicting Where and How the World Changes for Robotic Action.pdf`
- `Latent Reasoning/Moto Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos.pdf`
- `Latent Reasoning/PearlVLA Progressive Embodied Action-Plan Refinement in Latent Space.pdf`
- `Latent Reasoning.docx`
- `latent_reasoning_vla_research_plan.md`
