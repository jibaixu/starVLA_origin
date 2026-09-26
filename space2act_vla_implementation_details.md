# Space2Act-VLA 实现说明

本文档以仓库当前代码和
`examples/simBenchmarks/LIBERO/train_files/run_space2act_cdb_confidence_aware_gradnorm_pcgrad_train.sh`
为准，目标是让读者无需逐行阅读原始代码即可理解方法、训练目标和工程行为。

当前主实验是 **Stage 3 三通道联合训练**，启用：

- S/F/P 三通道 latent teacher alignment；
- Causal Dynamics Bottleneck（CDB）；
- Confidence-Aware teacher alignment；
- GradNorm 三通道权重平衡；
- action-anchored feature-level PCGrad。

AISER/evidence routing、课程学习、progressive schedule 和 DWA 虽然在仓库中有独立实现，但不属于本训练脚本和本文主方法。

## 第一部分：详细实现方案

### 1. 问题定义与总体架构

给定当前多视角图像 (o_t)、语言指令 (l)、机器人状态 (q_t) 和专家动作 chunk (a^*_{t:t+H-1})，模型使用 Qwen3-VL 作为视觉语言 backbone，使用 GR00T Flowmatching Action Head 预测连续动作：

[
H=operatorname{Qwen}(o_t,l;S,F,P),qquad
hat a=operatorname{GR00T}(H,q_t).
]

S、F、P 是插入 Qwen 输入序列的三类可学习 typed latent tokens：

- **S（spatial）**：表达物体、相机、可达区域和接触区域之间的空间关系；
- **F（future）**：表达语言任务约束下未来场景的状态变化；
- **P（plan）**：表达由空间结构和任务目标决定的高层潜在运动规划。

训练时，三类学生 hidden 分别对齐三个冻结教师的中间特征；动作损失、教师对齐损失和 CDB 结构损失共同更新模型。推理时仅保留 Qwen、typed tokens 和 Action Head，不读取教师特征、confidence 或训练期 mapper。

### 2. Typed latent token 与因果信息流

模型一次性将 S/F/P 三类 token 注册到 tokenizer 和 Qwen embedding。主实验的输入顺序为：

```text
视觉内容 -> S tokens -> instruction -> P tokens -> F tokens
```

Qwen 使用原生 causal attention，不额外构造 block-wise mask。因此后出现的 token 可以读取前面 block 的信息，形成：

[
	ext{visual observation}
ightarrow 	ext{spatial grounding}
ightarrow 	ext{task-conditioned planning}
ightarrow 	ext{future-state prediction}.
]

词表注册和实际插入相互独立：词表始终保留三类 token 的稳定布局，实验通过 `token_enabled` 决定是否插入，从而保证 checkpoint 的 token id 和 embedding shape 稳定。

### 3. 三通道教师蒸馏

对每个通道 (cin{S,F,P})，学生从 Qwen 最后一层取出紧凑 hidden：

[
H_cinmathbb R^{B	imes K_c	imes D_v}.
]

独立 aligner 使用可学习 target queries，通过 cross-attention 将其映射到教师空间：

[
hat Z_c=operatorname{Proj}_c(operatorname{CrossAttn}_c(Q_c,
operatorname{LN}(W_cH_c))).
]

query 数与教师 token 数相同，query 之间不做 self-attention，只从紧凑学生 token 读取信息。大教师目标按 query chunk 分块计算。

单通道对齐损失为 LayerNorm 后的 cosine distance 与 MSE：

[
ell_c=
rac1{N_c}sum_n
[1-cos(operatorname{LN}(hat Z_{c,n}),
operatorname{LN}(Z^*_{c,n}))]
+alpha_coperatorname{MSE}(operatorname{LN}(hat Z_c),
operatorname{LN}(Z^*_c)).
]

S 教师提供空间基底，F 教师提供任务条件下的未来变化，P 教师提供动作生成所需的运动语义。教师特征始终 stop-gradient。

### 4. Causal Dynamics Bottleneck

CDB 只读取学生 S/P/F hidden。首先对每类 token 做 attention pooling：

[
z_S=operatorname{Pool}(H_S),quad
z_P=operatorname{Pool}(H_P),quad
z_F=operatorname{Pool}(H_F).
]

前向结构方程使用 S、P 和 proprio state (q) 预测未来：

[
Delta=f_Delta([z_S,z_P,q]),qquad
g=sigma(f_g([z_S,z_P,q])),
]
[
hat z_F=operatorname{LN}(z_S+godotDelta).
]

hybrid 模式还包括：

1. **逆向结构方程**：由 (z_S,z_F,q) 预测 (z_P)；
2. **P intervention ranking**：在 batch 内置换 P，要求正确 P 的预测更接近真实 (z_F)；
3. **variance floor**：约束 S/P/F 表示的 batch 方差，防止 dynamics space 坍缩。

[
mathcal L_{CDB}=
lambda_{fwd}mathcal L_{fwd}+
lambda_{inv}mathcal L_{inv}+
lambda_{int}mathcal L_{int}+
lambda_{var}mathcal L_{var}.
]

训练初期 CDB 梯度关闭，warmup 后通过 cosine ramp 逐步加入总损失；推理时完全跳过 CDB。

### 5. Confidence-Aware teacher alignment

对同一 anchor/channel 执行多次语义保持的教师推理，多次输出的均值作为最终 target：

[
ar Z_{a,c}=rac1Ksum_{k=1}^{K}Z^{(k)}_{a,c}.
]

对每个 teacher token 计算多次输出之间的 LayerNorm 后 cosine disagreement，沿 token 维聚合为 anchor/channel 级 uncertainty，再通过训练集分位数校准为 (c_{a,c}in[0,1])。

训练时：

[
w_{a,c}=w_{min}+(1-w_{min})c_{a,c},
]
[
mathcal L^{conf}_c=
rac{sum_a w_{a,c}ell_{a,c}}
{sum_a w_{a,c}+epsilon}.
]

有效权重归一化保证不同 batch 的 confidence 分布不会改变 alignment loss 的整体尺度。confidence 只作用于对应 teacher alignment，不改变 action loss、CDB 或推理路径。

### 6. GradNorm 三通道权重平衡

在共享 Qwen 最后一层 hidden 上测量每个 alignment loss 的梯度范数：

[
g_c=left|rac{partialmathcal L^{conf}_c}
{partial H_{last}}ight|_2.
]

梯度范数使用 EMA 平滑，根据其倒数构造目标通道权重，再投影到带上下界的 simplex：

[
sum_clambda_c=B,qquad
lambda_{min}leqlambda_cleqlambda_{max}.
]

GradNorm 只改变 S/F/P alignment 的外层系数，不改变单通道 alignment loss 或 action loss。权重、EMA 梯度、最近梯度和 probe window 都随 checkpoint 保存。

### 7. Action-anchored feature-level PCGrad

令：

[
g_a=rac{partialmathcal L_{action}}{partial H_{last}},qquad
g_c=rac{partial(lambda_cmathcal L^{conf}_c)}
{partial H_{last}}.
]

当二者 cosine 小于阈值时：

[
g'_c=g_c-rac{g_c^	op g_a}
{|g_a|_2^2+epsilon}g_a.
]

实现通过 `last_hidden.register_hook` 修正进入 Qwen backbone 的梯度。alignment head 和 action head 的私有参数梯度保持不变。分布式训练中，内积和范数先 all-reduce，再决定是否投影。

### 8. 总体优化目标

[
mathcal L_{total}=
mathcal L_{action}
+sum_{cin{S,F,P}}lambda_c^{GN}mathcal L_c^{conf}
+lambda_{CDB}mathcal L_{CDB}.
]

Confidence-Aware 改变样本级监督可靠性，GradNorm 改变通道级辅助损失权重，PCGrad 改变共享 hidden 的梯度方向，CDB 提供显式的 S/P 到 F 因果约束。

## 第二部分：具体实现细节

### 1. 代码入口与模块职责

| 模块 | 作用 |
|---|---|
| `QwenSpace2ActGR00T.py` | 模型组装、typed hidden 提取、CDB、alignment、GradNorm、PCGrad、训练和推理 |
| `space2act_alignment.py` | cross-attention mapper 和 cosine + MSE 对齐损失 |
| `space2act_dynamics.py` | CDB 四项结构损失 |
| `teacher_latents.py` | safetensors/packed teacher cache 索引、校验和 anchor 读取 |
| `datasets.py` | 将 teacher feature、confidence、episode/frame 注入训练样本 |
| `train_starvla.py` | optimizer step、梯度累积、GradNorm 状态更新和日志 |
| `space2act_libero_state.yaml` | 基础模型、数据和默认配置 |
| 目标 shell 脚本 | 主实验的全部 dotlist 覆盖参数 |

### 2. 主训练脚本

脚本使用 CUDA 12.4、GCC 11.4、Accelerate、DeepSpeed ZeRO-2 和 8 个进程，调用：

```text
starVLA/training/train_starvla.py
```

训练步数为 100000。默认使用完整 LIBERO 数据集；将 `SPACE2ACT_EPISODES` 改为 `[0,1,2]` 可进行 pilot。脚本会复制自身并将 stdout/stderr 写入 run output 的 `train.log`。

### 3. Typed token 与实际序列

脚本显式设置：

```text
stage = 3
token_order = spatial_instruction_plan_future
token_init_strategy = avg
query_chunk_size = 128
```

三类 token 和三类 alignment 全部启用，每类 32 个 token。Qwen 输入构造在 `starVLA/model/modules/vlm/QWen3.py`：图像之后插入 S，再插入指令，最后按顺序插入 P 和 F。模型通过 token id 精确 gather 每类最后一层 hidden，缺失或重复 token 会直接报错。

### 4. 教师 target 与 aligner

| 通道 | 教师来源 | target shape | 语义 |
|---|---|---:|---|
| spatial | LingBot layer 24 | `[1521, 5120]` | 空间结构 |
| future | Wan2.2 TI2V block 20 | `[1125, 3072]` | 未来状态 |
| plan | Moto pre-VQ latent | `[32, 32]` | 潜动作规划 |

每个通道独立使用 alignment dimension 1024、8-head attention 和 128 query chunk。teacher tensor 留在 CPU，aligner 按块搬运目标。

### 5. Teacher cache

主脚本配置的缓存目录是：

```text
lingbot_latent_step4_numlayer24_stride16_samples4_confv1
wan2_2_future_latent_h16_step20_block20z_stride8_480x480_samples4_confv1
moto_plan_latent_pre_vq_delta4x4_h16_stride1_epoch10_samples4_confv1
```

同时配置对应 packed mmap 目录。每个缓存包含：

```text
features       [A, N, D]
anchor_frames  [A]
confidence     [A]
```

`EpisodeTeacherStore` 使用最近且不晚于当前帧的 anchor。加载器校验 feature shape/dtype、anchor 顺序和 stride、episode length、metadata、confidence shape 以及 confidence 是否为有限的 `[0,1]` 值。confidence-aware 模式缺少 confidence 会直接报错。

dataset 将字段组织为：

```python
sample["space2act_teacher"][channel]
sample["space2act_confidence"][channel]
sample["episode_index"]
sample["frame_index"]
```

### 6. CDB 实际配置

```text
enabled = true
mode = hybrid
dim = 256
forward_weight = 0.05
inverse_weight = 0.03
intervention_weight = 0.02
variance_weight = 0.005
warmup_steps = 5000
ramp_steps = 35000
margin = 0.1
variance_floor = 0.05
```

CDB 的 pooling query 分别对应 spatial、plan、future。forward/inverse 网络是 LayerNorm-GELU MLP。batch size 大于 1 时才启用 P intervention ranking，否则该分量为零。

### 7. Confidence-Aware 实际配置

```text
confidence_aware_alignment.enabled = true
confidence_aware_alignment.min_weight = 0.0
confidence_aware_alignment.require_cached_confidence = true
```

confidence 为 0 的样本不产生该通道 alignment 梯度，confidence 为 1 的样本保留完整监督。实现先计算 `[B]` per-example loss，再进行全局有效权重归一化；分布式环境中有效权重和样本数使用 all-reduce。

### 8. GradNorm 实际配置和状态更新

```text
mode = gradnorm
enabled = true
warmup_steps = 5000
gradient_probe_interval = 20
gradient_update_interval = 200
gradient_ema_decay = 0.9
alpha = 0.0
gradient_epsilon = 1e-8
auxiliary_budget = 0.30
min_weight = 0.05
max_weight = 0.20
adaptive_weight_ramp_steps = 35000
knots = []
```

forward 阶段按 probe interval 输出各通道梯度范数；训练器在同步 optimizer step 调用 `update_adaptive_state`。当 update interval 到达且 probe window 完整时，模型更新 EMA 和 channel weights。因为 `knots=[]` 且 mode 为 `gradnorm`，不会执行 progressive 固定 knot 插值，也不会执行 DWA。

### 9. PCGrad 实际配置和 hook

```text
pcgrad.enabled = true
pcgrad.trigger_cosine = -0.05
pcgrad.warmup_steps = 5000
pcgrad.epsilon = 1e-12
pcgrad.project_router = false
```

PCGrad 在 forward 末尾分别计算 action loss 和每个加权 alignment loss 对 `last_hidden` 的梯度，按全局 cosine 决定是否投影，然后注册 hook。主要日志包括：

```text
pcgrad/<channel>_cosine_before
pcgrad/<channel>_cosine_after
pcgrad/<channel>_applied
pcgrad/<channel>_removed_norm_ratio
pcgrad/<channel>_trigger_rate
pcgrad/total_projection_norm
```

主脚本未启用 evidence router，因此 `project_router=false` 不会排除主实验分支。

### 10. 完整 forward、反向和更新顺序

`QwenSpace2Act_GR00T.forward` 的顺序为：

1. 读取 image、lang、action、state；
2. Qwen 一次前向得到最后一层 hidden；
3. 提取 S/F/P hidden；
4. 计算 CDB；
5. 读取三路 teacher feature；
6. 计算 per-example alignment 并应用 confidence；
7. 使用当前 GradNorm 权重累加 alignment loss；
8. 使用 GR00T Action Head 计算 action loss；
9. 计算 GradNorm probe/diagnostic 并注册 PCGrad hook；
10. 返回分项 loss 和 total loss；
11. trainer 执行 backward、gradient clipping、optimizer step；
12. 同步边界更新 GradNorm 非参数状态并记录指标。

代码中的总损失为：

```python
total_loss = action_loss + total_alignment_loss + dynamics_loss
```

### 11. 推理路径

`predict_action` 只执行：

```text
Qwen forward -> action context -> GR00T action_model.predict_action
```

推理不读取 teacher、confidence，不运行 alignment head、CDB、GradNorm 或 PCGrad。训练期的教师模型和可靠性信息不会增加部署依赖。

### 12. 不属于当前主实验的路径

以下功能不得写入当前主方法：

- evidence routing/AISER：需要 A tokens 和 action-conditioned router；
- progressive fixed knots：主脚本将 `knots` 设为空；
- DWA：主脚本 mode 为 `gradnorm`；
- Stage 1/2 课程迁移：主脚本直接运行 Stage 3；
- VLM-only distillation 和 frozen action-head：属于其他入口。

## 验证与论文报告建议

建议报告：

1. 三类 cache 的 shape、anchor、metadata 和 confidence 合法性；
2. CDB 四项子损失是否有限；
3. confidence 加权在全零、全一和混合 confidence 下的归一化行为；
4. GradNorm 权重和是否保持 auxiliary budget 且满足上下界；
5. PCGrad 投影前后 cosine、触发率和投影范数；
6. cold-start 前向、反向、optimizer step 和 checkpoint 保存；
7. 不提供 teacher tensor 时 checkpoint 能否输出有限 action；
8. 8 卡 ZeRO-2 下 all-reduce、梯度累积和状态恢复是否一致。

论文中可将本实现概括为：多教师 latent supervision 提供互补的空间、未来和运动先验；CDB 在学生 latent 内显式建模空间到未来的因果转移；Confidence-Aware alignment 过滤不可靠 target；GradNorm 平衡三路监督；PCGrad 减少辅助任务与动作目标在共享 backbone 上的梯度冲突。
