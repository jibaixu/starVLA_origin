# Space2Act 两阶段训练

本目录的两阶段路径将 Space2Act 训练拆为 VLM 潜在蒸馏和冻结 VLM 的 Action Head 训练。
二者均复用 LIBERO LeRobot 数据、图像/语言输入、episode/frame 对齐和教师 latent cache。

联合训练新增的 Causal Dynamics Bottleneck（CDB）不属于本两阶段路径：它需要 action loss
与 S/P/F 三路 hidden 同时存在，因此只在 `QwenSpace2ActGR00T` 的 Stage 3 joint script 中
通过 `dynamics_bottleneck.enabled=true` 开启。两阶段基线保持关闭，用于区分冻结 VLM 的
性能影响与联合动力学约束的收益。

## 阶段一：VLM 潜在蒸馏

`QwenSpace2ActVLM` 只构造 Qwen、S/P/F typed token 和三路 alignment mapper；不构造
GR00T Action Head，也不使用样本中的 action 标签。它与旧的端到端
`QwenSpace2ActGR00T` 不共享模型实现。Qwen 的因果序列固定为：

```text
images -> S -> instruction -> P -> F
```

唯一的 VLM YAML 默认使用 fixed 权重，且不启用 ramp、Confidence-Aware 或自适应权重：

```bash
bash examples/simBenchmarks/LIBERO/train_files/run_space2act_vlm_distill.sh
```

每种创新由独立 shell 脚本通过 CLI 覆盖这同一份 YAML；可用 `RUN_ID` 覆盖默认实验名：

| 启动脚本 | 启用能力 |
| --- | --- |
| `run_space2act_vlm_distill.sh` | fixed，使用每个 channel 的 `loss_weight` |
| `run_space2act_vlm_progressive.sh` | S/P/F progressive knots |
| `run_space2act_vlm_gradnorm_ramp.sh` | GradNorm + cosine ramp |
| `run_space2act_vlm_confidence_aware_gradnorm_ramp.sh` | Confidence-Aware + GradNorm + cosine ramp |

VLM-only 的总损失仅由三路 alignment 组成。GradNorm 因而使用总预算 `3.0`、权重范围
`[0.25, 2.0]`，以 fixed 基准中三路各 `loss_weight=1.0` 为尺度；这与旧端到端训练为
action loss 预留主导权时的 `auxiliary_budget=0.30` 不同。Confidence-Aware 仍只缩放每个
样本的 alignment loss，不改变 GradNorm 权重预算。

GradNorm 的状态保存在定期 checkpoint 同名的 `_adaptive.json` sidecar 中，resume 时自动恢复。
两阶段配置不包含 PCGrad、evidence routing 或 DWA。VLM 阶段仍可使用 fixed、progressive、
GradNorm/ramp 与 Confidence-Aware alignment，因为它们只作用于三路蒸馏损失。

## 阶段二：冻结 VLM 训练 Action Head

阶段二构造独立的 `QwenSpace2ActActionHead`，仅从阶段一 checkpoint 部分加载
`qwen_vl_interface`，然后冻结该模块。S/P/F token 保留在完整 cross-attention context 中；
alignment mapper、教师 cache、PCGrad、router 与所有蒸馏调度均不会构造或读取。

```bash
VLM_CHECKPOINT=playground/Checkpoints/space2act_libero_vlm_distill_spf/final_model/pytorch_model.pt \
  bash examples/simBenchmarks/LIBERO/train_files/run_space2act_action_head.sh
```

加载前会比较 VLM stage 的 `config.full.yaml` 与当前 action 配置中的 base VLM、typed-token
数量和 `token_order`。移动 checkpoint 时需一并保留其 run root 下的 `config.full.yaml`。

## 配置关系

`space2act_vlm_distill_libero.yaml` 与 `space2act_action_head_libero.yaml` 是各自完整的配置，
不再继承或裁剪旧端到端 YAML。VLM 配置保留原有教师数据契约：默认逐 episode
`.safetensors`，也可将每个 channel 的 `packed_relative_dir` 指向
`manifest.json + features.bin`（可选 `confidence.bin`）的 mmap-packed 缓存。Action Head
配置不含 `teacher_latents`，因此不要求任何教师文件存在。

## 原始 Qwen Action Head Baselines

为区分 action head 本身、固定 token scaffold 和 VLM 蒸馏的贡献，提供两套同 seed（42）的
baseline。二者都冻结 `qwen_vl_interface`，只训练 `action_model`，不加载阶段一 checkpoint：

```bash
bash examples/simBenchmarks/LIBERO/train_files/run_action_head_raw_qwen_baseline.sh
bash examples/simBenchmarks/LIBERO/train_files/run_action_head_raw_qwen_spf_baseline.sh
```

`run_action_head_raw_qwen_baseline.sh` 使用 `QwenGR00T`，输入只有图像和 instruction，不注册
S/P/F token，是主要的原始 Qwen baseline。`run_action_head_raw_qwen_spf_baseline.sh` 使用
`QwenSpace2ActActionHead`，从原始 Qwen 词表均值初始化并冻结 S/P/F token；它不包含蒸馏语义，
但可测量固定 token scaffold 对 causal hidden states 和 action head 的影响，因此作为辅助消融。

两条脚本均支持通过环境变量覆盖 `BASE_VLM`、`LIBERO_DATA_ROOT`、`RUN_ROOT_DIR`、`RUN_ID`、
`MAX_TRAIN_STEPS`、`SAVE_INTERVAL`、`NUM_PROCESSES`，并可设置 `DRY_RUN=1` 检查展开后的命令。
产出的 checkpoint 可直接填入 `run_auto_eval_libero.sh` 的 `CKPT_LIST`。
