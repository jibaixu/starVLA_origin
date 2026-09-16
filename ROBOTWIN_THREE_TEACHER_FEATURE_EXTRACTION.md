# RoboTwin 三教师特征提取说明

本文档记录 Space2Act 在 RoboTwin 上使用 Spatial、Future、Plan 三个教师的特征提取约定、启动脚本、输出尺寸和时间覆盖方式。内容以当前仓库和三个教师项目中的实现为准。

## 1. 时间约定

RoboTwin 视频通常为 15 FPS。动作头仍预测 50 个控制步；当前推荐用 96 帧的 Future 窗口和 96 帧的 Plan 窗口，并通过较稀疏的 anchor 共享控制存储量：

```text
50 steps / 15 FPS = 3.33 seconds
96 frames / 15 FPS = 6.40 seconds (Future)
96 frames / 15 FPS = 6.40 seconds (Plan)
```

必须区分：

| 概念 | 含义 | RoboTwin 设置 |
| --- | --- | --- |
| `action_horizon` | 学生动作头一次预测的动作数量 | `50` |
| teacher temporal coverage | 一个教师目标对应的未来时间范围 | Future `96` 帧，Plan `96` 帧 |
| `share_stride` | 多久缓存一个教师 anchor | Spatial `16`，Future `16`，Plan `16` |

`action_horizon=50` 不要求教师特征变成 50 个 token。对 Future，时间合同满足 `96 - (16 - 1) >= 50`，保证最差复用位置仍覆盖动作窗口；Spatial 只需与 Future 共享 anchor，Plan 按自身潜动作时间合同覆盖 96 帧。教师 hidden 的原始维度也不必保留：本方案在抽取时使用共享的 token-wise PCA，将 Spatial `5120` 和 Future `3072` 压到 `1024`，以显著降低磁盘占用；token 位置不混合，模型端 mapper 的输入 shape 随之更新。

按“取不晚于当前帧的最近 anchor”加载时，三路特征最多被复用 `15` 帧（约 1 秒）。三路必须使用同一组 anchor `0,16,32,...`，从而保证 CDB 中的 S/P/F 监督对应同一个当前状态。Spatial 只提供该 anchor 的随机相机轨迹空间证据，不要求内部时间跨度与 Future 相同；Future 在最差复用位置仍有 `96-15=81` 帧有效覆盖。

### 1.1 CDB 共享 anchor 约束

CDB 的残差关系是学生侧共享 dynamics space 中的结构方程：`z_F_hat = z_S + gated_delta(z_S, z_P, q)`。这里的“Spatial + Plan = Future”不是对三份 teacher tensor 做原始维度相加，而是要求三路 teacher target 来自同一 episode、同一 `anchor_frame` 和同一当前观测。正式合同为：Spatial/Future/Plan 的 `share_stride` 均为 `16`；Spatial 只对齐 anchor 当前帧；Future 与 Plan 从同一 anchor 分别覆盖 96 帧。

## 2. 数据和统一 manifest

RoboTwin 数据根目录为：

```text
/ssdfs/datahome/u26101/jbx/Datasets/RoboTwin
```

数据按两个 domain 组织：

```text
RoboTwin/
├── Clean/
└── Randomized/
```

每个任务目录是独立的 LeRobot 数据集，包含 `meta/info.json`、`meta/episodes.jsonl` 和视频文件。三个教师共用：

```text
tools/build_robotwin_teacher_manifest.py
```

生成命令：

```bash
python tools/build_robotwin_teacher_manifest.py \
  --data-root /ssdfs/datahome/u26101/jbx/Datasets/RoboTwin \
  --output /ssdfs/datahome/u26101/jbx/Datasets/RoboTwin/robotwin_teacher_manifest.jsonl
```

manifest 每行对应一个 episode，包含 `task_id`、`domain`、`suite`、`dataset_root`、`episode_index`、`episode_length`、`fps`、`prompt`、视频路径和 video key。这样三个教师使用相同的 episode、anchor 和任务指令。

当前教师主要使用视频 key：

```text
observation.images.cam_high
```

训练入口和数据加载器仍使用 RoboTwin 原始 LeRobot 数据，不修改源视频和 episode 元数据。

### 2.1 受限存储下的 PCA 流程

Spatial 和 Future 不再把原始 hidden 维度写入全量最终 cache。两个教师启动脚本都遵循同一个两阶段流程。RoboTwin calibration manifest 按 `domain + task` 分层，每层固定抽取 8 个 episode，避免单个任务或 domain 主导投影：

1. 从每个 domain/task 分层中确定性抽取 8 个 calibration episode；若某层不足 8 个则终止并报告缺失分层。
2. 在这些 episode 上抽取原始 cache，并用 episode-balanced anchor/token sampling 拟合 token-wise PCA，保存 `mean/components` 和 `projection_sha256`。
3. 将 calibration cache 转换为 1024 维并删除临时原始文件。
4. 全量抽取时通过 `--projection` 在线投影，直接写入最终目录。

示例（实际路径由对应启动脚本定义）：

```bash
python tools/compress_teacher_latents.py fit \
  --input-root /ssdfs/datahome/u26101/jbx/Datasets/RoboTwin \
  --directory-name lingbot_latent_robotwin_calibration_stride16_samples4_confv1 \
  --projection /ssdfs/datahome/u26101/jbx/Datasets/RoboTwin/lingbot_robotwin_tokenwise_pca1024.safetensors \
  --output-dim 1024 --sample-rows 200000 \
  --sampling-mode episode-balanced --anchors-per-episode 2 --seed 42
```

`token-wise` 表示每个空间/时间 token 独立做同一个 `D→1024` 投影，不在 token 之间做 pooling；因此 `1521` 和 `1125` 的 token 几何保持不变。输出 metadata 必须包含 `feature_transform=tokenwise_pca_v1`、源/目标维度和 projection hash。

## 3. Spatial 教师

### 3.1 入口和参数

项目和启动脚本：

```text
/home/jbx/Codes/lingbot-world/run_extract_robotwin_latents.sh
/home/jbx/Codes/lingbot-world/run_extract_robotwin_calibration_multinode.sh
/home/jbx/Codes/lingbot-world/run_extract_robotwin_full_online_pca_multinode.sh
/home/jbx/Codes/lingbot-world/run_finalize_robotwin_confidence.sh
```

核心抽取程序：

```text
/home/jbx/Codes/lingbot-world/tools/extract_lingbot_latents.py
```

`run_extract_robotwin_latents.sh` 保留了单次 calibration、PCA transform 和全量抽取的
参考命令；当前 8 卡/多节点生产流程应使用下面两个专用 launcher，再使用 confidence
finalize launcher。不要直接把该参考脚本中连续覆盖的 `CALIB_MANIFEST` 赋值或旧的
`torchrun` GPU 配置当作多节点配置。

关键参数：

```text
--share-stride 16
--frame-num 9
--layer 24
--num-denoise-steps 4
--confidence-aware
--num-teacher-samples 4
--video-key observation.images.cam_high
```

### 3.2 特征尺寸和时间含义

LingBot 原始输出为：

```text
[1521, 5120]
```

其中 `1521=39×39` 是固定空间 latent 网格，`5120` 是第 24 层教师 hidden dimension。

`frame_num=9` 是 LingBot 内部的固定相机/空间探索轨迹长度，不是 RoboTwin 的 50 个控制动作。最终训练缓存使用 token-wise PCA 压缩为：

```text
[1521, 1024]  (BF16)
```

`share_stride=16` 表示每 16 个 RoboTwin 视频帧建立一个教师 anchor，并与 Future、Plan 共用 anchor 网格。

Spatial 教师提供当前场景的物体关系、空间布局和可达区域证据，不负责单独编码未来 50 步动作。

### 3.3 Spatial 的 PCA 和多节点抽取流程

Spatial 的抽取必须分成 calibration 和 full extraction 两个阶段。calibration
阶段只用于拟合 token-wise PCA；全量阶段使用已经拟合好的 projection 在线压缩，
不再保存 5120 维原始 hidden。

#### 3.3.1 Calibration manifest 和 PCA

calibration manifest 由全量 manifest 按 `domain + task` 分层确定性抽取，每个分层
固定选择 8 个 episode。推荐先生成：

```bash
/share/home/usersht/.conda/envs/lingbot-world-fast/bin/python \
  /home/jbx/Codes/lingbot-world/tools/build_robotwin_teacher_manifest.py \
  --data-root /ssdfs/datahome/u26101/jbx/Codes/lingbot-world/playground/RoboTwin_part \
  --output /ssdfs/datahome/u26101/jbx/Codes/lingbot-world/playground/RoboTwin_part/robotwin_teacher_calibration_manifest_8pergroup.jsonl \
  --episodes-per-group 8 \
  --seed 42
```

如果需要多节点并行，可用 `split_robotwin_manifest.py` 将这个 calibration manifest
切成 8 个 part。每个 part 只负责不同的 episode；不能在不同 part 上分别拟合 PCA，
也不能用单个 task 的特征拟合最终 projection。

在 4 个节点上启动 8 个独立抽取任务：

```bash
/home/jbx/Codes/lingbot-world/run_extract_robotwin_calibration_multinode.sh
```

该 launcher 的资源约定为：

| 配置 | 值 |
| --- | --- |
| 节点数 | `4` |
| 每节点任务数 | `2` |
| 总抽取任务数 | `8` |
| 每个任务 GPU 数 | `4` |
| 每个任务 torchrun 进程数 | `4` |
| master port | `BASE_PORT + SLURM_PROCID` |

Slurm 在一个节点上可能把 8 张卡同时暴露给两个 task。launcher 会根据
`SLURM_LOCALID` 将前 4 张卡分给 local task 0，将后 4 张卡分给 local task 1，
再启动本地 `torchrun --nproc_per_node=4`。因此每个 task 只能看到自己负责的 4 张卡。
每个 task 使用独立的 `master_port`，地址为 `127.0.0.1`，不会与同节点上的另一个
torchrun 冲突。

calibration 抽取时仍然启用：

```text
--confidence-aware
--num-teacher-samples 4
```

但必须使用：

```text
--episode-start 0
--episode-end 2147483647
```

这两个参数的作用是明确告诉抽取器当前是分片运行，阻止它尝试对局部 manifest
自动计算全局 confidence。它们不会改变 manifest 中实际要抽取的 episode。

所有 calibration part 完成并检查 cache 存在后，在单节点执行一次 PCA 拟合：

```bash
/share/home/usersht/.conda/envs/lingbot-world-fast/bin/python \
  /ssdfs/datahome/u26101/jbx/Codes/starVLA_origin/tools/compress_teacher_latents.py fit \
  --input-root /ssdfs/datahome/u26101/jbx/Codes/lingbot-world/playground/RoboTwin_part \
  --projection /ssdfs/datahome/u26101/jbx/Codes/lingbot-world/playground/RoboTwin_part/lingbot_robotwin_tokenwise_pca1024.safetensors \
  --output-dim 1024 \
  --sample-rows 200000 \
  --sampling-mode episode-balanced \
  --anchors-per-episode 2 \
  --seed 42 \
  --directory-name lingbot_latent_robotwin_calibration_stride16_samples4_confv1
```

PCA 的采样是 episode-balanced：每个 episode 默认最多抽 2 个 anchor，再从这些
anchor 的 token 中采样，避免长 episode 或单个 task 主导 projection。拟合成功后，
将 calibration raw cache 转换成 `[1521, 1024]`，确认转换结果无误后删除临时的
`lingbot_latent_robotwin_calibration_stride16_samples4_confv1` 目录。

#### 3.3.2 全量在线 PCA 抽取

PCA projection 拟合完成后，在 4 个节点上启动全量抽取：

```bash
/home/jbx/Codes/lingbot-world/run_extract_robotwin_full_online_pca_multinode.sh
```

该 launcher 使用全量 `robotwin_teacher_manifest.jsonl`，再次均衡切成 8 个 part，
并为每个 episode 在线执行：

```text
LingBot hidden [1521, 5120]
    -> token-wise PCA
    -> final feature [1521, 1024] (BF16)
```

启动参数与 calibration 多节点任务相同：每节点 2 个 task、每 task 4 张 GPU、
每 task 4 个 torchrun 进程。full launcher 会在抽取结束后校验全量 manifest 对应的
cache 是否全部存在，但不会计算 confidence。

不得在 8 个 full extraction part 上分别计算 confidence，也不得把不同 part 的
disagreement 分别归一化；confidence 必须等待全量 cache 都完成后统一计算。

### 3.4 Spatial confidence 的计算规则

每个 anchor 使用 4 次 teacher sampling。抽取阶段保存：

```text
features       # PCA 后的 teacher feature
disagreement   # 每个 anchor 一个 uncertainty scalar
confidence     # 临时占位值 1.0，待 finalize 覆盖
```

抽取阶段 metadata 为：

```text
confidence_aware=true
confidence_version=v1
calibration=pending
uncertainty_metric=pairwise_cosine_disagreement
```

抽取完成后，在单节点运行：

```bash
/home/jbx/Codes/lingbot-world/run_finalize_robotwin_confidence.sh
```

该步骤要求全量 manifest 中每个 `(domain, task_id)` 分组的所有 episode cache 都存在，
并由 LingBot 抽取器按分组独立收集 disagreement、计算自己的 `q05` 和 `q95`。
Clean 与 Randomized 不共享 quantile；同一 domain 下不同 task 也不共享 quantile。
三个教师均在各自抽取器内部完成该校准，不能使用跨教师的通用后处理器。

```text
confidence = clamp(
    1 - (disagreement - q05) / (q95 - q05 + 1e-8),
    0,
    1,
)
```

因此 Randomized 场景不会改变 Clean 或其他 task 的归一化范围，confidence 只表达
对应 domain/task 分组内的相对可靠性。

finalize 会原子地回写每个 episode cache，并更新 metadata：

```text
calibration=train_quantile_05_95
calibration_scope=task_training_episodes
calibration_domain=clean
calibration_task_id=<task_id>
```

或：

```text
calibration=train_quantile_05_95
calibration_scope=task_training_episodes
calibration_domain=randomized
calibration_task_id=<task_id>
```

同时在最终 feature 目录生成：

```text
confidence_manifest.json
```

其中按 `domain/task_id` 记录每个分组的 episode 数、anchor 数和 `q05/q95`。只有所有
分组完成该步骤后，Spatial cache 的 confidence 才是最终可用于训练和分析的值。Future
和 Plan cache 同样遵循各自抽取器的 task-level manifest。

### 3.5 输出和 metadata

confidence-aware 输出目录：

```text
/ssdfs/datahome/u26101/jbx/Datasets/RoboTwin/lingbot_latent_robotwin_stride16_samples4_confv1_pca1024
```

训练配置检查：

```text
dataset_format=robotwin
fps=15
share_stride=16
layer=24
spatial_tokens=1521
feature_dim=1024
source_feature_dim=5120
feature_transform=tokenwise_pca_v1
```

## 4. Future 教师

### 4.1 入口和参数

项目和启动脚本：

```text
/home/jbx/Codes/DiffSynth-Studio/examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B-ROBOTWIN.sh
```

脚本依次构建 manifest、训练 Wan LoRA、验证并选择 checkpoint、抽取 confidence-aware Future cache。

关键参数：

```text
--num-frames 17
--temporal-stride 6
--train-stride 1
--val-stride 1
--share-stride 16
--confidence-aware
--num-teacher-samples 4
```

### 4.2 保持 token 数、压缩 hidden 并扩大时间覆盖

Future 原始目标尺寸为：

```text
[1125, 3072]
```

最终训练缓存通过 token-wise PCA 压缩为 `[1125, 1024]`，不混合 token 位置。

Wan 的 17 帧输入经过时间 VAE 后对应 5 个时间 latent，每个时间 latent 的空间网格为 `15×15`：

```text
5 × 15 × 15 = 1125 tokens
```

时间采样使用：

```text
anchor + [0, 6, 12, 18, ..., 90, 96]
```

所以：

```text
(17 - 1) × 6 = 96 frames
```

输出 token 数保持 `1125`，压缩后 hidden dimension 为 `1024`。`share_stride=16` 控制 anchor 密度，`temporal_stride=6` 控制一个 Future 目标内部的时间间隔。覆盖范围 `96 - (16 - 1) = 81` 帧，足够覆盖 50 步动作窗口。

### 4.3 真实稀疏 clip 输入

抽取器会按 `anchor + [0,6,...,96]` 从原视频读取 17 张真实帧，并调用：

```python
pipe.extract_dit_feature(input_video=clip, num_frames=17)
```

TI2V 模型同时使用 clip 的首帧作为 fused image condition；其余帧通过 VAE video latent 进入 Wan。训练 manifest、训练数据集和 extraction 使用同一个 `temporal_stride=6` 合同，尾部不足窗口时采用 repeat-last-frame，并记录 `valid_future_frames`。

### 4.4 输出和 metadata

confidence-aware 输出目录：

```text
/ssdfs/datahome/u26101/jbx/Datasets/RoboTwin/wan2_2_future_robotwin_stride16_tstride6_samples4_confv1_pca1024
```

训练配置检查：

```text
dataset_format=robotwin
fps=15
share_stride=16
temporal_stride=6
effective_future_steps=96
future_horizon_frames=96
num_frames=17
target_tokens=1125
feature_dim=1024
source_feature_dim=3072
feature_transform=tokenwise_pca_v1
```

## 5. Plan 教师

### 5.1 入口和参数

项目和启动脚本：

```text
/home/jbx/Codes/Moto/scripts/finetune_latent_motion_tokenizer_on_robotwin.sh
```

核心抽取程序：

```text
/home/jbx/Codes/Moto/latent_motion_tokenizer/extraction/extract_libero_plan_features.py
```

关键参数：

```text
--segment-delta 16
--segment-count 6
--share-stride 16
--fps 15
--dataset-format robotwin
--confidence-aware
--num-teacher-samples 4
```

### 5.2 保持输出 shape、扩大 segment 覆盖

Plan 输出使用六个与 Moto 原生一致的 delta-16 segment：

```text
[48, 32]
```

每个 segment 保留 8 个 pre-VQ latent，每个 latent 维度为 32：

```text
6 segments × 8 queries = 48 tokens
```

LIBERO 的 delta-4 合同是：

```text
0:4, 4:8, 8:12, 12:16
```

RoboTwin 使用 delta-16 合同：

```text
0:16, 16:32, 32:48, 48:64, 64:80, 80:96
```

总覆盖为：

```text
6 × 16 = 96 frames
```

RoboTwin 保持 Moto 每段 16 帧和每段 8 个 query，只将每个 anchor 的 segment 数扩展到 6，以覆盖 Future 的 96 帧窗口；latent 维度仍为 32。

Moto 必须使用 delta-16 的 motion pair 训练或微调。LIBERO delta-4 checkpoint 可以用于程序 smoke test，但不能直接当作 RoboTwin delta-16 语义下的最终 Plan 教师。

### 5.3 输出和 metadata

confidence-aware 输出目录：

```text
/ssdfs/datahome/u26101/jbx/Datasets/RoboTwin/moto_plan_robotwin_delta16x6_h96_stride16_samples4_confv1
```

训练配置检查：

```text
dataset_format=robotwin
fps=15
segment_delta=16
share_stride=16
segment_count=6
segment_offsets=0:16,16:32,32:48,48:64,64:80,80:96
future_horizon_frames=96
target_tokens=48
feature_dim=32
```

## 6. StarVLA 训练对齐

RoboTwin 三教师配置：

```text
examples/simBenchmarks/Robotwin/train_files/space2act_robotwin.yaml
```

关键合同：

```text
action_horizon = 50
Spatial target = [1521, 1024] (source [1521, 5120])
Future  target = [1125, 1024] (source [1125, 3072])
Plan    target = [48, 32]

Spatial share_stride = 16
Future  share_stride = 16, temporal_stride = 6
Plan    share_stride = 16, segment_delta = 16, segment_count = 6
```

最终训练入口：

```text
examples/simBenchmarks/Robotwin/train_files/run_space2act_cdb_confidence_aware_gradnorm_pcgrad_train.sh
```

该入口沿用 LIBERO 最终方案的训练风格，启用 Causal Dynamics Bottleneck、confidence-aware alignment、GradNorm loss schedule、feature-level PCGrad 和三教师 strict metadata 校验。

学生侧每个通道仍然只有 32 个 compact token。教师目标只在训练期通过 mapper 对齐；推理时不加载教师模型，也不读取教师 cache。

## 7. 提取后检查清单

开始 StarVLA 训练前至少检查：

1. 全量 manifest 中每个 episode 的视频路径存在，episode length 与 LeRobot metadata 一致；calibration manifest 的每个 `domain/task` 分层恰好包含 8 个 episode。
2. 三种 cache 都从 anchor `0` 开始，且 stride 均为 `16`；三者 `anchor_frames` 必须完全一致。
3. Spatial cache 形状为 `[num_anchors,1521,1024]`，metadata 含 `source_feature_dim=5120` 和 `feature_transform=tokenwise_pca_v1`。
4. Future cache 形状为 `[num_anchors,1125,1024]`，metadata 为 `temporal_stride=6`、`effective_future_steps=96`，并含 `source_feature_dim=3072`。
5. Plan cache 形状为 `[num_anchors,48,32]`，metadata 为 `share_stride=16`、`segment_delta=16`、`segment_count=6`、`future_horizon_frames=96`，并与 Spatial/Future 使用相同 anchor_frames。
6. confidence-aware cache 同时包含 `confidence` 和 `disagreement`，数值有限且 confidence 在 `[0,1]`；metadata 含 `calibration_scope=task_training_episodes`、domain 和 task_id。
7. cache metadata 与 `space2act_robotwin.yaml` 的 `expected_metadata` 完全一致。
8. episode 尾部不足完整未来窗口时，确认 clamp/repeat-last-frame padding 策略与训练配置一致。
9. PCA projection metadata 包含 `sampling_mode=episode-balanced`、`anchors_per_episode=2` 和固定 seed，并确认实际采样行数不超过 `sample_rows`。
10. Future 教师额外确认稀疏视频帧实际进入 Wan 输入；仅检查 `temporal_stride` metadata 不能证明这一点。

Confidence 校准在分片抽取阶段不执行。全量 cache 完成后，三个教师分别在各自项目中
运行完整抽取/校准流程：LingBot 使用 `/home/jbx/Codes/lingbot-world/run_finalize_robotwin_confidence.sh`，
Future 和 Plan 使用各自抽取器。所有实现均按 `(domain, task_id)` 汇总 disagreement、
计算独立 q05/q95 并写回对应 task cache；starVLA 不再提供通用 finalize 工具。

## 8. 总结

推荐合同是：Spatial/Future/Plan 均用 `share_stride=16`；Spatial 只对齐 anchor 当前帧，Future 用 17 个稀疏采样帧覆盖 96 帧，Plan 用 6 个 delta-16 segment 覆盖同一 96 帧；Spatial/Future 通过 token-wise PCA 压到 1024 维，Plan 输出 `[48,32]`。


按当前 RoboTwin 全量统计的 `6,075,103` 帧估算，最终 BF16 cache（不含文件系统块开销）约为：

| 教师 | stride | 每 anchor 特征 | 估算存储 |
| --- | ---: | ---: | ---: |
| LingBot Spatial | 16 | `1521×1024×2` = 3.12 MB | 约 1.18 TB |
| Wan Future | 16 | `1125×1024×2` = 2.30 MB | 约 0.88 TB |
| Moto Plan | 16 | `48×32×2` = 3 KB | 约 1.2 GB |
| 合计 | - | - | 约 2.06 TB |

`confidence`、`disagreement` 和 `anchor_frames` 只增加少量空间；`num_teacher_samples=4` 是在线均值统计，不会把完整 feature 保存四份。抽取脚本先在有限 calibration 子集上拟合 PCA，转换并删除临时原始 cache，再对全量 episode 在线投影，因此不会长期同时保留原始 5120/3072 维 cache。

PCA projection 文件本身很小（Spatial 约 `5120×1024×4` 字节，Future 约 `3072×1024×4` 字节），但拟合和抽取必须使用同一个 projection hash。若需要重新拟合，必须生成新的输出目录和对应训练配置，不能混用不同 hash 的 episode。
