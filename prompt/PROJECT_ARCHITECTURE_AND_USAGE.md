# PROJECT_ARCHITECTURE_AND_USAGE

> 本文档只分析当前主项目代码，忽略 `unitraj-flow/`。结论以当前可执行代码为准。

## 1. 项目概览

当前仓库的主路径已经切换为一个**单阶段轨迹生成/恢复系统**：

- 任务：trajectory completion、trajectory prediction
- 训练范式：Flow Matching / Rectified Flow
- 主模型：DiT 风格 Transformer + AdaLN
- 可选能力：SparseMoE、spectral consistency loss

当前真正接入训练和评估的核心文件是：

- `main.py`
- `evaluate_gpt.py`
- `utils/dataset.py`
- `utils/flow_matching.py`
- `utils/config.py`

原始 `UniTraj` 主干 `utils/unitraj.py` 仍保留，但**不在当前主训练/主评估路径中**。

## 2. 整体流程

### 2.1 数据流

pickle DataFrame -> `TrajectoryDataset` -> 重采样 -> 起点平移 -> 标准化 -> padding -> 构造 `loss_mask / observed_mask / observed_trajectory / attention_mask / intervals` -> DataLoader

### 2.2 模型流

`observed_trajectory` + `intervals` + `attention_mask` + `observed_mask` -> 构造 `x_0` -> 与完整轨迹 `x_1` 组成线性 flow 路径 -> `SingleStageTrajectoryFlow` 预测速度场

### 2.3 训练流

`main.py`：

- 读取 `utils/config.py`
- 构建训练/验证集
- 构建 `SingleStageTrajectoryFlow`
- 计算 MSE velocity loss
- 可选叠加 spectral loss
- 按验证损失保存 `best_flow_model.pt`

### 2.4 推理流

`utils/flow_matching.py` 中的 `sample_trajectory_flow()`：

- 起点是“已观测轨迹 + 未观测区随机噪声”
- 用 Euler 积分做 ODE 采样
- 每一步都把已观测点写回，保证 known points 不漂移

### 2.5 评估流

`evaluate_gpt.py`：

- 加载 flow checkpoint
- 为 completion 或 prediction 手工构造掩码
- 调用 `sample_trajectory_flow()`
- 反标准化并加回起点
- 用 geodesic 距离统计 MAE / RMSE

## 3. 核心模块

### 3.1 `utils/dataset.py`

当前数据主入口，负责：

- 读取 pickle 数据
- 轨迹重采样
- 起点平移和标准化
- padding / truncation
- completion / prediction mask 构造
- 训练样本打包

#### 输入数据假设

每条样本至少包含：

- `time`
- `trajectory`

并且代码假定单个点格式为：

- `point[0] = latitude`
- `point[1] = longitude`

#### 返回字段

- `trajectory`: 完整目标轨迹，形状 `[2, L]`
- `observed_trajectory`: 已观测轨迹，未知位置为 0，形状 `[2, L]`
- `attention_mask`: 真实点为 1，padding 为 0
- `loss_mask`: 需要预测的位置为 1
- `observed_mask`: 已知真实点为 1
- `original`: 原始首点，用于反平移
- `intervals`: 时间间隔序列
- `sample_id`: 样本编号

#### mask 策略

训练支持三种模式：

- `completion`
- `prediction`
- `mixed`

其中 `mixed` 下由 `completion_prob` 决定任务采样。completion 内部再混合：

- random mask
- RDP-based mask
- block mask

prediction 固定遮盖最后 `prediction_horizon` 个真实点。

### 3.2 `utils/flow_matching.py`

当前主模型与 flow 核心逻辑都在这里。

#### 主要组件

- `SparseMoE`
- `DiTBlock`
- `FinalLayer`
- `SingleStageTrajectoryFlow`
- `build_flow_source`
- `build_flow_path`
- `compute_spectral_loss`
- `sample_trajectory_flow`

#### Flow Matching 定义

当前代码的定义是：

- `x_0 = observed_trajectory + masked_region_noise`
- `x_1 = target_trajectory`
- `x_t = (1 - t) * x_0 + t * x_1`
- `target_velocity = x_1 - x_0`

这说明当前版本是**单阶段线性路径 flow matching**，而不是 coarse-to-refinement 二阶段方案。

#### 模型输入与输出

输入：

- `x_t`
- `t`
- `observed_trajectory`
- `intervals`
- `attention_mask`
- `observed_mask`

输出：

- 速度场预测 `[B, L, 2]`

#### 条件注入方式

- token 级条件：`observed_trajectory + intervals + [observed_mask, attention_mask]`
- 全局条件：条件 token 的 masked pooling + timestep embedding
- AdaLN 用全局条件调制每个 `DiTBlock`

#### MoE 生效位置

MoE 只替换 `DiTBlock` 的 MLP 分支，不作用于 attention 分支。

#### spectral loss 约束对象

spectral loss 约束的是：

- `pred_velocity`
- `target_velocity`

也就是速度场的频域一致性，而不是直接约束最终轨迹。

### 3.3 `main.py`

当前训练入口，负责：

- 构建数据集和 DataLoader
- 构建 `SingleStageTrajectoryFlow`
- 训练、验证、早停
- 保存 checkpoint
- 创建实验目录

训练时真正参与 early stopping 的是**flow loss**，不是 geodesic MAE。

输出目录格式：

```text
UniTraj/<dataset>_bs=<batch_size>/<timestamp>/
```

关键输出：

- `models/best_flow_model.pt`
- `models/final_flow_model.pt`

### 3.4 `evaluate_gpt.py`

当前评估脚本只覆盖：

- completion
- prediction

主要工作：

- 加载 flow 模型
- 构造 batch 级掩码
- 调用 `sample_trajectory_flow()`
- 计算 geodesic MAE / RMSE
- 输出 JSON 到 `results/`

注意：这里用的是

```python
model.load_state_dict(state, strict=False)
```

所以评估配置如果和训练不一致，可能出现“部分加载成功但实验无效”的情况。

### 3.5 `utils/config.py`

当前采用静态 Python 配置，不是命令行训练配置。

分为三部分：

- `data`
- `model`
- `training`

因此训练方式是：

1. 修改 `utils/config.py`
2. 运行 `python3 main.py`

### 3.6 保留但未接入主路径的模块

#### `utils/unitraj.py`

仍保留原始 UniTraj 风格的：

- `Encoder`
- `Decoder`
- `PatchShuffle`
- RoPE attention

但当前不参与主训练和主评估。

#### `utils/EMA.py`

提供 EMA 工具类，当前未使用。

#### `calculate_norm_params/calculate_norm_params.py`

用于统计标准化参数，但脚本里有硬编码绝对路径，不能直接无修改复用。

## 4. 关键参数

### 4.1 数据参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `train_path` | `./data/worldtrace_sample.pkl` | 训练集路径 |
| `val_path` | `./data/worldtrace_sample.pkl` | 验证集路径 |
| `traj_length` | `200` | 统一序列长度 |
| `mask_ratio` | `0.5` | completion 掩码比例 |
| `task_mode` | `mixed` | 训练任务模式 |
| `completion_prob` | `0.7` | mixed 下 completion 概率 |
| `predict_len` | `8` | prediction 遮盖末尾点数 |

### 4.2 模型参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `hidden_size` | `256` | 隐层维度 |
| `depth` | `6` | DiT block 数 |
| `num_heads` | `4` | 注意力头数 |
| `mlp_ratio` | `4.0` | MLP/专家扩张倍数 |
| `use_moe` | `True` | 是否启用 SparseMoE |
| `num_experts` | `4` | 专家数 |
| `top_k` | `2` | 每个 token 选取专家数 |
| `use_spectral_loss` | `True` | 是否启用 spectral loss |
| `spectral_loss_weight` | `0.1` | spectral loss 权重 |
| `sample_steps` | `16` | Euler 采样步数 |

### 4.3 训练参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `batch_size` | `256` | batch size |
| `n_epochs` | `100` | 最大 epoch |
| `lr` | `1e-4` | AdamW 学习率 |
| `weight_decay` | `1e-4` | 权重衰减 |
| `patience` | `20` | 早停 patience |
| `lr_patience` | `3` | LR scheduler patience |
| `lr_decay_factor` | `0.5` | LR 衰减倍率 |
| `grad_clip` | `1.0` | 梯度裁剪 |
| `device` | `auto` | 自动选 CPU/GPU |

最影响结果的通常是：

- `mask_ratio`
- `task_mode`
- `predict_len`
- `hidden_size`
- `depth`
- `use_moe`
- `num_experts`
- `top_k`
- `sample_steps`
- `spectral_loss_weight`

## 5. 如何使用

### 5.1 环境准备

当前 `requirements.txt` 不是完全可靠的最终版本，至少有两个问题：

- `timm` 被误写成 `timme`
- 多个标准库模块被错误写进 requirements

建议手动安装核心依赖：

```bash
pip install torch numpy pandas matplotlib scikit-learn timm einops colored rdp geopy tqdm
```

### 5.2 数据准备

准备一个 pickle DataFrame，至少包含：

- `time`
- `trajectory`

其中 `trajectory` 的点顺序应与代码假设一致，即 `[lat, lon]`。

### 5.3 训练

先修改 `utils/config.py`，再运行：

```bash
python3 main.py
```

### 5.4 评估

#### Completion

```bash
python3 evaluate_gpt.py \
  --task completion \
  --model_path ./UniTraj/worldtrace_bs=256/<timestamp>/models/best_flow_model.pt \
  --test_data ./data/worldtrace_sample.pkl \
  --mask_ratio 0.5 \
  --sample_steps 16
```

#### Prediction

```bash
python3 evaluate_gpt.py \
  --task prediction \
  --model_path ./UniTraj/worldtrace_bs=256/<timestamp>/models/best_flow_model.pt \
  --test_data ./data/worldtrace_sample.pkl \
  --predict_len 5 \
  --sample_steps 16
```

#### Completion + Prediction

```bash
python3 evaluate_gpt.py \
  --task completion,prediction \
  --model_path ./UniTraj/worldtrace_bs=256/<timestamp>/models/best_flow_model.pt \
  --test_data ./data/worldtrace_sample.pkl \
  --mask_ratio 0.5 \
  --predict_len 5 \
  --sample_steps 16
```

### 5.5 输出内容

训练输出：

- `UniTraj/.../models/best_flow_model.pt`
- `UniTraj/.../models/final_flow_model.pt`

评估输出：

- `results/flow_reconstruction_<timestamp>.json`

## 6. 修改点总结

### 6.1 原始部分

仍保留但不再是主路径的原始部分：

- `utils/unitraj.py`
- `utils/EMA.py`

继续复用的原始公共部分：

- `utils/logger.py`

### 6.2 新增/重构部分

真正构成当前主路径的新增或重构部分：

- `utils/flow_matching.py`
- `utils/dataset.py`
- `main.py`
- `evaluate_gpt.py`
- `utils/config.py`

### 6.3 与老师参考版本的关系

当前版本保留了老师版本的关键思想：

- Flow Matching
- DiT
- AdaLN
- SparseMoE
- spectral loss

但改成了当前仓库自己的**单阶段实现**：

- 不再依赖 coarse trajectory
- 不再导出 coarse/gt pair 作为主训练数据
- 不再走 refinement-only 二阶段流程

## 7. 重点结论

### 7.1 当前版本是单阶段还是二阶段

当前版本是**单阶段**。

原因：

- 训练时只有一个主模型 `SingleStageTrajectoryFlow`
- `x_0` 直接来自 `observed_trajectory + noise`
- `x_1` 直接是完整目标轨迹
- 没有 coarse predictor + flow refiner 的两阶段串联

### 7.2 训练和推理是否一致

一致的部分：

- 条件输入一致
- 已知点保留机制一致
- 起点初始化逻辑一致

不一致的部分：

- 训练监督速度场
- 推理通过 Euler 积分生成最终轨迹
- early stopping 看 flow loss，最终报告看 geodesic 指标

## 8. 注意事项

- `requirements.txt` 需要人工修正。
- `evaluate_gpt.py` 使用 `strict=False` 加载权重，评估配置必须与训练配置手动保持一致。
- 评估采样默认带随机噪声，多次运行结果可能轻微波动。
- `training.seed` 当前没有被用作全局随机种子，只影响验证集数据确定性。
- `calculate_norm_params/calculate_norm_params.py` 里有硬编码绝对路径。
- `utils/unitraj.py` 仍在仓库内，容易让人误以为它还是主模型入口，但实际上当前主路径已经切换到 `utils/flow_matching.py`。

## 9. 一句话总结

当前仓库的实际主路径已经是一个**单阶段 Flow Matching + DiT/AdaLN + 可选 SparseMoE + spectral loss** 的轨迹恢复系统；原始 `UniTraj` 骨干仍被保留为遗留代码资产，但不再是当前训练与评估入口。
