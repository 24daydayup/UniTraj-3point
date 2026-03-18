# UniTraj Single-Stage Flow Matching

这个仓库现在只保留一条主线：

> 基于单阶段 Flow Matching 的轨迹补全/预测系统，并在条件分支中支持可选的 adaptive patch 编码。

旧的 `UniTraj` MAE 主线、两阶段 diffusion/refinement 实现，以及平行的 `unitraj-flow/` 子项目都已经移除。当前 README 以根目录代码为准。

## 1. 项目概览

### 当前任务

- `completion`：随机遮住一部分真实轨迹点，恢复缺失轨迹
- `prediction`：遮住轨迹最后 `N` 个真实点，预测未来轨迹

### 当前模型

- 主模型：`SingleStageTrajectoryFlow`
- 训练范式：Flow Matching / Rectified Flow 风格线性路径
- 主干结构：DiT-style Transformer + AdaLN
- 可选组件：
  - SparseMoE
  - spectral consistency loss
  - adaptive patch 条件编码

### 当前主入口

- 训练：`main.py`
- 评估：`evaluate_gpt.py`
- 数据统计：`calculate_norm_params/calculate_norm_params.py`

## 2. 当前目录结构

```text
.
├── README.md
├── main.py
├── evaluate_gpt.py
├── debug_smoke_test.py
├── requirements.txt
├── calculate_norm_params/
│   └── calculate_norm_params.py
└── utils/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── flow_matching.py
    ├── adaptive_patcher.py
    ├── adaptive_patch_encoder.py
    ├── project.py
    └── logger.py
```

### 说明

- `utils/project.py` 是共享运行工具，负责配置转换、建模、随机种子、checkpoint 命名和 config snapshot。
- 根目录中其他 `*.md` 文件如果存在，更多是开发笔记或提示词，不是项目真相源；请优先以 README 和代码为准。

## 3. 数据格式

训练和评估都默认读取一个 pickle DataFrame。

每条样本至少需要包含：

- `time`
- `trajectory`

其中：

- `time` 是时间序列，可被 `pandas.to_datetime` 解析
- `trajectory` 是点序列
- 每个点默认按 `[lat, lon]` 存储

数据在进入模型前会做这些处理：

1. 轨迹重采样
2. 转为相对首点坐标
3. 标准化
4. padding / truncate 到固定长度
5. 构造 `loss_mask / observed_mask / observed_trajectory / attention_mask / intervals`

## 4. 安装环境

建议 Python 3.10+。

安装依赖：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 只保留主线实际需要的依赖：

- `torch`
- `numpy`
- `pandas`
- `tqdm`
- `colored`
- `rdp`
- `geopy`

如果没有安装 `colored`，日志模块也能退化为普通无颜色输出。

## 5. 配置文件

项目使用静态 Python 配置：

- `utils/config.py`

分为三部分：

- `data`
- `model`
- `training`

### 重点配置项

#### 数据侧

- `train_path`
- `val_path`
- `traj_length`
- `mask_ratio`
- `task_mode`
- `completion_prob`
- `predict_len`

#### 模型侧

- `hidden_size`
- `depth`
- `num_heads`
- `mlp_ratio`
- `use_moe`
- `use_spectral_loss`
- `sample_steps`

#### Adaptive Patch

`utils/config.py -> args["model"]["adaptive_patch"]`

关键开关：

- `enabled`
- `score_mode`
- `min_patch_len`
- `max_patch_len`
- `patch_encoder_dim`
- `use_point_patch_fusion`

当 `enabled=False` 时，模型会退化为普通 point-level 条件编码，不使用 variable patch。

## 6. 训练

先修改 `utils/config.py`，再运行：

```bash
python3 main.py
```

训练时会自动：

1. 读取配置
2. 设置随机种子
3. 构建训练/验证集
4. 训练 `SingleStageTrajectoryFlow`
5. 在实验目录里保存代码快照和 `config_snapshot.json`

输出目录格式：

```text
UniTraj/<dataset>_bs=<batch_size>/<timestamp>/
```

常见输出文件：

- `models/best_flow_model.pt`
- `models/final_flow_model.pt`
- `models/best_flow_adaptive_patch.pt`
- `models/final_flow_adaptive_patch.pt`
- `config_snapshot.json`
- `out.log`

说明：

- 如果 adaptive patch 开启，checkpoint 会使用 `*_adaptive_patch.pt`
- `config_snapshot.json` 会在评估时自动读取，减少训练/评估超参不一致的问题

## 7. 评估

### 轨迹补全

```bash
python3 evaluate_gpt.py \
  --task completion \
  --model_path ./UniTraj/worldtrace_bs=256/<timestamp>/models/best_flow_adaptive_patch.pt \
  --test_data ./data/worldtrace_sample.pkl \
  --mask_ratio 0.5
```

### 轨迹预测

```bash
python3 evaluate_gpt.py \
  --task prediction \
  --model_path ./UniTraj/worldtrace_bs=256/<timestamp>/models/best_flow_adaptive_patch.pt \
  --test_data ./data/worldtrace_sample.pkl \
  --predict_len 5
```

### 同时评估补全和预测

```bash
python3 evaluate_gpt.py \
  --task completion,prediction \
  --model_path ./UniTraj/worldtrace_bs=256/<timestamp>/models/best_flow_adaptive_patch.pt \
  --test_data ./data/worldtrace_sample.pkl \
  --mask_ratio 0.5 \
  --predict_len 5
```

### 评估脚本特性

- 会优先自动寻找 checkpoint 同目录实验下的 `config_snapshot.json`
- 找不到 snapshot 时，回退到默认配置 + CLI 覆盖
- 输出 geodesic `MAE / RMSE`
- 如果 adaptive patch 开启，还会额外打印 patch 统计

结果默认写到：

```text
results/flow_reconstruction_<timestamp>.json
```

## 8. 计算标准化参数

如果你换了数据集，建议重新计算 `Normalize` 的均值和标准差：

```bash
python3 calculate_norm_params/calculate_norm_params.py \
  --data_path ./data/worldtrace_sample.pkl \
  --max_len 200 \
  --batch_size 4096 \
  --num_workers 4
```

脚本会输出：

- `mean`
- `std`

然后把结果回填到 `utils/dataset.py -> Normalize`。

## 9. 模型与数据流

### 训练时

1. 从 `observed_trajectory` 构造 `x_0`
2. 用完整轨迹作为 `x_1`
3. 采样时间 `t`
4. 构造线性路径 `x_t = (1 - t) * x_0 + t * x_1`
5. 预测 velocity `x_1 - x_0`
6. 只在未知区域计算 loss

### 采样时

1. 用 `observed_trajectory + noise` 初始化
2. 做 Euler 积分
3. 每一步都把 observed points 写回
4. 得到最终补全/预测轨迹

## 10. Adaptive Patch 机制

当前 adaptive patch 只增强条件编码，不改变：

- `x_t` 的 point-level 输入形式
- velocity 的 point-level 输出形式
- `sample_trajectory_flow()` 的采样方式

流程是：

1. `AdaptiveTrajectoryPatcher` 根据轨迹复杂度切分 variable-length patches
2. `AdaptivePatchEncoder` 将每个 patch 编码为 fixed-size patch token
3. `PointPatchFusion` 把 patch context 融回 point-level condition

这意味着：

- 主生成器仍然是 point-level flow matching
- patch 机制只影响条件表达能力

## 11. 快速自检

如果环境里已经安装了 `torch`，可以运行：

```bash
python3 debug_smoke_test.py
```

这个脚本会检查：

- patcher 输出是否合法
- patch encoder 输出形状是否正确
- flow model 前向/反向是否能跑通
- sampler 是否能正常执行

## 12. 常见问题

### 1. `config_snapshot.json` 找不到怎么办？

评估脚本会自动回退到默认配置，但更推荐直接用训练输出目录下的 checkpoint，这样能自动对齐训练配置。

### 2. 为什么评估结果会有轻微波动？

采样阶段默认包含随机噪声初始化，所以多次运行可能略有差异。

### 3. adaptive patch 开启后显存更高正常吗？

正常。因为会新增 patch 检测、patch 编码和 point-patch cross-attention。

### 4. 为什么会出现长度相关报错？

模型会检查输入序列长度是否超过 `max_len`。请确认：

- 数据集的 `max_len`
- 配置里的 `traj_length`
- checkpoint 对应的训练配置

三者一致。

## 13. 当前保留与删除情况

### 已保留

- 单阶段 flow matching 主线
- adaptive patch 条件编码
- 训练/评估/标准化脚本

### 已删除

- 旧 `unitraj-flow/` 平行子项目
- 旧 `utils/unitraj.py`
- 旧 `utils/EMA.py`
- 两阶段 diffusion/refinement 相关代码路径

这次整理后的目标很明确：

> 仓库只服务当前单阶段轨迹恢复主线，不再维护历史双系统。
