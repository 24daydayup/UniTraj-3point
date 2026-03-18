# UniTraj 项目快速导读

> 说明：当前工作树并不是论文仓库的完整原始结构。你指定的 `train.py`、`evaluation.py`、`data_analysis.py`、`unitraj/configs/`、`unitraj/datasets/`、`unitraj/models/` 在这个版本里并不存在；当前实际对应入口和核心实现分别是：
>
> - 训练入口：`main.py`
> - 评估入口：`evaluate_gpt.py`
> - 数据统计/预处理入口：`calculate_norm_params/calculate_norm_params.py`
> - 配置：`utils/config.py`
> - 数据集：`utils/dataset.py`
> - 模型：`utils/unitraj.py`

## 1. 项目整体流程

### 1.1 训练主流程

`data/*.pkl` 中的轨迹数据先进入 [`utils/dataset.py`](utils/dataset.py) 里的 `TrajectoryDataset`。数据集会先做重采样，再生成 mask，再把轨迹改成“相对起点坐标”，然后标准化、padding 到固定长度 200，最后返回：

- `trajectory`: `[B, 2, L]`
- `attention_mask`: 哪些位置是真实点
- `intervals`: 时间间隔
- `indices`: 需要 mask 的位置
- `original`: 原始起点坐标

训练入口 [`main.py`](main.py) 会构建 [`utils/unitraj.py`](utils/unitraj.py) 里的 `UniTraj` 模型，然后把 `trajectory + intervals + indices` 喂给模型。模型内部流程是：

1. `Encoder` 用 `Conv1d` 把轨迹切成 token。
2. `interval_embedding` 把时间间隔编码后加到 token 上。
3. `PatchShuffle` 把被 mask 的 token 放到后面，只让可见 token 进入编码器。
4. `Transformer` 编码可见 token。
5. `Decoder` 用 `mask_token` 把被遮掉的位置补回来并重建整条轨迹。
6. 训练只对“真实点且被 mask 的位置”计算 MSE。
7. 每个 epoch 做一次验证，保存最优权重并做 early stopping。

一句话概括：**这是一个把 MAE 思想迁移到轨迹序列上的“掩码重建式预训练”项目。**

### 1.2 评估主流程

评估入口是 [`evaluate_gpt.py`](evaluate_gpt.py)。它支持三类下游任务：

- `completion`: 随机遮住一部分真实点，评估补全能力
- `prediction`: 遮住最后 `N` 个真实点，评估预测能力
- `classification`: 复用预训练 encoder，接分类头

补全/预测的链路是：

1. 用 `TrajectoryDataset(mask_ratio=0.0)` 只做预处理，不做训练时那种随机 mask。
2. 用 `make_completion_masks()` 或 `make_prediction_masks()` 重新构造“评估用 mask”。
3. 用 `load_pretrained_unitraj()` 加载预训练权重。
4. 前向得到重建轨迹。
5. 用 `denormalise()` 把相对坐标恢复成真实经纬度。
6. 用 `geodesic_mae_rmse()` 按地理距离计算 MAE / RMSE，最后写入 `results/*.json`。

### 1.3 数据处理主流程

当前工作树里没有单独的 `data_analysis.py`，实际“最关键的数据处理入口”是 [`calculate_norm_params/calculate_norm_params.py`](calculate_norm_params/calculate_norm_params.py)：

1. 读取 `TrajectoryDataset(transform=None)`。
2. 遍历所有真实点。
3. 统计经纬度均值和标准差。
4. 把结果回填到 [`utils/dataset.py`](utils/dataset.py) 的 `Normalize` 里。

它的作用不是生成训练样本，而是**为训练/评估提供稳定的标准化参数**。

### 1.4 三个入口文件

| 任务 | 实际入口文件 | 作用 |
| --- | --- | --- |
| 训练 | `main.py` | 预训练 UniTraj，保存模型、日志和实验快照 |
| 评估 | `evaluate_gpt.py` | 做补全/预测评估，也提供分类微调骨架 |
| 数据处理 | `calculate_norm_params/calculate_norm_params.py` | 统计数据均值/方差，服务 `Normalize` |

## 2. 关键目录说明

| 目录/文件 | 说明 | 在主流程中的位置 |
| --- | --- | --- |
| `main.py` | 训练主入口 | 负责组网、DataLoader、训练循环、验证、保存权重 |
| `evaluate_gpt.py` | 评估与下游任务入口 | 负责补全/预测评估，以及分类任务骨架 |
| `calculate_norm_params/` | 数据统计脚本 | 用于计算标准化参数 |
| `utils/config.py` | 轻量配置文件 | 提供训练批大小、epoch、轨迹长度等基础配置 |
| `utils/dataset.py` | 数据集与预处理核心 | 负责读取 pkl、重采样、mask、标准化、padding |
| `utils/unitraj.py` | 模型核心实现 | 定义 RoPE、Attention、Encoder、Decoder、UniTraj |
| `utils/logger.py` | 日志工具 | 负责实验日志和参数打印 |
| `data/` | 样例数据 | 默认使用 `worldtrace_sample.pkl` |
| `results/` | 评估结果目录 | 保存重建任务的 JSON 指标 |
| `UniTraj/` | 实验输出目录 | `main.py` 会按时间戳保存模型、日志、代码快照 |

## 3. 最重要的函数/类

| 文件路径 | 类/函数 | 作用 | 为什么关键 |
| --- | --- | --- | --- |
| `main.py` | `main(config, logger)` | 训练主循环 | 把数据、模型、loss、验证、early stopping 串成完整训练链路 |
| `main.py` | `setup_experiment_directories(config, Exp_name)` | 创建实验目录并备份代码 | 决定权重、日志、结果输出结构 |
| `utils/dataset.py` | `Normalize` | 对相对坐标做标准化 | 训练和评估共用，影响数值稳定性 |
| `utils/dataset.py` | `logarithmic_sampling_ratio()` | 根据轨迹长度决定采样率 | 控制长短轨迹的重采样强度 |
| `utils/dataset.py` | `TrajectoryDataset` | 训练/评估统一数据集封装 | 所有样本都经过这里进入模型 |
| `utils/dataset.py` | `TrajectoryDataset.__getitem__()` | 单样本主处理逻辑 | 串起重采样、mask、相对化、标准化、padding |
| `utils/dataset.py` | `TrajectoryDataset.resample_trajectory()` | 轨迹重采样 | 直接决定输入轨迹的长度分布和时间间隔 |
| `utils/unitraj.py` | `PatchShuffle` | 把 mask 位置移到后部并返回索引映射 | 是“MAE 风格只编码可见 token”的关键 |
| `utils/unitraj.py` | `Encoder` | 编码可见轨迹 token | 负责提取时空表征，也是下游分类复用的主体 |
| `utils/unitraj.py` | `Decoder` | 用 `mask_token` 恢复完整序列并输出重建轨迹 | 决定预训练目标是否能实现 |
| `utils/unitraj.py` | `UniTraj` | 组装 encoder、decoder、interval embedding | 整个项目的总模型入口 |
| `evaluate_gpt.py` | `load_pretrained_unitraj()` | 加载预训练权重并处理 `module.` 前缀 | 是评估/微调复用模型的入口 |
| `evaluate_gpt.py` | `make_completion_masks()` | 为补全任务构造 mask | 让评估 mask 与训练 mask 解耦 |
| `evaluate_gpt.py` | `make_prediction_masks()` | 为预测任务构造“最后 N 点”mask | 直接定义预测任务的评估协议 |
| `evaluate_gpt.py` | `evaluate_reconstruction_tasks(args)` | 补全/预测评估主流程 | 串起数据、mask、推理、反归一化、地理距离指标 |

补充两个虽然没进前 15，但阅读时很值得顺手看：

- `evaluate_gpt.py::TrajectoryClassifier`：说明如何复用 encoder 做分类任务
- `calculate_norm_params/calculate_norm_params.py::calculate_stats`：说明标准化参数是怎么来的

## 4. 最重要的参数

> 这个版本没有 `unitraj/configs/` 目录，参数分散在三处：`utils/config.py`、`main.py` 里的模型硬编码、`evaluate_gpt.py` 的 CLI 参数。

| 参数 | 位置 | 默认值 | 控制什么 |
| --- | --- | --- | --- |
| `args.data.dataset` | `utils/config.py` | `worldtrace` | 实验命名和日志里的数据集标识 |
| `args.data.traj_length` | `utils/config.py` | `200` | 轨迹统一长度；也决定 encoder/decoder token 数 |
| `args.data.emb_dim` | `utils/config.py` | `128` | 期望的 embedding 维度标识；真实建模时对应 `embedding_dim` |
| `args.training.batch_size` | `utils/config.py` | `1024` | 训练与验证 batch 大小 |
| `args.training.n_epochs` | `utils/config.py` | `1000` | 最大训练 epoch 数 |
| `trajectory_length` | `main.py` / `evaluate_gpt.py` | `200` | 模型期望输入长度；和数据集 `max_len` 必须一致 |
| `patch_size` | `main.py` / `evaluate_gpt.py` | `1` | 每个 token 覆盖多少轨迹点；当前实现等于“每点一个 token” |
| `embedding_dim` | `main.py` / `evaluate_gpt.py` | `128` | token、时间间隔 embedding、Transformer 隐层宽度 |
| `encoder_layers` | `main.py` / `evaluate_gpt.py` | `8` | 编码器深度 |
| `encoder_heads` | `main.py` / `evaluate_gpt.py` | `4` | 编码器多头注意力头数 |
| `decoder_layers` | `main.py` / `evaluate_gpt.py` | 训练时 `8`，评估加载时 `4` | 解码器深度；负责重建能力与计算量 |
| `decoder_heads` | `main.py` / `evaluate_gpt.py` | `4` | 解码器多头注意力头数 |
| `mask_ratio` | `main.py`、`utils/dataset.py`、`evaluate_gpt.py` | 训练默认 `0.5`，评估数据集里设为 `0.0` | 控制要遮掉多少点；训练和评估语义不同但都很关键 |
| `max_len` | `utils/dataset.py` | `200` | 数据集 padding / truncate 的目标长度 |
| `MIN_POINTS` / `MAX_POINTS` / `MIN_SAMPLING_RATIO` | `utils/dataset.py` | `36 / 600 / 0.35` | 控制重采样曲线，影响长轨迹保留多少点 |
| `num_workers` | `main.py` / `evaluate_gpt.py` / `utils/config.py` | 训练 `32/16`，评估 `8` | DataLoader 并行度 |
| `lr` | `main.py` / `evaluate_gpt.py` | 训练 `1e-3`，分类 `1e-3` | 优化步长 |
| `patience` | `main.py` | `20` | early stopping 容忍多少轮验证不提升 |
| `predict_len` | `evaluate_gpt.py` | `5` | 预测任务中遮掉最后多少个真实点 |
| `freeze_encoder` | `evaluate_gpt.py` | `False` | 分类任务里是否冻结预训练 backbone |

### 最值得优先盯住的 8 个参数

如果你只是想先抓住项目“调什么会显著影响行为”，优先看这 8 个：

1. `max_len / trajectory_length`
2. `mask_ratio`
3. `patch_size`
4. `embedding_dim`
5. `encoder_layers`
6. `decoder_layers`
7. `batch_size`
8. `predict_len`

## 5. 建议阅读顺序

推荐按“先主流程，后细节”的顺序读：

1. 先看 `main.py`：抓训练链路、输入输出张量、loss 怎么算。
2. 再看 `utils/dataset.py`：搞清楚数据到底被怎么改造后才送进模型。
3. 再看 `utils/unitraj.py`：重点看 `PatchShuffle -> Encoder -> Decoder -> UniTraj`。
4. 然后看 `evaluate_gpt.py`：理解这个预训练模型怎样落到补全/预测/分类。
5. 最后看 `calculate_norm_params/calculate_norm_params.py`：理解 `Normalize` 的均值方差来源。

如果只留 30 分钟，我建议你只看这 7 个代码位置：

1. `main.py::main`
2. `utils/dataset.py::TrajectoryDataset.__getitem__`
3. `utils/dataset.py::TrajectoryDataset.resample_trajectory`
4. `utils/unitraj.py::PatchShuffle.forward`
5. `utils/unitraj.py::Encoder.forward`
6. `utils/unitraj.py::Decoder.forward`
7. `evaluate_gpt.py::evaluate_reconstruction_tasks`

---

## 一句话总结

这个项目的主线非常清楚：**`TrajectoryDataset` 负责把原始 GPS 轨迹变成固定长度、带 mask 的训练样本；`UniTraj` 用 MAE 风格的 encoder-decoder 重建被遮住的轨迹点；`evaluate_gpt.py` 再把同一个预训练模型迁移到补全、预测和分类任务。**
