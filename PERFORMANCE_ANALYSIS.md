# UniTraj-3point 性能分析与优化路线（2026-04-07）

## 1) 当前项目训练/评估主流程（基于现有代码）

### 1.1 训练入口与损失
- 训练主入口是 `main.py`，核心循环在 `train()`：构建 `TrajectoryDataset`、`DataLoader`、`SingleStageTrajectoryFlow`，再做 AdamW + ReduceLROnPlateau + early stopping。 
- 训练 loss 在 `compute_training_loss()`：
  - Flow Matching 的 velocity MSE（主损失）
  - 可选 spectral loss（速度场频域一致性）
  - 总损失 = `mse + spectral_weight * spectral_loss`

### 1.2 模型结构
- 模型定义在 `utils/flow_matching.py::SingleStageTrajectoryFlow`。
- 主干是 DiT-style Transformer block（AdaLN 调制），可选 MLP->SparseMoE。
- 条件输入包括：`observed_trajectory`、`intervals`、`observed_mask`、`attention_mask`。
- 可选 adaptive patch 分支：`AdaptiveTrajectoryPatcher` + `AdaptivePatchEncoder` + `PointPatchFusion`。

### 1.3 数据与 mask
- 数据由 `utils/dataset.py::TrajectoryDataset` 读取 pkl 后处理：重采样、相对化、Normalize、pad/truncate。
- 训练 mask 支持 `completion / prediction / mixed`：
  - mixed 下按 `completion_prob` 在补全和预测之间采样
  - completion 内部再混合 random / rdp / block 三类 mask

### 1.4 推理与评估
- 推理使用 `sample_trajectory_flow()` 做 Euler 积分，多步更新 unknown 区域，known 点每步强制写回。
- 评估在 `evaluate_gpt.py`：支持 completion/prediction，输出 geodesic MAE/RMSE。
- 评估会优先加载训练时保存的 `config_snapshot.json`，但 checkpoint 加载是 `strict=False`（需警惕静默不一致）。

---

## 2) 与“原始 UniTraj 主线”的差异（可确认与不可确认）

> 说明：仓库中没有完整“原始 UniTraj”实现文件（如旧的 MAE 主线代码），因此只能依据当前 README 与项目说明文档做差异梳理，不能逐行 diff 原始模型。

### 可确认差异
1. 当前主线已经切换为**单阶段 Flow Matching**，不再走旧 MAE 主线。  
2. 当前主线包含可选 **adaptive patch 条件编码**（patcher + encoder + fusion）。  
3. 当前评估任务聚焦 completion/prediction，指标是 geodesic MAE/RMSE。  
4. 当前训练目标是 velocity matching（+ 可选频域约束），而不是纯重建式 MAE 目标。

### 需要谨慎的点
- “与原始 UniTraj”的精确结构差异、mask 策略差异、推理细节差异：由于原始代码不在当前仓库，无法做严格源码级对照结论。

---

## 3) 主要性能瓶颈判断（按优先级）

### P0：训练/验证数据泄漏风险
- 默认配置里 `train_path == val_path`，会让验证指标偏乐观，early stopping 与调参方向都可能失真。

### P0：训练目标与最终指标不完全一致
- 早停与调度器依据 flow loss（velocity MSE + spectral），最终关心却是 geodesic MAE/RMSE。
- 当二者相关性不稳定时，会出现“loss 好看但地理误差不降”。

### P1：时间间隔 `intervals` 缺少显式尺度控制
- `intervals` 直接进线性层，若间隔分布重尾，条件输入尺度可能不稳定。

### P1：mask 分布与目标任务分布可能不匹配
- mixed + completion 内三种 mask 混合策略较重，但线上/最终关注通常更偏 prediction（尾部外推）或特定缺失模式。

### P1：推理步数固定且偏小
- 默认 `sample_steps=16`，在复杂轨迹上可能积分误差偏大，影响最终 MAE/RMSE。

### P2：实现层面潜在隐患
- `evaluate_gpt.py` 用 `strict=False` 加载模型，结构不匹配时可能“部分加载成功但性能下降且不易察觉”。

---

## 4) 最值得先试的 5 条建议（结合当前实现）

## 1. 先修复数据切分（最高优先级）
- 改什么：将 `utils/config.py` 的 `val_path` 改为独立验证集文件；确保 train/val 不重叠。
- 为什么有效：验证信号真实后，early stopping、LR 衰减和所有调参才有意义。
- 影响指标：主要影响验证可信度，间接提升最终 MAE/RMSE。
- 难度/风险：低难度、低风险。

## 2. 将“模型选择”对齐到 geodesic 指标
- 改什么：在训练每个 epoch 后，调用小规模采样评估（如固定若干 val batch），记录 completion/prediction geodesic MAE；保存 best-geodesic checkpoint。
- 为什么有效：避免“优化了代理损失但没优化目标指标”。
- 影响指标：直接影响 MAE/RMSE。
- 难度/风险：中低难度、低风险（增一点验证开销）。

## 3. 做推理步数扫参（16/24/32/48）
- 改什么：评估脚本里只改 `--sample_steps`，不改训练。
- 为什么有效：Flow ODE 采样质量对最终误差敏感，通常是高性价比提升项。
- 影响指标：completion/prediction MAE、RMSE。
- 难度/风险：极低难度、低风险，仅增加推理耗时。

## 4. 调整 mixed 任务采样比例（提高 prediction 占比）
- 改什么：把 `completion_prob` 从 0.7 逐步降到 0.5 / 0.3 做对照。
- 为什么有效：当前默认训练更偏 completion，若目标是 prediction，任务分布偏差会拉低尾部预测效果。
- 影响指标：主要提升 prediction MAE/RMSE，可能轻微影响 completion。
- 难度/风险：低难度、中低风险。

## 5. 降低/关闭 spectral loss 做对照
- 改什么：`spectral_loss_weight` 从 0.1 -> 0.05 -> 0.0；必要时 `use_spectral_loss=False`。
- 为什么有效：频域约束在速度场上未必总与 geodesic 误差同向，过强可能抑制局部几何细节拟合。
- 影响指标：可能改善 MAE（尤其短期 prediction）；RMSE 视数据而定。
- 难度/风险：低难度、中风险（可能对长轨迹平滑性有副作用）。

---

## 5) 建议分级：A/B/C 三类优化

## A. 低风险（仅调参，不改主体）
1. `sample_steps` 扫参：16/24/32/48。  
2. `completion_prob` 扫参：0.7/0.5/0.3。  
3. `spectral_loss_weight` 扫参：0.1/0.05/0.0。  
4. `mask_ratio` 扫参：0.3/0.5/0.7（completion 为主时重点看）。  
5. `predict_len` 与训练一致性对齐（例如训练 8、评估 5 时做双向对照）。

## B. 中风险（小改代码，不重构）
1. 训练中增加 geodesic 验证分支，按 geodesic 保存 best。  
2. checkpoint 加载改为可切换 `strict=True`（至少默认报警 missing/unexpected key）。  
3. 对 `intervals` 做 clip/log1p/标准化（训练与评估一致）。

## C. 高风险（收益可能大，但改动更大）
1. 推理器从固定 Euler 升级到 Heun / predictor-corrector。  
2. 改为多任务 loss（velocity + trajectory position consistency + spectral 动态权重）。  
3. 让 patcher 从 rule/hybrid 进化为可学习边界（并引入边界正则），替换当前阈值规则。

---

## 6) 推荐实验路线（先小改、快闭环）

### Step 0（前置校准）
- 修改：`utils/config.py` 的 `val_path` 独立。
- 判断标准：验证指标波动更真实，不再异常乐观。

### Step 1（零代码、最快验证）
- 修改：固定 checkpoint，仅改 `evaluate_gpt.py --sample_steps`。
- 预期：存在 1 个较优步数（常见在 24~32）。

### Step 2（低风险调参）
- 修改：`completion_prob` 和 `spectral_loss_weight` 两维网格（例如 3x3）。
- 文件：`utils/config.py`。
- 预期：prediction MAE 有明显改善区间；若 spectral 降低后 MAE 降、RMSE 升，需按业务取舍。

### Step 3（小幅代码优化）
- 修改：`main.py` 增加轻量 geodesic 验证并保存 `best_geodesic.pt`。
- 预期：最终 checkpoint 与目标指标更一致，复现实验更稳定。

### Step 4（必要时）
- 修改：`utils/dataset.py` / `evaluate_gpt.py` 引入 interval 标准化策略。
- 预期：训练更稳定、不同时采样率样本鲁棒性提升。

---

## 7) 结论（给你一个可直接执行的优先清单）

1. **当前最可能的瓶颈**：数据切分问题 + 优化目标与评估指标错位。  
2. **最优先调参项**：`sample_steps`、`completion_prob`、`spectral_loss_weight`。  
3. **最优先代码项**：训练期加入 geodesic 选模；checkpoint 严格加载/告警。  
4. **推荐顺序**：先修数据切分 -> 推理步数扫参 -> 低风险 loss/mask 配比 -> 再做小代码优化。  
5. **最可能带来显著提升**：
   - 若当前 val 泄漏：修复切分会立刻改善“实验可信度与泛化”
   - 若当前 geodesic 不理想：`sample_steps` + geodesic 选模通常是最稳的增益组合
