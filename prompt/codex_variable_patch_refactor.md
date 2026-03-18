# 给本地 Codex 的完整改造说明：在当前项目中引入基于 EntroPE 思路的可变 Patch 机制

## 0. 任务目标

你正在修改的仓库不是老的 `utils/unitraj.py` 主线，而是当前正在使用的 **single-stage flow matching 主线**：

- `main.py`
- `evaluate_gpt.py`
- `utils/config.py`
- `utils/dataset.py`
- `utils/flow_matching.py`

本次改造的目标不是复活老 UniTraj，也不是把整个模型重写成纯 patch-level 生成器，而是：

> **在保持当前 point-level flow matching 输出与采样器不变的前提下，引入“内容复杂度驱动的可变 patch 条件编码器”。**

也就是说：

- `x_t` 仍然是逐点输入
- velocity 仍然是逐点输出
- `sample_trajectory_flow()` 的 Euler 积分逻辑保持不变
- 改造重点是 `observed_trajectory` 的条件侧编码
- 用 **Adaptive Trajectory Patcher + Adaptive Patch Encoder** 替换当前简单的 `cond_embedder(observed_trajectory)`

本方案参考 EntroPE 的公开思路：

- 用动态边界检测替代固定 patch
- 用变长 patch 编码器把不等长 patch 映射成定长 patch tokens
- 再把 patch-level context 融合回 point-level 条件表示

注意：不要复制第三方仓库源码；请用 clean-room 方式按本文档实现。

---

## 1. 总体设计原则

### 1.1 保持当前训练目标不变

仍然训练：

- `x_0 = observed + noise on unknown region`
- `x_t = (1 - t) * x_0 + t * x_1`
- 预测 `velocity = x_1 - x_0`
- loss 仍然只在 `unknown_mask` 上计算

### 1.2 可变 patch 只改条件分支，不改主输出头

当前 `SingleStageTrajectoryFlow.forward()` 里这一段：

```python
interval_tokens = self.interval_embedder(intervals.unsqueeze(-1))
mask_features = torch.stack([observed_mask, attention_mask], dim=-1)
mask_tokens = self.mask_embedder(mask_features)
cond_tokens = self.cond_embedder(observed_trajectory) + interval_tokens + mask_tokens

x = self.x_embedder(x_t) + cond_tokens + self.pos_embed[:, :seq_len, :]
```

需要改成：

1. 先根据 `observed_trajectory + intervals + observed_mask + attention_mask` 生成可变 patch 边界
2. 把每个 patch 编码为一个 fixed-size patch token
3. 再把 patch token 融合回 point-level condition tokens
4. 最终 `x = x_embedder(x_t) + fused_cond_tokens + pos_embed`

### 1.3 第一版不要做这些事情

第一版**不要**：

- 不要把 `x_t` 改成 patch-level
- 不要把 velocity 改成 patch-level 再上采样
- 不要在 flow 采样的每个 step 都重新计算 patch 边界
- 不要依赖外部大型库
- 不要直接修改为必须训练 entropy model 才能跑

第一版要支持：

- 纯规则可变 patch：可直接训练
- 规则 + 轻量 learned scorer：默认可关
- 开关关闭时完全退化为现有模型行为

---

## 2. 目标目录改动

### 2.1 新增文件

新增以下文件：

- `utils/adaptive_patcher.py`
- `utils/adaptive_patch_encoder.py`

### 2.2 修改文件

修改以下文件：

- `utils/config.py`
- `utils/flow_matching.py`
- `main.py`
- `evaluate_gpt.py`
- `utils/__init__.py`（如需要）

### 2.3 不要改动的文件

本次尽量不要动：

- `utils/unitraj.py`
- `utils/EMA.py`
- `model.pt`

如果仓库已经准备清理老残留，可以另开 commit 删除；本 commit 只聚焦 variable patch 改造。

---

## 3. 新增配置项：修改 `utils/config.py`

在 `args["model"]` 下加入一个 `adaptive_patch` 配置块，默认开启规则版，默认不启用 learned scorer：

```python
"adaptive_patch": {
    "enabled": True,
    "score_mode": "rule",            # "rule" | "learned" | "hybrid"
    "hidden_dim": 64,
    "threshold_global": 0.55,
    "threshold_relative": 0.10,
    "min_patch_len": 4,
    "max_patch_len": 24,
    "smooth_kernel": 5,
    "learned_weight": 0.5,
    "patch_encoder_dim": 256,
    "patch_encoder_heads": 4,
    "patch_encoder_layers": 2,
    "patch_dropout": 0.1,
    "use_patch_length_embedding": True,
    "use_point_patch_fusion": True,
    "fusion_heads": 4,
    "fusion_dropout": 0.1,
    "detach_patch_scores": False
}
```

要求：

- 配置关闭时必须退化为原模型行为
- `patch_encoder_dim` 默认与 `hidden_size` 一致，若不一致则在模块内做线性映射
- 训练脚本和评估脚本都从同一套配置读取这些参数

---

## 4. 新增文件一：`utils/adaptive_patcher.py`

实现一个 **AdaptiveTrajectoryPatcher**，负责在 point-level 序列上检测 patch 边界。

### 4.1 模块职责

输入：

- `trajectory`: `[B, L, 2]`
- `attention_mask`: `[B, L]`
- `intervals`: `[B, L]`
- `observed_mask`: `[B, L]`

输出：

- `scores`: `[B, L]` 每个位置的信息密度分数
- `patch_specs`: Python list，每个 batch 一个 `PatchSpec`
- `patch2point_mask`: `[B, Pmax, L]`
- `patch_lengths`: `[B, Pmax]`

### 4.2 数据结构

实现：

```python
@dataclass
class PatchSpec:
    starts: List[int]
    ends: List[int]
    scores: List[float]
```

### 4.3 规则分数设计

至少融合以下几种局部复杂度信号：

1. **速度变化率**
2. **加速度幅值**
3. **heading / 方向变化幅值**
4. **observed/unobserved 边界变化先验**（可选小权重）

具体要求：

- 先计算 point-level velocity 和 acceleration
- 再计算 speed、`abs(diff(speed))`、heading change
- 对每条轨迹内部做 robust normalization（例如除以样本内 max 或 95 分位）
- 线性加权成 `rule_scores`
- 用 average pooling 做一维平滑
- 所有无效位置乘 `attention_mask`

建议默认权重：

- heading_change: 0.35
- acceleration_norm: 0.30
- speed_change: 0.25
- observed_transition_bonus: 0.10

### 4.4 learned scorer

实现一个轻量 scorer：

```python
class LocalComplexityScorer(nn.Module):
```

使用：

- 两层或三层 `Conv1d + GELU`
- 输入特征至少包括：
  - position (2)
  - velocity (2)
  - acceleration (2)
  - speed (1)
  - heading change (1)
  - intervals (1)
  - observed flag (1)
- 输出 `[B, L]` 的 score，经过 sigmoid

### 4.5 score_mode 行为

支持：

- `rule`：只用规则分数
- `learned`：只用 learned scorer
- `hybrid`：线性融合两者

`hybrid` 时：

```python
scores = (1 - learned_weight) * rule_scores + learned_weight * learned_scores
```

### 4.6 边界检测算法

实现一个 **dual-threshold + max-length** 的 patch 边界策略：

对单条轨迹：

- 从位置 0 开始
- 当某位置同时满足：
  - `score[i] >= threshold_global`
  - `score[i] >= local_mean + threshold_relative`
  - 当前 patch 长度 >= `min_patch_len`
  时，在该位置断开
- 如果当前 patch 长度达到 `max_patch_len`，强制断开
- 轨迹结尾必须闭合成 patch
- 如果出现过短 patch，需要与前一个 patch 合并

### 4.7 输出 `patch2point_mask`

将每个 patch 表示为一个 mask：

- 形状 `[B, Pmax, L]`
- 如果第 `p` 个 patch 覆盖 `[s:e]`，则 `patch2point_mask[b, p, s:e+1] = 1`
- 不足 `Pmax` 的 patch 行用 0 填充

### 4.8 编码风格要求

要求：

- 所有函数有 type hints
- 不使用任何第三方依赖
- 兼容 CPU 和 CUDA
- 不在模块内写训练逻辑

---

## 5. 新增文件二：`utils/adaptive_patch_encoder.py`

实现一个 **AdaptivePatchEncoder**，负责把 variable-length patches 映射为 fixed-size patch tokens。

### 5.1 模块职责

输入：

- `trajectory`: `[B, L, 2]`
- `attention_mask`: `[B, L]`
- `patch2point_mask`: `[B, P, L]`
- `intervals`: `[B, L]`
- `observed_mask`: `[B, L]`

输出：

- `patch_tokens`: `[B, P, D]`
- `patch_padding_mask`: `[B, P]`，无效 patch 为 `True`
- `point_tokens`: `[B, L, D]`
- `patch_lengths`: `[B, P]`

### 5.2 point-level embedding

先把 point-level 特征映射到 hidden space：

输入特征至少包括：

- trajectory `(lon, lat)`
- intervals
- observed_mask（如存在）

通过一个小的 `MLP`：

```python
Linear -> GELU -> Linear
```

得到 `point_tokens: [B, L, D]`

### 5.3 patch seed pooling

根据 `patch2point_mask` 做 masked average pooling：

```python
patch_seed = (patch2point_mask @ point_tokens) / patch_length
```

注意：

- 要乘 `attention_mask`
- patch length 至少 clamp 到 1

### 5.4 patch length embedding

加入 patch length embedding：

- 用 `nn.Embedding(max_patch_len_embed + 1, D)`
- 将 `patch_lengths` clamp 到 embedding 上限
- `patch_seed += length_embedding`

### 5.5 patch 内 refinement

为保留 intra-patch 依赖，加入 patch 内 cross-attention refinement：

实现：

```python
class PatchCrossAttention(nn.Module)
```

要求：

- query = patch seed，形状 `[B*P, 1, D]`
- key/value = patch 内所有 point tokens，padding 到同一长度
- `nn.MultiheadAttention(batch_first=True)`
- 残差 + LayerNorm + FFN
- 默认堆叠 2 层

### 5.6 patch sequence 提取工具

实现辅助函数，把 `patch2point_mask` 展平成每个 patch 的 point token 序列：

- 输出 `[B*P, Tm, D]`
- 输出 `[B*P, Tm]` 的 `key_padding_mask`
- 输出 `(b, p)` 索引映射

### 5.7 输出规范

最后返回：

- `patch_tokens`: refinement 后的 patch 表示
- `patch_padding_mask`: 空 patch 行置 `True`
- `point_tokens`: 原 point-level token
- `patch_lengths`

---

## 6. 修改 `utils/flow_matching.py`

这是最核心的改造文件。

### 6.1 新增模块导入

在文件头部新增：

```python
from utils.adaptive_patcher import AdaptiveTrajectoryPatcher
from utils.adaptive_patch_encoder import AdaptivePatchEncoder
```

### 6.2 新增 PointPatchFusion 模块

在 `flow_matching.py` 内新增：

```python
class PointPatchFusion(nn.Module):
```

功能：

- 输入：
  - `point_q: [B, L, D]`
  - `patch_kv: [B, P, D]`
  - `patch_padding_mask: [B, P]`
- 用 cross-attention 将 patch context 融合到 point-level condition 中
- 输出仍是 `[B, L, D]`

实现要求：

- `nn.MultiheadAttention(batch_first=True)`
- LayerNorm + residual
- FFN 可选，但建议保留一层简短 FFN
- patch padding mask 传给 `key_padding_mask`

### 6.3 修改 `SingleStageTrajectoryFlow.__init__`

#### 6.3.1 新增参数

在构造函数新增：

```python
adaptive_patch_cfg: Optional[dict] = None
```

#### 6.3.2 保留现有模块

保留原有：

- `self.x_embedder`
- `self.cond_embedder`
- `self.interval_embedder`
- `self.mask_embedder`
- `self.pos_embed`
- `self.blocks`
- `self.final_layer`

#### 6.3.3 新增可变 patch 相关模块

如果 `adaptive_patch_cfg["enabled"]` 为真，则新增：

- `self.adaptive_patcher`
- `self.patch_encoder`
- `self.point_patch_fusion`
- 必要时新增 `self.patch_proj` / `self.cond_proj`

要求：

- `patch_encoder_dim` 若不等于 `hidden_size`，则把 patch token 投影到 `hidden_size`
- 允许 `score_mode="rule"` 时 learned scorer 仍然存在，但不参与计算

### 6.4 修改 `forward()`

当前 `forward()` 签名不变：

```python
def forward(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    observed_trajectory: torch.Tensor,
    intervals: torch.Tensor,
    attention_mask: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
```

#### 6.4.1 新的数据流

在 `forward()` 中，按如下逻辑构造条件 token：

1. 先构造基础 point-level 条件：

```python
interval_tokens = self.interval_embedder(intervals.unsqueeze(-1))
mask_features = torch.stack([observed_mask, attention_mask], dim=-1)
mask_tokens = self.mask_embedder(mask_features)
base_cond_tokens = self.cond_embedder(observed_trajectory) + interval_tokens + mask_tokens
```

2. 如果 adaptive patch 开启：

- 用原始 `observed_trajectory` 调用 `self.adaptive_patcher(...)`
- 用 patcher 输出调用 `self.patch_encoder(...)`
- 用 `self.point_patch_fusion(...)` 把 patch context 融回 point-level token
- 最终：

```python
cond_tokens = fused_cond_tokens + interval_tokens + mask_tokens
```

或者：

```python
cond_tokens = base_cond_tokens + fused_patch_context
```

要求保留清晰的一致性，不要重复叠加导致数值爆炸。

推荐实现：

- `point_tokens = self.cond_embedder(observed_trajectory)`
- `patch_context = fusion(point_tokens, patch_tokens)`
- `cond_tokens = patch_context + interval_tokens + mask_tokens`

3. 如果 adaptive patch 关闭：

- 完全回退到原来的 `cond_tokens`

4. 最终：

```python
x = self.x_embedder(x_t) + cond_tokens + self.pos_embed[:, :seq_len, :]
```

其余：

- `t_embedding`
- `pooled`
- `context_projector`
- `DiTBlock`
- `final_layer`

尽量保持不变。

### 6.5 pooled context 的来源

当前有：

```python
pooled = cond_tokens * attention_mask.unsqueeze(-1)
pooled = pooled.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
condition = t_embedding + self.context_projector(pooled)
```

这里继续沿用 `cond_tokens` 即可。

### 6.6 可选：debug 输出

在模型里增加一个可选属性或最近一次缓存（仅调试用，不参与训练逻辑），例如：

- `self._last_patch_scores`
- `self._last_patch_lengths`
- `self._last_num_patches`

要求：

- 默认不打印
- 不影响梯度
- 仅用于训练/评估时可视化分析

---

## 7. 修改 `main.py`

### 7.1 修改 `build_model(config)`

当前：

```python
return SingleStageTrajectoryFlow(
    input_dim=2,
    hidden_size=config.model.hidden_size,
    depth=config.model.depth,
    num_heads=config.model.num_heads,
    max_len=config.data.traj_length,
    mlp_ratio=config.model.mlp_ratio,
    use_moe=config.model.use_moe,
    num_experts=config.model.num_experts,
    top_k=config.model.top_k,
)
```

改为把 `adaptive_patch_cfg` 传进去：

```python
adaptive_patch_cfg=getattr(config.model, "adaptive_patch", None)
```

或者如果 `config` 被转成 `SimpleNamespace`，则用合适的方式读取。

### 7.2 增加训练日志

在训练日志中增加以下统计，建议每个 epoch 的 train/val 至少输出平均值：

- 平均 patch 数量
- 平均 patch 长度
- 最大 patch 长度
- 如果有 scorer：平均 score

如果实现复杂，可先在 `run_quick_sample_check` 或 `evaluate_epoch` 中聚合。

### 7.3 不改 loss

`compute_training_loss()` 不需要改动 loss 本身。

---

## 8. 修改 `evaluate_gpt.py`

目标：让评估脚本能够兼容新的 adaptive patch 模型。

要求：

- 构建模型时传入同样的 `adaptive_patch_cfg`
- 保持原有评估逻辑
- 可选增加 patch 统计输出：
  - 平均 patch 数
  - 平均有效 patch 长度

不要修改评估指标定义。

---

## 9. 关键实现细节要求

### 9.1 tensor 形状约定

当前项目里 `TrajectoryDataset` 返回的轨迹在 batch 中是 `[B, 2, L]`，而 `main.py` 在训练前会 `.transpose(1, 2)` 变成 `[B, L, 2]`。

所有新模块统一要求输入为：

- `trajectory: [B, L, 2]`
- `intervals: [B, L]`
- `attention_mask: [B, L]`
- `observed_mask: [B, L]`

### 9.2 不能破坏 padding 语义

任何 patch 计算必须 respect `attention_mask`：

- padding 部分不得参与 score 计算
- padding 部分不得被划入有效 patch
- pooling/cross-attention 时必须屏蔽 padding

### 9.3 observed point 不应被改变

本次 patch 只是条件编码增强，不改变：

- `build_flow_source()`
- `sample_trajectory_flow()` 中“每步写回 observed points”的逻辑

### 9.4 数值稳定性

所有除法都必须：

```python
clamp_min(1e-6)
```

所有 patch length 至少 clamp 到 1。

### 9.5 性能要求

默认配置下相较原模型：

- 显存增加应尽量可控
- 不要引入 O(L^2 * P) 级别的明显低效 Python 循环
- 允许 patch sequence 展平阶段使用有限 Python 循环，但要保持逻辑清晰

---

## 10. 建议的类与函数签名

请严格按下面签名实现，便于后续维护。

### 10.1 `utils/adaptive_patcher.py`

```python
@dataclass
class PatchSpec:
    starts: List[int]
    ends: List[int]
    scores: List[float]


def masked_diff(x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    ...


class LocalComplexityScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, kernel_size: int = 5):
        ...

    def forward(self, feats: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ...


class AdaptiveTrajectoryPatcher(nn.Module):
    def __init__(
        self,
        coord_dim: int = 2,
        hidden_dim: int = 64,
        score_mode: str = "hybrid",
        threshold_global: float = 0.55,
        threshold_relative: float = 0.10,
        min_patch_len: int = 4,
        max_patch_len: int = 24,
        smooth_kernel: int = 5,
        learned_weight: float = 0.5,
    ):
        ...

    def build_features(
        self,
        trajectory: torch.Tensor,
        intervals: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ...

    def compute_rule_scores(
        self,
        trajectory: torch.Tensor,
        intervals: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...

    def compute_scores(
        self,
        trajectory: torch.Tensor,
        intervals: Optional[torch.Tensor],
        attention_mask: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...

    def detect_boundaries_single(
        self,
        scores: torch.Tensor,
        valid_len: int,
    ) -> PatchSpec:
        ...

    def forward(
        self,
        trajectory: torch.Tensor,
        attention_mask: torch.Tensor,
        intervals: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        ...
```

### 10.2 `utils/adaptive_patch_encoder.py`

```python
class PatchCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        ...

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...


class AdaptivePatchEncoder(nn.Module):
    def __init__(
        self,
        point_dim: int = 2,
        model_dim: int = 128,
        interval_dim: int = 1,
        use_observed_flag: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_refine_layers: int = 2,
        max_patch_len_embed: int = 64,
    ):
        ...

    def build_point_features(
        self,
        trajectory: torch.Tensor,
        intervals: Optional[torch.Tensor],
        observed_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ...

    def masked_patch_pool(
        self,
        point_tokens: torch.Tensor,
        patch2point_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def extract_patch_sequences(
        self,
        point_tokens: torch.Tensor,
        patch2point_mask: torch.Tensor,
    ):
        ...

    def forward(
        self,
        trajectory: torch.Tensor,
        attention_mask: torch.Tensor,
        patch2point_mask: torch.Tensor,
        intervals: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ...
```

### 10.3 `utils/flow_matching.py`

新增：

```python
class PointPatchFusion(nn.Module):
    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        ...

    def forward(
        self,
        point_q: torch.Tensor,
        patch_kv: torch.Tensor,
        patch_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...
```

---

## 11. 推荐的融合逻辑

在 `SingleStageTrajectoryFlow.forward()` 中，推荐按以下逻辑实现：

```python
point_cond = self.cond_embedder(observed_trajectory)
interval_tokens = self.interval_embedder(intervals.unsqueeze(-1))
mask_features = torch.stack([observed_mask, attention_mask], dim=-1)
mask_tokens = self.mask_embedder(mask_features)

if self.use_adaptive_patch:
    patcher_out = self.adaptive_patcher(
        trajectory=observed_trajectory,
        attention_mask=attention_mask,
        intervals=intervals,
        observed_mask=observed_mask,
    )
    encoder_out = self.patch_encoder(
        trajectory=observed_trajectory,
        attention_mask=attention_mask,
        patch2point_mask=patcher_out["patch2point_mask"],
        intervals=intervals,
        observed_mask=observed_mask,
    )

    patch_tokens = encoder_out["patch_tokens"]
    if hasattr(self, "patch_proj"):
        patch_tokens = self.patch_proj(patch_tokens)

    fused_point = self.point_patch_fusion(
        point_q=point_cond,
        patch_kv=patch_tokens,
        patch_padding_mask=encoder_out["patch_padding_mask"],
    )
    cond_tokens = fused_point + interval_tokens + mask_tokens
else:
    cond_tokens = point_cond + interval_tokens + mask_tokens

x = self.x_embedder(x_t) + cond_tokens + self.pos_embed[:, :seq_len, :]
```

要求：

- `fused_point` 维度必须与 `hidden_size` 一致
- `patch_padding_mask` 要正确传入
- 最终输出仍然乘 `attention_mask.unsqueeze(-1)`

---

## 12. 训练与评估兼容性要求

### 12.1 backward compatibility

当 `adaptive_patch.enabled = False` 时：

- 训练 loss 数值趋势与原仓库应大致一致
- 评估脚本必须能正常运行
- 老 checkpoint 不保证兼容，但新模型应可正常从头训练

### 12.2 state dict 行为

不需要兼容旧 `model.pt`。

如果加载 checkpoint 时缺少新模块参数：

- 可以允许 `strict=False`，但要在日志里明确打印 missing/unexpected keys 数量
- 推荐新的 checkpoint 文件名使用：
  - `best_flow_adaptive_patch.pt`
  - `final_flow_adaptive_patch.pt`

---

## 13. 建议的调试输出

建议在模型或训练脚本中暴露以下调试信息：

- `avg_num_patches`
- `avg_patch_len`
- `max_patch_len`
- `avg_patch_score`

如果需要简单实现，可在 `forward()` 里把这些统计缓存到：

```python
self._last_patch_stats = {
    "avg_num_patches": ...,
    "avg_patch_len": ...,
    "max_patch_len": ...,
    "avg_patch_score": ...,
}
```

然后训练脚本在 epoch 末读取。

---

## 14. 验收标准

请确保以下检查全部通过。

### 14.1 代码层面

- 新增模块可 import
- `main.py` 能构建模型
- `evaluate_gpt.py` 能构建模型
- 不存在 shape mismatch
- 不存在 `key_padding_mask` dtype 错误
- 不存在 device mismatch

### 14.2 功能层面

- 开启 adaptive patch 后，单个 batch 能跑通 forward + backward
- `sample_trajectory_flow()` 能跑通
- patch 数量随轨迹复杂度变化，而不是固定值
- padding 区域不会被划入有效 patch

### 14.3 回退层面

- 关闭 adaptive patch 后，模型回退为原行为

### 14.4 最小 smoke test

至少补一个最小测试片段（可以写在文档注释里，也可以写成简单脚本），验证：

- B=2, L=16 的 fake trajectory 输入
- patcher 能输出合法 `patch2point_mask`
- patch encoder 能输出 `[B, P, D]`
- flow model 能输出 `[B, L, 2]`

---

## 15. 推荐的最小测试脚本（可选新增 `debug_smoke_test.py`）

如果方便，可新增一个最小测试脚本：

```python
import torch
from utils.flow_matching import SingleStageTrajectoryFlow

model = SingleStageTrajectoryFlow(
    input_dim=2,
    hidden_size=64,
    depth=2,
    num_heads=4,
    max_len=32,
    adaptive_patch_cfg={
        "enabled": True,
        "score_mode": "rule",
        "hidden_dim": 32,
        "threshold_global": 0.55,
        "threshold_relative": 0.10,
        "min_patch_len": 2,
        "max_patch_len": 8,
        "smooth_kernel": 3,
        "learned_weight": 0.5,
        "patch_encoder_dim": 64,
        "patch_encoder_heads": 4,
        "patch_encoder_layers": 2,
        "patch_dropout": 0.1,
        "use_patch_length_embedding": True,
        "use_point_patch_fusion": True,
        "fusion_heads": 4,
        "fusion_dropout": 0.1,
        "detach_patch_scores": False,
    },
)

B, L = 2, 16
x_t = torch.randn(B, L, 2)
obs = torch.randn(B, L, 2)
intervals = torch.ones(B, L)
attention_mask = torch.ones(B, L)
observed_mask = torch.ones(B, L)
observed_mask[:, 8:] = 0

t = torch.rand(B)
out = model(x_t, t, obs, intervals, attention_mask, observed_mask)
print(out.shape)
assert out.shape == (B, L, 2)
print("smoke test passed")
```

---

## 16. 实现优先级

请按以下顺序完成：

### Phase 1

- 新增 `adaptive_patcher.py`
- 新增 `adaptive_patch_encoder.py`
- 改 `flow_matching.py`
- 改 `config.py`
- 跑通 smoke test

### Phase 2

- 改 `main.py`
- 改 `evaluate_gpt.py`
- 加 patch 统计日志

### Phase 3

- 做 clean-up
- 加注释
- 保证关闭开关后完全回退

---

## 17. 代码风格要求

- 使用现有仓库风格，避免引入花哨框架
- 优先可读性，其次极限优化
- 每个新增类和关键函数写 docstring
- 不要删除当前已有功能
- 只做与 adaptive patch 相关的改动

---

## 18. 本次改造的核心判断标准

最终代码如果满足下面这句，就算方向正确：

> **模型仍然是 point-level flow matching 轨迹补全/预测器，但其条件编码器已经从均匀逐点建模升级为“内容复杂度驱动的可变 patch 编码”。**

如果你在实现过程中发现某一步会把整个生成器改成 patch-level 生成，请停止并退回，因为那已经超出本次改造范围。

---

## 19. 给 Codex 的明确执行要求

请直接修改项目代码并输出：

1. 新增和修改过的文件列表
2. 每个文件的主要改动说明
3. 如果遇到不确定点，优先选择“保持当前 flow 主线不变、只增强条件编码”
4. 不要复制第三方仓库源码
5. 不要恢复 `utils/unitraj.py` 主线
6. 先保证可运行，再考虑进一步优化

