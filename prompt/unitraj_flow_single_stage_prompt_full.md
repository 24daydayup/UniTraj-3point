# 直接修改 UniTraj 项目本体的单阶段 Flow Matching 迁移提示词

请基于原始 UniTraj 项目 **直接修改现有代码本体**，把老师修改版 `unitraj-flow` 中与 **Flow Matching / Rectified Flow、TrajectoryDiT、SparseMoE、AdaLN、spectral consistency loss，以及相关数据抽取、样本构造、mask 构造、重采样、预处理、训练与评估逻辑** 有关的内容，**迁移并合并到原始 `unitraj` 项目现有结构中**。

原始项目路径：

`/home/colorful/UniTraj-ori/`

老师修改版路径：

`/home/colorful/UniTraj-ori/unitraj-flow`

---

## 核心要求

这次的目标不是在原项目旁边新增一套实现，也不是新建一个平行模块或子项目，而是：

- **直接修改 `unitraj` 项目本身**
- 把 `unitraj-flow` 的内容迁移进 `unitraj` 现有工程结构
- 尽量复用、替换、扩展原有模块，而不是额外再挂一套独立系统
- 最终结果应表现为：**原始 UniTraj 项目被直接升级为支持单阶段 Flow Matching / DiT / MoE 方案的版本**
- 不要保留“原始 UniTraj 一套 + unitraj-flow 一套”的双系统并存形态
- 不要在仓库根目录再新建 `unitraj-flow/` 风格的新子项目
- 不要实现成“并行另一套主模型”
- 而是要把这些能力 **真正并入 `unitraj` 自身**

---

## 迁移目标

请将 `unitraj-flow` 中的以下能力，迁移并整合进原始 `unitraj` 项目中：

- Flow Matching / Rectified Flow
- TrajectoryDiT
- SparseMoE
- AdaLN
- spectral consistency loss
- 与 Flow / diffusion 训练相关的数据抽取、样本构造、mask 构造、重采样、padding、interval 处理、预处理逻辑
- 与训练、采样、评估相关的核心实现

但注意：

## 这次必须是**单阶段方案**

不要实现两阶段 coarse-to-refinement 流程，不要保留以下主流程设计：

- 原始 UniTraj 先生成 coarse trajectory
- 再由 flow 模型进行 refinement
- coarse/gt pair 数据导出
- refinement-only dataset
- export_pairs
- coarse -> gt 的二阶段训练目标

这次要求改成：

- **直接让迁移后的 Flow Matching + DiT/MoE 模型承担原始 UniTraj 的轨迹建模任务**
- 直接从 masked / incomplete trajectory 输入学习恢复或预测完整目标轨迹
- 模型直接适配原始 UniTraj 的任务，如：
  - trajectory completion
  - trajectory prediction

---

## 重要实现原则

### 1. 直接改原始 `unitraj` 项目，不要新增平行系统

你需要把 `unitraj-flow` 的内容尽可能：

- 合并进原始 `unitraj` 的现有目录结构
- 接入原有模型、数据、训练、评估入口
- 必要时替换原有实现中的某些模块
- 或在原有模块体系中新增少量文件，但这些新增文件必须是 `unitraj` 项目自身的一部分

不要做下面这些事：

- 不要在根目录再新建一个独立 `unitraj-flow/`
- 不要把 flow 方案做成一个独立训练子系统
- 不要让新代码只存在于一个旁路目录里而不真正进入主项目
- 不要保留两套完全独立的数据流、训练流、评估流长期并存

---

### 2. 只做这一次迁移相关修改，其他部分不要动

要求最小侵入：

- 除了把 `unitraj-flow` 的相关能力迁移进 `unitraj` 所必须做的改动外
- 不要顺手重构其他模块
- 不要修改无关代码风格
- 不要改无关接口
- 不要改无关功能
- 不要改无关训练流程
- 不要改无关评估逻辑

凡是与本次迁移无关的部分，保持原样。

---

### 3. 单阶段 Flow Matching 必须直接适配原始 UniTraj 输入输出

不能依赖 coarse trajectory 作为第二阶段条件输入。

也就是说：

输入应来自原始 UniTraj 的任务输入范式，例如：

- masked trajectory
- visible trajectory
- interval
- attention mask
- 原始任务已有的条件特征

输出应直接是：

- 目标轨迹恢复结果
- 或 Flow Matching 所需的目标状态 / 速度场

请根据原始 UniTraj 的数据流，重新设计并实现：

- 单阶段 `x_0`
- 单阶段 `x_1`
- Flow Matching 路径构造
- 条件输入形式
- 采样逻辑
- completion / prediction 任务下的适配方式

不允许简单沿用二阶段 `coarse -> gt` 定义。

---

## 参考重点文件

请重点阅读并迁移老师版本中的以下文件内容：

- `diffusion_dit_flow.py`
- `evaluate_dit_flow.py`
- `train_flow_quick.py`
- `test_flow.py`
- `diffusion_train.py`
- `evaluate_gpt.py`
- `utils/toy_io.py`
- `utils/unitraj.py`
- `utils/dataset.py`

并结合原始 UniTraj 项目的现有结构，决定这些内容应该如何合并进原项目。

---

## 数据部分是必须项

不要只迁移模型，**必须把数据抽取 / 数据构造 / mask 构造 / 预处理逻辑一起迁移并整合进原项目**。

请重点审查并迁移以下内容：

- 轨迹读取
- 重采样
- 标准化 / 起点平移 / 归一化
- mask 构造
- completion 掩码策略
- prediction 掩码策略
- padding / truncation
- interval 处理
- attention mask 构造
- 训练样本构造
- 验证 / 测试样本构造
- 与评估相关的样本生成逻辑

要求：

- 数据处理逻辑必须成为原始 `unitraj` 项目的一部分
- 不要把这些逻辑散落在训练脚本里
- 要整理成清晰的数据管线或模块
- DataLoader 可以直接消费
- 尽量复用原始 UniTraj 的现有数据集格式和接口
- 同时吸收 `unitraj-flow` 中更合理的数据处理策略

---

## MoE 要求

请支持可配置的 SparseMoE，至少包含：

- `use_moe`
- `num_experts`
- `top_k`
- `mlp_ratio`

要求可启用 / 可关闭。

---

## spectral loss 要求

请支持 spectral consistency loss 的可配置启用 / 关闭与权重控制。

保留其设计目的：

- 约束频域接近目标
- 抑制高频抖动
- 改善轨迹平滑性

---

## 不要写死环境

禁止写死：

- 绝对路径
- GPU 编号
- 私有环境依赖
- 私有账号路径
- 私有数据目录

实现必须具备可移植性。

---

## 优先复用原始 UniTraj 现有能力

请优先复用原始项目中已有的：

- 数据处理接口
- mask 机制
- 评估逻辑
- 工具函数
- 配置风格
- 模型组织方式
- 训练入口风格

必要时再写 adapter，但整体目标是：

> **把 `unitraj-flow` 的内容融进 `unitraj` 本体，而不是外挂一套新系统。**

---

## 开始修改前必须先做的事情

在开始改代码之前，请先审查：

1. 原始 UniTraj 项目的目录结构
2. 原始 UniTraj 的模型入口、数据入口、训练入口、评估入口
3. `unitraj-flow` 中对应的模型、数据、loss、训练、评估、工具实现
4. 哪些内容适合直接迁移
5. 哪些内容需要改造成单阶段版本
6. 哪些原始文件需要修改
7. 哪些 `unitraj-flow` 文件内容应被拆分并并入原始项目哪些位置

然后先输出一份**简短迁移计划**，至少说明：

- 原始 UniTraj 的数据入口在哪里
- 原始 UniTraj 的模型入口在哪里
- 原始 UniTraj 的训练/评估入口在哪里
- `unitraj-flow` 的数据抽取 / mask / preprocessing 逻辑在哪里
- 哪些逻辑会直接迁移
- 哪些逻辑会改成单阶段版本
- 你准备修改哪些原文件
- 你准备新增哪些文件
- 哪些现有部分保持不变

然后再开始实施代码修改。

---

## 代码落地要求

请直接落地成**可运行代码**，不是只给设计建议。

要求包括但不限于：

- 模型代码已经并入原项目
- loss 已接入原项目
- dataset / adapter 已接入原项目
- 数据抽取 / 样本构造 / 预处理已接入原项目
- 训练入口可运行
- 推理 / 采样入口可运行
- 评估入口可运行
- 配置项可控制 MoE / spectral loss
- 单元测试或集成测试可基本验证链路

---

## 最终交付必须包含

最终请输出：

1. **新增/修改文件清单**
2. **每个关键修改点的作用说明**
3. **架构说明**
4. **单阶段 Flow Matching 的 `x_0 / x_1 / 路径 / 条件输入 / 采样逻辑` 说明**
5. **数据抽取与样本构造说明**
6. **completion / prediction 两类任务的 mask 构造说明**
7. **训练、推理、评估运行命令示例**
8. **风险说明**
9. **明确说明哪些地方是直接迁移自 `unitraj-flow`，哪些地方做了单阶段改造**
10. **明确说明为了满足“只做这一项迁移，其他部分不修改”，你采取了哪些最小侵入策略**

---

## 特别约束：禁止跑偏实现

为了避免误解，请严格遵守以下约束：

- 不要把任务实现成“在原项目旁边增加一个新工程”
- 不要新增一套长期并行维护的 flow 子项目
- 不要把结果做成“原始 UniTraj + 新 flow 子系统”双入口架构
- 不要主要通过新建独立目录来规避对原项目的真实整合
- 不要保留 coarse-to-refinement 二阶段设计作为主路径
- 不要只迁移模型定义而忽略数据部分
- 不要只迁移训练脚本而不改造样本构造逻辑
- 不要省略 mask / resample / padding / interval / preprocess 这些数据链路细节

你必须做的是：

> **把 `unitraj-flow` 的关键能力拆解、改造、吸收，并真正并入原始 `unitraj` 项目，使其成为原项目的一部分。**

---

## 验收标准

只有满足以下条件才算完成：

1. 修改是直接发生在原始 `unitraj` 项目中
2. 没有再新建一套平行实现作为主系统
3. `unitraj-flow` 的关键能力已经迁移并整合进 `unitraj`
4. 实现的是**单阶段** Flow Matching，而不是二阶段 refinement
5. 模型能直接对接原始 UniTraj 的 completion / prediction 等任务
6. MoE 可以配置启用/关闭
7. spectral loss 可以配置启用/关闭及权重
8. 数据抽取 / 样本构造 / mask / preprocess 逻辑已经迁移并整理
9. 没有写死路径、GPU、私有环境依赖
10. 原项目中与本次迁移无关的部分保持不变
11. 最终代码是原始 `unitraj` 项目的一部分，而不是外挂模块

---

## 输出风格要求

请按以下顺序输出结果：

### 第一部分：迁移计划
先给出简短迁移计划，再开始修改代码。

### 第二部分：实施过程
说明你修改了哪些原文件、新增了哪些文件、各自作用是什么。

### 第三部分：核心设计说明
说明单阶段 Flow Matching 的：

- `x_0`
- `x_1`
- 路径构造
- 条件输入
- loss 设计
- 采样方式
- completion / prediction 的适配逻辑

### 第四部分：数据链路说明
说明：

- 数据抽取
- 样本构造
- mask 构造
- resample
- normalize
- padding / truncation
- attention mask
- interval 特征
- train / val / test 适配

### 第五部分：运行方式
提供训练、推理、评估命令示例。

### 第六部分：风险与兼容性
说明可能风险、兼容性问题、以及你如何保证“其他部分不修改”。

---