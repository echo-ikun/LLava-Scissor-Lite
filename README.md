# LLava-Scissor-Lite

ScissorLite 是我整理的一份轻量版 LLaVA-Scissor 复现与实验项目。

目的是把 **LLaVA-Scissor 最关键的视频 token 压缩路径** 拆清楚：哪些地方应该复用官方实现，哪些地方才是我们真正要研究和改进的压缩模块。

## 方法概览

LLaVA-Scissor 的出发点很直接：视频视觉 token 很多，但语义高度重复，不必把所有 token 都送进语言模型。ScissorLite 保留原方法的两阶段压缩思路：

```text
Video frames
  -> Vision tower
  -> Projector
  -> Spatial pooling
  -> Spatial SCC compression
  -> Temporal SCC compression
  -> Original-token merge
  -> LLM generation
```

其中 SCC 是 Semantic Connected Components。可以把它理解成：先根据余弦相似度构建 token 图，再把语义相近的 token 合成连通分量。

压缩过程分三步：

1. **Spatial compression**：每一帧内部做 SCC，合并语义相近的 patch token。
2. **Temporal compression**：跨帧继续做 SCC，合并不同帧之间重复的语义 token。
3. **Original-token merge**：每个原始 token 找最近的压缩 token，再聚合回去，减少信息损失。

## 维度变化

以当前本地 0.5B 模型、4 帧短视频测试为例：

```text
输入视频帧
  frames = 4

Vision tower 输出
  [4, 729, 1152]
  729 = 27 x 27 patch tokens

Projector 后
  [4, 729, 896]

2D pooling 后
  [4, 196, 896]
  196 = 14 x 14 pooled tokens

Flatten video tokens
  [784, 896]
  784 = 4 x 196

Spatial SCC compression
  [690, 896]

Temporal SCC compression + token merge
  [524, 896]

送入 LLM
  text tokens + 524 visual tokens   (其中，每个token的隐藏维度均为896)
```

这个短测试里，视觉 token 从 `784` 压到 `524`，并且与原 LLaVA-Scissor 路径保持一致。

## 架构图

```text
                         ┌──────────────────────────────┐
                         │         ScissorLite          │
                         │        inference.py          │
                         └───────────────┬──────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
          ┌─────────▼─────────┐                     ┌─────────▼─────────┐
          │ official backend  │                     │   nano backend    │
          │  推荐实验入口       │                     │  教学/对照入口      │
          └─────────┬─────────┘                     └─────────┬─────────┘
                    │                                         │
     复用官方 load_pretrained_model              手写更小的 vision/projector/LLM
     复用官方 image_processor                    方便理解端到端链路
     复用官方 prompt/token 拼接                   不作为严格效果对比入口
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │        compression/           │
                         │  Spatial SCC                  │
                         │  Temporal SCC                 │
                         │  Original-token merge         │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │          LLM generate         │
                         └───────────────────────────────┘
```

## 项目结构

```text
LLava-Scissor-Lite/
├── README.md                 -> 项目说明，从这里开始读
├── plan.md                   -> 设计记录和迭代过程
│
├── inference.py              -> CLI 入口：python inference.py
│                               official backend、视频读取、返回对象都在这里
├── config.py                 -> 运行配置、默认 checkpoint、默认生成参数
├── smoke_test.py             -> official backend 端到端短测试
│
├── compression/              -> token 压缩核心
│   ├── compressor.py         -> spatial / temporal / original-token merge
│   ├── components.py         -> Union-Find + Semantic Connected Components
│   ├── config.py             -> 压缩参数
│   ├── stats.py              -> 压缩统计信息
│   ├── smoke_test.py         -> 压缩模块独立 smoke test
│   └── parity_test.py        -> 与原始局部实现的一致性测试
│
└── backends/                 -> 两种推理 backend
    ├── official.py           -> 推荐实验 backend，指向 inference.py
    └── nano.py               -> 极简教学 backend，手写较小端到端链路
```

## Backend 选择

默认使用 `official`。

`official` 复用原项目的模型加载、图像预处理、prompt 构造、`<image>` token 插入和 `model.generate`。这样可以让实验变量尽量只来自 token 压缩算法。

`nano` 是教学版。它手写一个更小的端到端链路，包括 SigLIP、projector、Qwen2 和 visual token 插入。它能跑通，也尽量贴近官方路径，但不建议用它做严格效果对比。

## 快速运行

准备好 Python 环境后，从仓库根目录运行。下面用占位路径表示，你可以替换成自己的环境、仓库和视频路径：

```text
<PYTHON_BIN>      例如 /path/to/conda/env/bin/python
<REPO_ROOT>       例如 /path/to/LLaVA-Scissor
<VIDEO_PATH>      例如 /path/to/video.mp4
```

```bash
cd <REPO_ROOT>

# 1. 运行原版默认 LLaVA-Scissor (Baseline)
CUDA_VISIBLE_DEVICES=0 <PYTHON_BIN> inference.py \
  --backend official \
  --video <VIDEO_PATH> \
  --question "Describe briefly."

# 2. 改进一：自适应阈值 (Adaptive Tau)
# 作用：根据当前帧内特征的复杂度和冗余度动态调整 tau 阈值，简单画面高压缩，复杂场景低压缩，更合理分配算力。
CUDA_VISIBLE_DEVICES=0 <PYTHON_BIN> inference.py \
  --backend official \
  --video <VIDEO_PATH> \
  --question "Describe briefly." \
  --scissor-adaptive-tau

# 3. 改进二：中心度合并与软分配 (Centrality & Soft Refill)
# 作用：组件合并时由直接取特征平均（mean）改进为通过特征相似度的中心度（centrality）加权，并且将原始 token 放回时采用 Softmax（soft refill），从而保留更多画面重要细节。
CUDA_VISIBLE_DEVICES=0 <PYTHON_BIN> inference.py \
  --backend official \
  --video <VIDEO_PATH> \
  --question "Describe briefly." \
  --scissor-component-merge centrality \
  --scissor-original-merge-strategy soft

# 4. 改进三：窗口化时序压缩 (Windowed Temporal SCC)
# 作用：限制时序上只在滑动窗口（例如近 4 帧内）找相似 Token 压缩，防止长视频中不同时间出现的无关相似画面被错误地压在一起。
CUDA_VISIBLE_DEVICES=0 <PYTHON_BIN> inference.py \
  --backend official \
  --video <VIDEO_PATH> \
  --question "Describe briefly." \
  --scissor-temporal-strategy windowed \
  --scissor-temporal-window-size 4
```

`reproduce_scissor` 是干嘛的？
外层庞大的 LLaVA 主库如果想运行我们 `LLava_Scissor_Lite` 最新写的这些压缩逻辑，就需要通过一个桥接模块，`reproduce_scissor` 就是这个**兼容桥梁**的作用。主库源码通过 `import reproduce_scissor.compression` 就能无缝调用这里最新的压缩算法。

预期会看到类似日志：

```text
flat input size:  torch.Size([784, 896])
average token first step:  690.0 average token second step:  524.0
flat after zip torch.Size([524, 896])
backend: official
frames: (4, 320, 480, 3)
The video shows a sequence of objects placed on a flat surface, with the first
```

## 测试

### 1. 压缩模块 smoke test

检查 `compression/` 是否可以独立工作：

```bash
cd <REPO_ROOT>

<PYTHON_BIN> -m compression.smoke_test --device cpu
```

### 2. 压缩逻辑一致性测试

对比重构版压缩逻辑和原始局部实现：

```bash
<PYTHON_BIN> -m compression.parity_test --device cpu
```

当前验证结果：

```text
11 通过, 0 失败
```

### 3. official backend smoke test

检查官方复现路径：

```bash
CUDA_VISIBLE_DEVICES=0 <PYTHON_BIN> smoke_test.py
```

### 4. nano backend 对照运行

```bash
CUDA_VISIBLE_DEVICES=0 <PYTHON_BIN> inference.py \
  --backend nano \
  --video <VIDEO_PATH> \
  --question "Describe briefly." \
  --max-frames 4 \
  --max-new-tokens 16 \
  --seed 123
```



## 模型和视频路径

默认 checkpoint 可以在 `config.py` 中修改，也可以在运行时显式指定：

```bash
--checkpoint /path/to/checkpoint
```

或使用环境变量：

```bash
export REPRODUCE_SCISSOR_CHECKPOINT=/path/to/checkpoint
```

如果本地没有模型并且需要联网下载，可以加：

```bash
--online
```
