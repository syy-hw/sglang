# Gemma 4 NPU A3 Sliding Window Attention (SWA) 适配研究报告

> 日期：2026-05-06
> 分支：feat/ascend-npu-gemma4
> 状态：研究阶段完成，待进入实现阶段

---

## 目录

1. [背景与问题](#1-背景与问题)
2. [GPU 端 SWA 架构](#2-gpu-端-swa-架构)
3. [NPU 端 SWA 架构](#3-npu-端-swa-架构)
4. [同事 POC 代码分析](#4-同事-poc-代码分析)
5. [根因分析](#5-根因分析)
6. [GPU vs NPU 对比](#6-gpu-vs-npu-对比)
7. [实现建议](#7-实现建议)

---

## Gemma 4 模型架构概述

> 本章节为理解后续 SWA 内容提供模型架构上下文。
> 数据来源：HuggingFace config.json、Google 官方博客、sglang 代码库。

### 1. 模型变体总览

Gemma 4 于 2026 年 4 月发布（Apache 2.0 许可），包含 **四种模型变体**、**三种不同架构**：

| 变体 | 架构类型 | 总参数量 | 活跃参数量 | 层数 | 上下文长度 | 模态 | 滑动窗口大小 |
|------|---------|---------|-----------|------|-----------|------|------------|
| **E2B** | Dense + PLE | 5.1B | 2.3B | 35 | 128K | Text, Image, Audio | 512 |
| **E4B** | Dense + PLE | 8B | 4.5B | 42 | 128K | Text, Image, Audio | 512 |
| **26B A4B** | MoE | 25.2B | 3.8B | 30 | 256K | Text, Image | 1024 |
| **31B** | Dense | 30.7B | 30.7B | 60 | 256K | Text, Image | 1024 |

> PLE = Per-Layer Embedding（每层嵌入），使总参数量大于有效参数量。

### 2. Transformer 解码器详细参数

| 参数 | E2B | E4B | 26B A4B (MoE) | 31B (Dense) |
|------|-----|-----|---------------|-------------|
| `hidden_size` | 1536 | 2560 | 2816 | 5376 |
| `num_hidden_layers` | 35 | 42 | 30 | 60 |
| `num_attention_heads` | 8 | 8 | 16 | 32 |
| `num_key_value_heads`（SWA 层） | 1 | 2 | 8 | 16 |
| `num_global_key_value_heads`（Full 层） | 1 (=kv) | 2 (=kv) | 2 | 4 |
| `head_dim`（SWA 层） | 256 | 256 | 256 | 256 |
| `global_head_dim`（Full 层） | 512 | 512 | 512 | 512 |
| `intermediate_size` | 6144 | 10240 | 2112 | 21504 |
| `moe_intermediate_size` | — | — | 704 | — |
| `vocab_size` | 262144 | 262144 | 262144 | 262144 |
| `rms_norm_eps` | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| `final_logit_softcapping` | 30.0 | 30.0 | 30.0 | 30.0 |
| `hidden_activation` | gelu_pytorch_tanh | gelu_pytorch_tanh | gelu_pytorch_tanh | gelu_pytorch_tanh |

### 3. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Gemma 4 Transformer Decoder                      │
│                                                                     │
│  Input Tokens ──► Token Embedding (262K vocab)                      │
│                    + Per-Layer Embedding (PLE, E2B/E4B only)        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Decoder Layer × N (交替 SWA / Full Attention)              │    │
│  │                                                             │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  Attention Block                                      │  │    │
│  │  │                                                       │  │    │
│  │  │  SWA 层 (每 6 层中 5 层):                              │  │    │
│  │  │    head_dim=256, KV heads=N/16                        │  │    │
│  │  │    sliding_window=512(E2B/E4B)/1024(26B/31B)          │  │    │
│  │  │    RoPE: theta=10000, full rotation                   │  │    │
│  │  │    + Attention Sinks (可学习偏置)                      │  │    │
│  │  │                                                       │  │    │
│  │  │  Full 层 (每 6 层中 1 层 + 最后一层):                   │  │    │
│  │  │    global_head_dim=512, global_KV_heads=N/16          │  │    │
│  │  │    无窗口限制 (全局注意力)                              │  │    │
│  │  │    p-RoPE: theta=1M, partial_rotary_factor=0.25       │  │    │
│  │  │    K=V 权重共享 (26B/31B)                              │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  │                          │                                  │    │
│  │                          ▼                                  │    │
│  │  ┌───────────────────────────────────────────────────────┐  │    │
│  │  │  FFN / MoE Block                                      │  │    │
│  │  │                                                       │  │    │
│  │  │  Dense 变体 (E2B/E4B/31B):                            │  │    │
│  │  │    标准 MLP (GELU tanh)                                │  │    │
│  │  │    E2B: use_double_wide_mlp (中间层 2x)                │  │    │
│  │  │                                                       │  │    │
│  │  │  MoE 变体 (26B A4B only):                             │  │    │
│  │  │    128 路由专家 + 1 共享专家                            │  │    │
│  │  │    top-k=8 (每 token 激活 8 个专家)                    │  │    │
│  │  │    expert intermediate_size=704                        │  │    │
│  │  │    稀疏度 ~85.4%                                       │  │    │
│  │  └───────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ──► RMSNorm ──► Final Logit Softcapping (30.0) ──► Output          │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. 混合注意力机制详解

Gemma 4 的核心创新是 **双几何注意力（Dual-Geometry Attention）**：

```
Layer:  0    1    2    3    4    5    6    7    8    9   10   11   ...
Type:  SWA  SWA  SWA  SWA  SWA Full  SWA  SWA  SWA  SWA  SWA Full  ...
       ├─────── 5:1 交替 ───────┤    ├─────── 5:1 交替 ───────┤

最后一层始终为 Full Attention。
```

| 特性 | SWA 层（滑动窗口） | Full 层（全局注意力） |
|------|-------------------|---------------------|
| 出现频率 | 每 6 层中 5 层 | 每 6 层中 1 层 + 最后一层 |
| `head_dim` | 256 | 512 |
| KV heads | `num_key_value_heads` | `num_global_key_value_heads`（更少） |
| RoPE | 标准, theta=10000, 全维度旋转 | p-RoPE, theta=1000000, 仅旋转 25% 维度 |
| 窗口限制 | 512 (E2B/E4B) / 1024 (26B/31B) | 无限制（全局） |
| Attention Sinks | ✅（可学习标量偏置） | ❌ |
| K=V 共享 | ❌ | ✅（26B/31B 变体） |

### 5. MoE 架构（仅 26B A4B）

```
                    Input Token
                        │
                        ▼
              ┌─────────────────┐
              │  Gemma4Router   │
              │  (softmax 路由)  │
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │ Expert 0 │  │ Expert 1 │  │Expert 127│  × 128 路由专家
   │ MLP(704) │  │ MLP(704) │  │ MLP(704) │  每个: intermediate=704
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │              │              │
        └──────selected top-8────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Shared Expert  │  × 1 共享专家（始终激活）
              │  MLP(704)       │
              └────────┬────────┘
                       │
                       ▼
                Weighted Sum
                       │
                       ▼
                  FFN Output
```

- **路由专家**：128 个，每个 `moe_intermediate_size=704`
- **共享专家**：1 个，始终激活
- **top-k 路由**：每 token 选择 8 个专家
- **稀疏度**：~85.4%（仅 14.6% 参数活跃）
- 代码实现：`gemma4_causal.py` 第 104-189 行，包含 `Gemma4Router` 和专家缩放

### 6. 与 Gemma 3 的架构差异

| 特性 | Gemma 3 | Gemma 4 |
|------|---------|---------|
| 模型变体 | 1B, 4B, 12B, 27B | E2B, E4B, 26B A4B, 31B |
| 架构类型 | 全部 Dense | Dense + MoE 混合 |
| `head_dim` | 统一 128 | 双几何：SWA=256, Full=512 |
| 全局层 RoPE | 线性缩放 8x | p-RoPE（仅旋转 25% 维度） |
| K=V 权重共享 | 无 | 全局层启用（26B/31B） |
| MoE | 无 | 128 experts, top-k=8（26B A4B） |
| 每层嵌入（PLE） | 无 | E2B/E4B 专有 |
| KV 共享层 | 无 | E2B=20 层, E4B=18 层 |
| Vocab 大小 | 256K | 262K |
| 许可证 | Gemma 使用条款 | Apache 2.0 |
| SWA 窗口 | 512/4096/8192 | 512 (小模型) / 1024 (大模型) |

### 7. 代码库关键文件

| 文件 | 内容 | 行号 |
|------|------|------|
| `models/gemma4_causal.py` | 模型主实现，包含注意力、MoE、PLE | 全文 |
| `models/gemma4_causal.py:104-189` | `Gemma4Router` + MoE 路由逻辑 | 104-189 |
| `models/gemma4_causal.py:221-232` | 混合注意力层定义（`layer_types`, `sliding_window`） | 221-232 |
| `models/gemma4_causal.py:394-463` | PLE 机制（门控 + 投影） | 394-463 |
| `models/gemma4_causal.py:422-426` | 双宽 MLP 支持 | 422-426 |
| `models/gemma4_causal.py:474` | `enable_moe_block` MoE 开关 | 474 |
| `configs/model_config.py` | SWA 检测、层 ID 分配、窗口大小解析 | 431-441, 989-994 |

### 8. SWA 在整体架构中的位置

> **过渡说明**：以上架构概述表明，Gemma 4 的 SWA 是其 **双几何注意力** 设计的核心组成部分。SWA 层占全部 Transformer 层的 5/6，负责高效捕获局部上下文；Full 层占 1/6，负责全局信息传递。两者共享同一套 KV 缓存基础设施（`SWAKVPool` 双池架构），但具有不同的头维度、KV 头数和 RoPE 配置。理解这一架构背景后，以下章节将详细分析 SWA 在 GPU 和 NPU 上的具体实现。

---

## 1. 背景与问题

为 Ascend NPU A3 适配 Gemma 4 系列模型的 Sliding Window Attention (SWA)。已知问题：同事 POC 版本单次 curl 正常，长跑精度异常（零点几），疑似 SWA 逻辑问题。

### Gemma 4 的 SWA 特性

- Gemma 4 使用混合 SWA 架构（hybrid SWA）：部分层使用滑动窗口注意力，部分层使用全局注意力
- 支持 Attention Sinks：可学习的标量偏置，补偿因 SWA 淘汰导致的注意力分数偏移
- 窗口大小从 HuggingFace config 的 `sliding_window` 字段读取

---

## 2. GPU 端 SWA 架构

### 2.1 SWA 核心原理

#### 窗口大小如何确定

窗口大小来自多个来源，按以下优先级解析：

1. **模型层面覆盖**：如果模型类有 `get_attention_sliding_window_size()` 方法，优先使用
2. **配置层面（混合 SWA）**：如果 `model_config.is_hybrid_swa` 为 true 且 `model_config.sliding_window_size` 已设置，使用该值
3. **Attention chunk size 回退**：否则使用 `model_config.attention_chunk_size`

`ModelConfig` 中的 `_get_sliding_window_size()` 方法按顺序检查配置键 `"sliding_window_size"`、`"sliding_window"`、`"window_size"`（`model_config.py` 第 989-994 行）。

#### Token 溢出处理

当 Token 超出滑动窗口时，两个独立机制：

**a) 运行时注意力掩码（GPU 端）：** FlashInfer 后端使用 `window_left` 参数限制每个 Token 可见的 KV 缓存范围。

- **Decode**：`FlashInferIndicesUpdaterDecode.update_sliding_window()` 将 `paged_kernel_lens` 限制为 `sliding_window_size + 1`，计算 `kv_start_idx = seq_lens - paged_kernel_lens`
- **Prefill**：`FlashInferIndicesUpdaterPrefill.update_sliding_window()` 设置 `paged_kernel_lens = min(seq_lens, sliding_window_size + seq_lens - prefix_lens)`

**b) 显式 SWA KV 缓存淘汰（调度器端）：** `ScheduleBatch.maybe_evict_swa()` 在每次 decode 步骤或分块 prefill 步骤期间被调用。计算 `new_swa_evicted_seqlen = max(req.swa_evicted_seqlen, pre_len - sliding_window_size - page_size)`。额外 `page_size` 减去确保至少一个非墓碑页面仍可用于基数树重用。

#### 关键概念：`paged_kernel_lens`

`paged_kernel_lens` 和 `kv_start_idx` 是 FlashInfer `call_begin_forward` API 的参数（`flashinfer_backend.py:1106-1109`）。两者配合定义内核实际访问的 KV cache 范围：

```
KV Cache:  [token_0, token_1, ..., token_N-1]
                                   ↑ kv_start_idx
              <--- paged_kernel_lens --->
```

**Decode 路径**（`flashinfer_backend.py:1031-1041`）：
```python
# Sliding window attention
paged_kernel_lens_tmp = torch.clamp(
    seq_lens, max=self.sliding_window_size + 1
)
kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
```

其中 `seq_lens` 来源于 `schedule_batch.py:1678`：
```python
seq_lens = [len(r.fill_ids) for r in reqs]
```

- 例：序列长度 10000，窗口 4096 → `paged_kernel_lens = min(10000, 4097) = 4097`，`kv_start_idx = 10000 - 4097 = 5903`
- **关于 +1**：代码中无 inline comment。从行为推断——decode 步骤中 `seq_lens` 已包含刚生成的 1 个新 token，而 `sliding_window_size` 是模型 config 定义的窗口大小，`+1` 使内核访问范围覆盖当前新 token

**Prefill 路径**（`flashinfer_backend.py:1315-1325`）：
```python
# window attention use paged only
paged_kernel_lens = torch.minimum(
    seq_lens,
    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
)
kv_start_idx = seq_lens - paged_kernel_lens
```

- `seq_lens - prefix_lens`：由 `schedule_batch.py:1747` 的 assertion 确认等于 `req.extend_input_len`（本次 extend 新增 token 数）：
  ```python
  # schedule_batch.py:1678, 1680 — seq_lens 和 prefix_lens 的来源
  seq_lens = [len(r.fill_ids) for r in reqs]          # fill_ids 总长度
  prefix_lens = [len(r.prefix_indices) for r in reqs]  # radix tree 已缓存的前缀 token 数

  # schedule_batch.py:1747 — assertion 确认差值 = extend_input_len
  assert seq_len - pre_len == req.extend_input_len
  ```
- `sliding_window_size + (seq_lens - prefix_lens)`：代码中无 inline comment。从行为推断——prefill 中新增的每个 query token 需要看到窗口内的历史 token 加上本次 extend 的所有新 token，因此 `paged_kernel_lens` 上限 = 窗口大小 + 新增 token 数
- `min(seq_lens, ...)`：当总序列长度短于窗口时，不截断

**为什么需要 `paged_kernel_lens` 和 `kv_start_idx`？**

FlashInfer 分页注意力内核默认会访问**整个 KV cache**。对 Full 层这没问题，但 SWA 层只应看到窗口内的 token——如果不限制，SWA 层就变成了 Full 层，失去了滑动窗口节省计算的意义。`paged_kernel_lens` 和 `kv_start_idx` 就是告诉内核"从哪个位置开始看，看多少个"。

```
=== Decode 场景 ===
Gemma 4 31B, sliding_window_size=1024, 正在 decode 第 5000 个 token

seq_lens = 5000 (已包含刚生成的 token)

如果不限制，内核访问 KV cache 的 [0, 5000) → 5000 个 token，与 Full 层无异
实际限制：
  paged_kernel_lens = min(5000, 1024+1) = 1025
  kv_start_idx       = 5000 - 1025      = 3975
  内核只访问 [3975, 5000) → 1024 个历史 + 1 个当前 = 1025 个 token

  [0 .......... 3975 ................. 5000]
                ↑ kv_start_idx           ↑ seq_lens
                <---- 1025 tokens ------>
                (paged_kernel_lens)

=== Prefill 场景 ===
同一个模型，prefill 一个 3000 token 的 prompt，radix tree 命中了 2000 token 前缀

seq_lens     = 3000 (总长度)
prefix_lens  = 2000 (已缓存前缀)
新增 token   = seq_lens - prefix_lens = 1000

sliding_window_size = 1024
paged_kernel_lens = min(3000, 1024 + 1000) = min(3000, 2024) = 2024
kv_start_idx      = 3000 - 2024            = 976

为什么是 2024？
  1024 (窗口大小)：新增 token 需要看到窗口内的历史 token
  + 1000 (新增数)：新增 token 之间也需要互相看到 (causal mask)
  = 2024 个 KV token 需要被访问

  [0 ... 976 ......................... 3000]
          ↑ kv_start_idx                 ↑ seq_lens
          <------- 2024 tokens ---------->
```

#### 关键概念：`new_swa_evicted_seqlen`

`new_swa_evicted_seqlen` 是 `_evict_swa()` 函数中的局部变量（`schedule_batch.py:2682`），表示本次淘汰后 SWA KV cache 的释放边界。

`_evict_swa` 函数头有 inline comment 说明其目的（`schedule_batch.py:2664`）：
```python
# For swa radix cache, we need to evict the tokens that are not in the
# tree cache and also not in the sliding window
```

计算逻辑（`schedule_batch.py:2671-2684`）：
```python
# Subtract an extra page_size so the eviction frontier never reaches the
# radix tree insert boundary (page_floor(seq_len)). This keeps at least one
# page of non-evicted SWA KV for the tree to store as a non-tombstone node,
# preserving cache reuse in multi-turn scenarios.
# See also: _insert_helper case 3 in swa_radix_cache.py (defensive counterpart).
new_swa_evicted_seqlen = max(
    req.swa_evicted_seqlen,
    pre_len - sliding_window_size - self.tree_cache.page_size,
)

if self.tree_cache.page_size > 1:
    new_swa_evicted_seqlen = (
        new_swa_evicted_seqlen // self.tree_cache.page_size
    ) * self.tree_cache.page_size
```

各部分来源：

- **`pre_len`**：`_evict_swa` 的参数，在不同调用路径中来源不同：
  - Decode 路径（`schedule_batch.py:2644`）：`self._evict_swa(req, req.seqlen - 1)`，传入 `req.seqlen - 1`
  - Extend 路径（`schedule_batch.py:2646`）：`self._evict_swa(req, pre_len)`，其中 `pre_len = self.prefix_lens[idx]`，即 radix tree 已缓存的前缀长度（`schedule_batch.py:1680`：`prefix_lens = [len(r.prefix_indices) for r in reqs]`）

- **`pre_len - sliding_window_size`**：代码中无 inline comment。从表达式推断——窗口覆盖范围是 `[pre_len - sliding_window_size, pre_len)`，此位置之前的 token 不在窗口内

- **再减 `page_size`**：代码 inline comment 原文——*"Subtract an extra page_size so the eviction frontier never reaches the radix tree insert boundary (page_floor(seq_len)). This keeps at least one page of non-evicted SWA KV for the tree to store as a non-tombstone node, preserving cache reuse in multi-turn scenarios."* 另见 `swa_radix_cache.py` 中 `_insert_helper case 3`（defensive counterpart）

- **`max(req.swa_evicted_seqlen, ...)`**：`req.swa_evicted_seqlen` 初始值为 0（`schedule_batch.py:746`：`self.swa_evicted_seqlen = 0`），每次淘汰后更新为 `new_swa_evicted_seqlen`（`schedule_batch.py:2691`：`req.swa_evicted_seqlen = new_swa_evicted_seqlen`）。`max` 保证淘汰边界只向前推进

- **页对齐**（`schedule_batch.py:2681-2684`）：当 `page_size > 1` 时，`new_swa_evicted_seqlen` 向下对齐到 `page_size` 的整数倍

示例（窗口 4096，page_size 16，pre_len 8000）：
```
序列位置: [0 ......... 3904 ......... 8000]
                       ↑ new_swa_evicted_seqlen = 8000 - 4096 - 16 = 3888
                       （实际会页对齐到 3888 // 16 * 16 = 3888）
[0, 3888) 范围的 SWA KV cache 可安全释放
[3888, 8000) 范围的 SWA KV cache 必须保留（在窗口内）
```

**为什么需要淘汰？为什么必须减 `page_size`？**

```
=== 为什么需要 SWA KV 淘汰？ ===

SWA pool 的容量是有限的（比 full pool 小得多）。
如果只写入不释放，decode 到第 10000 个 token 时 SWA pool 仍有 10000 条 KV，
SWA pool 会溢出。淘汰就是把"窗口已经滑过去"的 KV 从 SWA pool 中释放，
回收 slot 给新 token 使用。

=== 为什么必须减 page_size？不减会怎样？ ===

场景：sliding_window_size=4096, page_size=16, seq_len=8000
  radix tree 的 insert boundary = page_floor(8000) = 8000 // 16 * 16 = 8000

如果不算 page_size：
  eviction frontier = 8000 - 4096 = 3904
  释放 [0, 3904) 的 SWA KV → 看起来没问题

但如果 seq_len 继续增长到 8016（刚好一个 page 边界）：
  radix tree insert boundary = 8016
  上次 eviction frontier = 8016 - 4096 = 3920

问题在多轮对话中暴露：
  第一轮：用户发 8000 token → radix tree 缓存了前缀
  第二轮：用户追加 prompt → radix tree 尝试复用第一轮的缓存前缀
          → 但前缀对应的 SWA KV 已经被淘汰了！
          → 节点变成"墓碑"（SWA KV 已释放，full KV 仍在）
          → 无法复用缓存，必须重新计算 → 性能退化

减一个 page_size 留出安全裕量：
  → radix tree 至少能保留一页非墓碑 SWA KV
  → 多轮对话中第二轮可以复用第一轮的缓存前缀
  → 这就是代码 comment 中 "preserving cache reuse in multi-turn scenarios" 的含义
```

#### Sink Token

Sink token 的启用由 `ModelConfig.has_attention_sinks` 控制（`model_config.py:444`）：
```python
self.has_attention_sinks = self._detect_attention_sinks()
```

在注意力层中，sinks 作为可选参数传递（`radix_attention.py:161`）：
```python
sinks: Optional[torch.Tensor] = None,
```
当 `sinks is not None` 时，通过 kwargs 传入注意力内核（`radix_attention.py:178`）：
```python
if sinks is not None:
    kwargs["sinks"] = sinks
```

代码中无 inline comment 解释 sinks 的语义。从参数传递链和模型实现推断——sinks 是每头、每 token 的可学习标量，在 softmax 计算中作为偏置项，补偿因 SWA 淘汰导致的注意力分数偏移。sinks **不**存储在 KV 缓存中，是注意力内核在 softmax 阶段应用的标量值。

**为什么需要 Sink Token？**

```
场景：Gemma 4 31B, sliding_window_size=1024, 序列长度 5000

=== 没有 sink token 时 ===
  SWA 层只能看到 token [3976, 5000)
  token [0, 3976) 的 KV 被 attention 完全忽略
  softmax 分母只包含窗口内 token 的 exp(score) 之和
  → 分母变小，注意力分布偏移
  → 模型丢失了"全局上下文"信号，长序列输出质量下降

=== 有 sink token 时 ===
  每个 attention head 有一个可学习的标量 s (一个 float)
  softmax 计算变为：
    attention_weights = softmax(scores + s)
  → exp(s) 补偿了被淘汰 token 的贡献
  → 即使 SWA 只看窗口内 token，注意力分布仍然平衡

  sink 是标量（不是 KV），不占用 KV cache 空间
  → 内存开销极小：每层 × 每头 × 1 个 float

=== 为什么不是所有 SWA 模型都用？ ===
  从 _detect_attention_sinks() 代码（model_config.py:453-470）可知：
  - GptOss: 总是使用 sinks
  - MiMoV2: 仅当 config 中 add_swa_attention_sink_bias=True 时使用
  - Gemma4: 该函数返回 False → Gemma4 不使用 sinks
```

### 2.2 初始化阶段

#### 参数/数据结构

##### `ModelConfig`（`configs/model_config.py`）

| 字段 | 代码来源 |
|------|---------|
| `is_hybrid_swa` | `model_config.py:431-434`：`is_hybrid_swa_model(architectures) and not disable_hybrid_swa`。`is_hybrid_swa_model()`（`:1567-1579`）检查架构是否在 `hybrid_swa_archs` 集合中 |
| `sliding_window_size` | `model_config.py:989-994`：`_get_sliding_window_size()` 按优先级读取 `hf_text_config` 的 `sliding_window_size`、`sliding_window`、`window_size` |
| `swa_attention_layer_ids` | `model_config.py:1582-1642`：`get_hybrid_layer_ids()` 返回值。Gemma4 从 `hf_text_config.layer_types` 筛选 `"sliding_attention"` |
| `full_attention_layer_ids` | 同上。Gemma4 从 `layer_types` 筛选 `"full_attention"` |
| `has_attention_sinks` | `model_config.py:453-470`：`_detect_attention_sinks()` 返回值。函数 docstring 原文：*"Attention sinks are per-head scalars added to the softmax denominator to compensate for evicted KV-cache entries under sliding-window attention. Not every hybrid-SWA model uses them."* |

```python
# model_config.py:431-434
self.is_hybrid_swa = (
    is_hybrid_swa_model(self.hf_config.architectures)
    and not self.disable_hybrid_swa
)

# model_config.py:1567-1579
def is_hybrid_swa_model(model_architectures: List[str]):
    hybrid_swa_archs = {
        "Llama4ForConditionalGeneration",
        "GptOssForCausalLM",
        *MIMO_V2_MODEL_ARCHS,
        "MiMoV2MTP",
        "Step3p5ForCausalLM",
        "Step3p5MTP",
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
    }
    return any(arch in hybrid_swa_archs for arch in model_architectures)

# model_config.py:989-994
def _get_sliding_window_size(self) -> Optional[int]:
    for key in ("sliding_window_size", "sliding_window", "window_size"):
        value = getattr(self.hf_text_config, key, None)
        if value is not None:
            return value
    return None

# model_config.py:1582-1600 (Gemma4 分支)
elif (
    "Gemma4ForCausalLM" in model_architectures
    or "Gemma4ForConditionalGeneration" in model_architectures
):
    layer_types = getattr(hf_text_config, "layer_types", [])
    swa_attention_layer_ids = [
        i for i, x in enumerate(layer_types) if x == "sliding_attention"
    ]
    full_attention_layer_ids = [
        i for i, x in enumerate(layer_types) if x == "full_attention"
    ]

# model_config.py:453-470
def _detect_attention_sinks(self) -> bool:
    """Check whether the model uses learned attention sinks.

    Attention sinks are per-head scalars added to the softmax denominator
    to compensate for evicted KV-cache entries under sliding-window
    attention.  Not every hybrid-SWA model uses them.
    """
    archs = self.hf_config.architectures or []
    if "GptOssForCausalLM" in archs:
        return True
    if any(a in archs for a in (*MIMO_V2_MODEL_ARCHS, "MiMoV2MTP")):
        return getattr(
            self.hf_text_config, "add_swa_attention_sink_bias", False
        ) or getattr(self.hf_text_config, "add_full_attention_sink_bias", False)
    return False
```

##### `ModelRunner`（`model_executor/model_runner.py`）

| 字段 | 代码来源 |
|------|---------|
| `is_hybrid_swa` | `model_runner.py`：直接赋值 `model_config.is_hybrid_swa`，供 attention backend 和 memory pool 使用 |
| `sliding_window_size` | `model_runner.py:1495-1508`：三级优先级解析——`model.get_attention_sliding_window_size()` > `model_config.sliding_window_size` > `model_config.attention_chunk_size` |
| `full_max_total_num_tokens` | `model_runner_kv_cache_mixin.py:725`：`self.full_max_total_num_tokens = config.full_max_total_num_tokens`。代码中无 inline comment。从参数名推断：full KV pool 的最大 token 容量 |
| `swa_max_total_num_tokens` | `model_runner_kv_cache_mixin.py:726`：`self.swa_max_total_num_tokens = config.swa_max_total_num_tokens`。代码中无 inline comment。从参数名推断：SWA KV pool 的最大 token 容量 |

```python
# model_runner.py:1495-1508
self.sliding_window_size = None
if hasattr(self.model, "get_attention_sliding_window_size"):
    self.sliding_window_size = self.model.get_attention_sliding_window_size()
elif (
    self.model_config.is_hybrid_swa
    and self.model_config.sliding_window_size is not None
):
    self.sliding_window_size = self.model_config.sliding_window_size
elif self.model_config.attention_chunk_size is not None:
    self.sliding_window_size = self.model_config.attention_chunk_size
```

##### `SWAKVPool`（`mem_cache/swa_memory_pool.py`）

| 字段 | 代码来源 |
|------|---------|
| `full_kv_pool` | `swa_memory_pool.py:79-84`：`token_to_kv_pool_class(size=size, ...)`。代码中无 inline comment。从构造参数推断：full attention 层的 KV cache 池（`size` = `full_max_total_num_tokens`） |
| `swa_kv_pool` | `swa_memory_pool.py:71-76`：`token_to_kv_pool_class(size=size_swa, ...)`。代码中无 inline comment。从构造参数推断：SWA 层的 KV cache 池（`size_swa` = `swa_max_total_num_tokens`，容量小于 full pool） |
| `layers_mapping` | `swa_memory_pool.py:86-91`：`Dict[int, Tuple[int, bool]]`。inline comment 原文：`# {layer_id: (index, is_swa_layer)}` |
| `full_to_swa_index_mapping` | `swa_memory_pool.py:92`：`Optional[torch.Tensor] = None`。代码中无 inline comment。从赋值链路推断：在 `SWATokenToKVPoolAllocator` 中初始化并维护 full pool slot → swa pool slot 的映射 |

```python
# swa_memory_pool.py:71-92
self.swa_kv_pool = token_to_kv_pool_class(
    size=size_swa,
    dtype=dtype,
    layer_num=self.swa_layer_nums,
    **kwargs,
)
kwargs.pop("swa_head_num", None)
kwargs.pop("swa_head_dim", None)
kwargs.pop("swa_v_head_dim", None)
self.full_kv_pool = token_to_kv_pool_class(
    size=size,
    dtype=dtype,
    layer_num=self.full_layer_nums,
    **kwargs,
)
# {layer_id: (index, is_swa_layer)}
self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
for full_attn_layer_id, global_layer_id in enumerate(full_attention_layer_ids):
    self.layers_mapping[global_layer_id] = (full_attn_layer_id, False)
for swa_layer_id, global_layer_id in enumerate(swa_attention_layer_ids):
    self.layers_mapping[global_layer_id] = (swa_layer_id, True)
self.full_to_swa_index_mapping: Optional[torch.Tensor] = None
```

**双池映射详解：为什么需要 `full_to_swa_index_mapping`？**

Gemma 4 有两种注意力层（SWA 层和 Full 层），使用两个独立的 KV cache 池。但调度器只维护一个统一的 token 索引空间（`req_to_token`），它只知道 full pool 的 slot 编号。当 SWA 层需要写入或读取 KV cache 时，需要知道对应的 swa pool slot。

```
假设请求正在生成第 5 个 token（seq_len=5）

调度器分配了 full pool 的 slot 100-104 来存储这 5 个 token 的 KV：
  req_to_token[req_idx] = [100, 101, 102, 103, 104]   ← full pool slots

同时，SWA pool 也为这 5 个 token 分配了 slot 0-4：
  swa pool slots = [0, 1, 2, 3, 4]

full_to_swa_index_mapping 就是这个翻译表：
  mapping[100] = 0    ← full slot 100 对应 swa slot 0
  mapping[101] = 1    ← full slot 101 对应 swa slot 1
  mapping[102] = 2
  mapping[103] = 3
  mapping[104] = 4
```

Full 层不需要这个映射——它直接使用 `req_to_token` 中的 full pool slot。只有 SWA 层需要翻译。代码中的翻译发生在 SWA 层 attention 计算前（`flashinfer_backend.py:1102-1204`）：

```python
if use_sliding_window_kv_pool:
    kv_indices[:kv_last_index] = (
        self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            kv_indices[:kv_last_index]
        )
    )
```

类似的其他映射参数：

| 参数 | 存在原因 | 例子 |
|------|---------|------|
| `out_cache_loc` | 调度器为本次 forward 预分配的 full pool 输出位置，新 token 的 KV 写到这里 | `[100, 101, 102]` = full pool 中 3 个空 slot |
| `out_cache_loc_swa` | 同上的 SWA pool 版本。通过 `translate_loc_from_full_to_swa(out_cache_loc)` 得到 | `[0, 1, 2]` = swa pool 中对应的 slot |
| `layers_mapping` | 全局 layer ID → 池内 layer 索引 + 是否 SWA 层。两个池的层数不同（full pool 有 N_full 层，swa pool 有 N_swa 层），需要知道每个 layer 对应哪个池的第几层 | `{0: (0, True), ..., 5: (0, False)}` = layer 0-4 是 SWA 层（映射到 swa pool 第 0-4 层），layer 5 是 Full 层（映射到 full pool 第 0 层） |

一句话总结：所有这些映射存在的根本原因是**双池架构**——SWA 层和 Full 层各有自己的 KV cache 池，但调度器和 forward 流程只有一个统一的 token 索引空间，需要映射来连接两边。

| 字段 | 代码来源 |
|------|---------|
| `full_attn_allocator` | `swa_memory_pool.py:260-267`（`page_size==1`）或 `:278-284`（`page_size>1`）：分配器绑定 `kvcache.full_kv_pool`。代码中无 inline comment。从构造参数推断：管理 full pool 的 slot 分配/释放 |
| `swa_attn_allocator` | `swa_memory_pool.py:268-275`（`page_size==1`）或 `:285-291`（`page_size>1`）：分配器绑定 `kvcache.swa_kv_pool`。代码中无 inline comment。从构造参数推断：管理 SWA pool 的 slot 分配/释放 |
| `full_to_swa_index_mapping` | `swa_memory_pool.py:294-309`。inline comment 原文：*"Note: append one more item of value -1 in the end so -1 maps to -1. It is needed for the last_loc in alloc_extend, where the first full_last_loc is -1, and we need to map it to swa_last_loc -1 as well."* |

```python
# swa_memory_pool.py:260-309 (简化)
class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, size, size_swa, page_size, ...):
        # ... page_size == 1 和 > 1 两个分支 ...
        self.full_attn_allocator = PagedTokenToKVPoolAllocator(
            size, page_size, dtype, device, kvcache.full_kv_pool, need_sort,
        )
        self.swa_attn_allocator = PagedTokenToKVPoolAllocator(
            size_swa, page_size, dtype, device, kvcache.swa_kv_pool, need_sort,
        )
        # Note: append one more item of value -1 in the end so -1 maps to -1.
        # It is needed for the last_loc in alloc_extend, where the first full_last_loc
        # is -1, and we need to map it to swa_last_loc -1 as well.
        self.full_to_swa_index_mapping = torch.cat(
            [
                torch.zeros(size + self.page_size, dtype=torch.int64, device=device),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )
```

##### `SWARadixCache`（`mem_cache/swa_radix_cache.py`）

| 字段 | 代码来源 |
|------|---------|
| `sliding_window_size` | `swa_radix_cache.py`：`params.sliding_window_size`，用于 `_evict_swa()` 中计算淘汰边界 |
| `full_lru_list` | `swa_radix_cache.py`：`LRUList(is_swa_list=False)`。`LRUList.__init__`（`:116-131`）中 `is_swa_list=False` 时使用 `prev/next/full_lock_ref` 属性。代码中无 inline comment |
| `swa_lru_list` | `swa_radix_cache.py`：`LRUList(is_swa_list=True)`。`is_swa_list=True` 时使用 `swa_prev/swa_next/swa_lock_ref` 属性。代码中无 inline comment |
| `swa_tombstone` | `TreeNode` 类中 inline comment 原文（`swa_radix_cache.py:66`）：*"swa_tombstone is used to indicate the kv indices have been freed for swa layers"*。`SWARadixCache` 中初始化为空 dict |

```python
# swa_radix_cache.py TreeNode 类 (行 66-90)
# swa_tombstone is used to indicate the kv indices have been freed for swa layers
self.swa_tombstone = False
# invariant: for any node, if swa_lock_ref is locked, full_lock_ref must be locked;
# if full_lock_ref is locked, swa_lock_ref doesn't need to be locked. So,
# full_lock_ref is always >= swa_lock_ref.
self.full_lock_ref = 0
self.swa_lock_ref = 0

# swa_radix_cache.py LRUList 类 (行 116-131)
def __init__(self, is_swa_list: bool = False):
    self.is_swa_list = is_swa_list
    if self.is_swa_list:
        self.prv = "swa_prev"
        self.nxt = "swa_next"
        self.lock_ref = "swa_lock_ref"
    else:
        self.prv = "prev"
        self.nxt = "next"
        self.lock_ref = "full_lock_ref"
```

##### `ForwardBatch`（`model_executor/forward_batch_info.py`）

| 字段 | 代码来源 |
|------|---------|
| `out_cache_loc_swa` | `forward_batch_info.py:302` 定义，inline comment 原文：*"The indices of output tokens in the token_to_kv_pool_swa"*。赋值（`:595-600`）通过 `translate_loc_from_full_to_swa()` 将 `out_cache_loc`（full pool 位置）翻译为 SWA pool 位置 |

```python
# forward_batch_info.py:302 — 字段定义
# The indices of output tokens in the token_to_kv_pool_swa
out_cache_loc_swa: Optional[torch.Tensor] = None

# forward_batch_info.py:595-600 — 赋值
if model_runner.is_hybrid_swa and ret.out_cache_loc is not None:
    ret.out_cache_loc_swa = (
        model_runner.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            ret.out_cache_loc
        )
    )
```

##### `Req`（`managers/schedule_batch.py`）

| 字段 | 代码来源 |
|------|---------|
| `swa_evicted_seqlen` | `schedule_batch.py:746`：初始值 `0`。`_evict_swa()` 中更新（`:2691`）：`req.swa_evicted_seqlen = new_swa_evicted_seqlen`。代码中无 inline comment。从赋值链路推断：`[0, swa_evicted_seqlen)` 范围的 SWA KV slot 已被释放 |
| `swa_uuid_for_lock` | `schedule_batch.py:744`。inline comment 原文：*"The node to lock until for swa radix tree lock ref"* |
| `cache_protected_len` | `schedule_batch.py:746`。inline comment 原文：*"The prefix length that is inserted into the tree cache"*。`radix_cache.py:580-584` 的补充 inline comment：*"cache_protected_len is not always equal to len(req.prefix_indices) since for page_size > 1, the partial part is added to req.prefix_indices, but that part of kv indices is not added to the tree"*。详见下方"cache_protected_len 领地边界"说明 |

```python
# schedule_batch.py:744-746 — Req 类字段定义
# The node to lock until for swa radix tree lock ref
self.swa_uuid_for_lock: Optional[int] = None
# The prefix length that is inserted into the tree cache
self.cache_protected_len: int = 0
self.swa_evicted_seqlen = 0

# schedule_batch.py:1043-1046 — cache_protected_len 赋值
if match_result.cache_protected_len is not None:
    self.cache_protected_len = match_result.cache_protected_len
else:
    self.cache_protected_len = len(self.prefix_indices)

# radix_cache.py:580-584 — 为什么不直接用 len(prefix_indices)
# The cache_protected_len is not always equal to len(req.prefix_indices)
# since for page_size > 1, the partial part is added to req.prefix_indices,
# but that part of kv indices is not added to the tree.
# It should be freed in the next cache_unfinished_req and final
# cache_finished_req to avoid memory leak.
req.cache_protected_len = len(new_indices)
```

**`cache_protected_len` 领地边界详解**

关键前提：**radix tree 节点存的是 full pool 的 KV 索引，不是 SWA pool 的。**（`swa_radix_cache.py` 中 `swa_tombstone` inline comment：*"swa_tombstone is used to indicate the kv indices have been freed for swa layers"*）两个池有各自独立的生命周期。

```
=== 为什么 cache_protected_len ≠ len(prefix_indices)？ ===

page_size=16，请求完成，序列长度 = 100 token

radix tree 按 page 对齐存储：
  100 // 16 * 16 = 96 → 前 96 个 token 能整页插入 radix tree
  剩余 4 个 token (96-99) 无法构成完整 page，不插入树

  len(prefix_indices) = 100  ← 全部 100 个 token 都有 prefix 索引
  cache_protected_len  = 96   ← 只有 96 个 token 的 KV 在 radix tree 中

  多出的 4 个 token (96-99) 的 KV 不在树中：
  → cache_unfinished_req / cache_finished_req 负责释放它们
  → 如果 cache_protected_len 错误地等于 100，这 4 个 token 的 KV 就泄漏了

=== 在 _evict_swa 中的作用：领地边界 ===

req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, req.cache_protected_len)

这不是"保护 KV 不被释放"，而是领地声明：

  [0, cache_protected_len)  → radix tree 的领地
    - full KV 在 radix tree 节点中（用于多轮对话前缀匹配）
    - SWA KV 由 radix tree 自己通过 tombstoning 管理
    - _evict_swa 不碰这个区域

  [cache_protected_len, seq_len) → _evict_swa 的领地
    - 这些 token 不在 radix tree 中
    - 它们的 SWA KV 由 _evict_swa 负责释放

  把 swa_evicted_seqlen 推到 cache_protected_len，
  后续 _evict_swa 只释放 [cache_protected_len, seq_len) 区域的 SWA KV。
  如果 _evict_swa 也去释放 radix tree 领地内的 SWA KV，
  会与 radix tree 的 tombstoning 机制冲突（double-free 或状态不一致）。

=== 多轮对话完整流程 ===

第一轮："什么是 SWA？" (100 token)
  → cache_finished_req 将前 96 token (page 对齐) 插入 radix tree
  → cache_protected_len = 96
  → 释放剩余 4 token 的 KV（partial part）

第二轮："详细解释一下" (复用第一轮缓存)
  → radix tree match_prefix 命中 96 token 前缀
  → _evict_swa 中：swa_evicted_seqlen = max(0, 96) = 96
  → _evict_swa 只管理 [96, seq_len) 的 SWA KV
  → [0, 96) 由 radix tree 管理，full KV 可被前缀匹配复用
```

#### 初始化序列（含代码引用）

```
步骤 1: ModelConfig 解析配置
  文件: configs/model_config.py

  -> __init__() 读取 HF config
     -> _get_sliding_window_size()  [第 989-994 行]
        从 hf_text_config 读取 sliding_window_size / sliding_window / window_size

     -> is_hybrid_swa = is_hybrid_swa_model(architectures)  [第 431-434 行]
        检查架构是否在已知 SWA 列表中（Llama4, GptOss, MiMoV2, Step3.5, Gemma4）

     -> get_hybrid_layer_ids()  [第 437-441 行, 实现 第 1582-1600 行]
        根据架构特定的模式计算 swa_attention_layer_ids 和 full_attention_layer_ids
        例如 Llama4: (i+1)%4!=0 为 SWA, (i+1)%4==0 为 Full

     -> has_attention_sinks = _detect_attention_sinks()  [第 444 行]
        检查模型是否使用 attention sink 偏置参数

步骤 2: ModelRunner 存储配置
  文件: model_executor/model_runner.py

  -> __init__()
     -> self.is_hybrid_swa = model_config.is_hybrid_swa

  -> load_weights() 后
     -> self.sliding_window_size 解析  [第 1495-1508 行]
        优先级: model.get_attention_sliding_window_size() > config > attention_chunk_size

步骤 3: 内存池创建
  文件: model_executor/model_runner_kv_cache_mixin.py

  -> HybridSWAPoolConfigurator 计算 full/swa pool 大小  [第 184-233 行]
     根据 swa_full_tokens_ratio 分配可用内存:
     cell_size = full_per_token * full_layers + ratio * swa_per_token * swa_layers

  -> SWAKVPool 创建  [第 359-374 行]
     包含两个子池 swa_kv_pool + full_kv_pool, 以及 layers_mapping 路由字典

  -> SWATokenToKVPoolAllocator 创建  [第 590-598 行]
     包含两个子分配器 + full_to_swa_index_mapping 映射张量

步骤 4: 调度器创建基数树
  文件: managers/scheduler.py

  -> Scheduler.__init__()
     -> 创建 SWARadixCache 或 SWAChunkCache  [第 868-925 行]
        SWARadixCache: 双 LRU 链表 + 墓碑化机制
        SWAChunkCache: 分块缓存，适用于 chunked prefill

步骤 5: Attention Backend 初始化
  文件: layers/attention/flashinfer_backend.py

  -> FlashInferAttnBackend.__init__()  [第 153-161 行]
     if model_runner.sliding_window_size is not None:
         self.num_wrappers = 2   # 创建两个 wrapper
         self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
     -> wrapper 0: 处理 SWA 层（有窗口限制）
     -> wrapper 1: 处理 Full 层（无窗口限制）
```

### 2.3 运行时调用链

#### Decode 路径

```
步骤 1: SWA KV 淘汰
  文件: managers/schedule_batch.py

  -> ScheduleBatch.maybe_evict_swa()  [第 2623-2659 行]
     遍历所有请求，按 eviction_interval 周期触发淘汰

  -> _evict_swa(req, pre_len)  [第 2661-2689 行]
     -> 计算 new_swa_evicted_seqlen = max(old, pre_len - window_size - page_size)
     -> 通过 free_swa() 释放 [old, new) 范围的 SWA KV slots
```

```python
# schedule_batch.py 第 2661-2689 行
def _evict_swa(self, req: Req, pre_len: int):
    sliding_window_size = self.tree_cache.sliding_window_size
    # 领地边界：把淘汰指针推到 radix tree 的领地边界，后续只释放 radix tree 之外的 SWA KV
    req.swa_evicted_seqlen = max(req.swa_evicted_seqlen, req.cache_protected_len)
    # 计算新的淘汰边界
    new_swa_evicted_seqlen = max(
        req.swa_evicted_seqlen,
        pre_len - sliding_window_size - self.tree_cache.page_size,
    )
    if new_swa_evicted_seqlen > req.swa_evicted_seqlen:
        free_slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, req.swa_evicted_seqlen : new_swa_evicted_seqlen
        ]
        # 释放 SWA pool slots...
```

```
步骤 2: 预计算 SWA 输出位置
  文件: model_executor/forward_batch_info.py
```

```python
# forward_batch_info.py 第 595-600 行
if model_runner.is_hybrid_swa and ret.out_cache_loc is not None:
    ret.out_cache_loc_swa = (
        model_runner.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            ret.out_cache_loc
        )
    )
```

```
步骤 3: FlashInfer 元数据初始化（核心 SWA 逻辑）
  文件: layers/attention/flashinfer_backend.py

  -> FlashInferAttnBackend.init_forward_metadata()  [第 436-448 行]
     dispatch 到 FlashInferIndicesUpdaterDecode.update()
       -> 调用 update_sliding_window()  [第 1015-1063 行]
```

```python
# flashinfer_backend.py 第 1015-1063 行
def update_sliding_window(self, ...):
    for wrapper_id in range(2):
        if wrapper_id == 0:
            # === SWA 层 ===
            # 限制有效 KV 长度为窗口大小 + 1（含当前 token）
            paged_kernel_lens_tmp = torch.clamp(
                seq_lens, max=self.sliding_window_size + 1
            )
            # 计算窗口起始位置：seq_lens - paged_kernel_lens
            # 例: seq_len=10000, window=4096 → start=5903
            kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
        else:
            # === Full 层 ===
            # 使用全部 KV，不限制
            paged_kernel_lens_tmp = seq_lens
            kv_start_idx_tmp = None

        # 仅 SWA wrapper + SWA allocator 时翻译索引
        use_sliding_window_kv_pool = wrapper_id == 0 and isinstance(
            self.token_to_kv_pool_allocator, SWATokenToKVPoolAllocator
        )
        # 将 full pool KV 索引翻译为 swa pool 索引
        self.call_begin_forward(..., use_sliding_window_kv_pool=use_sliding_window_kv_pool)
```

```python
# flashinfer_backend.py 第 1102-1204 行（call_begin_forward 中索引翻译）
if use_sliding_window_kv_pool:
    kv_last_index = kv_indptr[-1]
    kv_indices[:kv_last_index] = (
        self.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            kv_indices[:kv_last_index]
        )
    )
```

```
步骤 4: 每层注意力计算
  文件: layers/attention/flashinfer_backend.py
```

```python
# flashinfer_backend.py 第 894-930 行
def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True):
    # 根据 layer 类型选择 wrapper 0 (SWA) 或 1 (Full)
    decode_wrapper = self.forward_metadata.decode_wrappers[
        self._get_wrapper_idx(layer)
    ]
    # 写入 KV cache（SWAKVPool 路由到正确的池）
    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, ...)
    # 执行注意力计算
    o = decode_wrapper.forward(q, kv_buffer, sm_scale=layer.scaling, ...)
```

```python
# flashinfer_backend.py 第 932-941 行（wrapper 选择逻辑）
def _get_wrapper_idx(self, layer: RadixAttention):
    if self.num_wrappers == 1:
        return 0
    if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
        return layer.sliding_window_size == -1  # SWA 层返回 0, Full 层返回 1
```

#### Prefill 路径

```
步骤 1: SWA KV 淘汰 -- 同 decode

步骤 2: 预计算 SWA 输出位置 -- 同 decode

步骤 3: FlashInfer 元数据初始化（Prefill SWA 逻辑）
  文件: layers/attention/flashinfer_backend.py

  -> FlashInferAttnBackend.init_forward_metadata()  [第 506-525 行]
     dispatch 到 FlashInferIndicesUpdaterPrefill.update()
       -> 调用 update_sliding_window()  [第 1297-1345 行]
```

```python
# flashinfer_backend.py 第 1297-1345 行
def update_sliding_window(self, ..., prefix_lens, prefill_wrappers, ...):
    for wrapper_id in range(2):
        if wrapper_id == 0:
            # === SWA 层 ===
            # 窗口 + 新 token 数量，取序列长度的较小值
            # 确保窗口覆盖新 token 可见的全部历史
            paged_kernel_lens = torch.minimum(
                seq_lens,
                torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
            )
        else:
            # === Full 层 ===
            paged_kernel_lens = seq_lens

        kv_start_idx = seq_lens - paged_kernel_lens
        self.call_begin_forward(..., use_sliding_window_kv_pool=...)
```

```
步骤 4: 每层注意力计算
  文件: layers/attention/flashinfer_backend.py
```

```python
# flashinfer_backend.py 第 780-891 行
def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
    # 选择 wrapper
    prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
        self._get_wrapper_idx(layer)
    ]
    # 写入 KV cache
    if save_kv_cache:
        forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, ...)
    # FlashInfer 内核通过 window_left 参数强制窗口边界
    o = prefill_wrapper_paged.forward(
        q, kv_buffer,
        window_left=layer.sliding_window_size,  # 窗口左边界
        sm_scale=layer.scaling, ...
    )
```

### 2.4 KV 缓存管理

#### 双池架构

- **`full_kv_pool`**：存储所有请求的完整序列长度 KV 缓存。由全注意力层使用。
- **`swa_kv_pool`**：仅存储滑动窗口内的 KV 缓存。由 SWA 层使用。

#### 索引映射

`full_to_swa_index_mapping` 张量维护 full 池索引到 SWA 池索引的映射。在每次 `alloc()`、`alloc_extend()` 和 `alloc_decode()` 期间更新。

#### KV 缓存淘汰策略

**a) 请求级别 SWA 淘汰** (`_evict_swa()`, schedule_batch.py:2661-2691):
- 触发周期：decode 由 `eviction_interval` 控制（默认 `sliding_window_size * 0.5`），chunked prefill 在每个 chunk 边界触发

**b) 全局基数树淘汰** (`SWARadixCache.evict()`, swa_radix_cache.py:550-638):
- 首先淘汰 full LRU 叶子节点（释放 full + SWA tokens）
- 如果 SWA 需要更多空间，内部节点在 SWA 中被"墓碑化"（SWA KV 释放，full KV 保留）

### 2.5 关键文件（GPU 端）

| 文件 | 角色 | 重要性 |
|------|------|--------|
| `python/sglang/srt/layers/radix_attention.py` | 注意力层入口，sinks 通过 kwargs 传递 | 核心 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | 双 wrapper 调度，window_left，SWA 池索引转换 | 核心 |
| `python/sglang/srt/mem_cache/swa_memory_pool.py` | 双 KV 池 + 索引映射 + 分配器 | 核心 |
| `python/sglang/srt/mem_cache/swa_radix_cache.py` | SWA 基数树 + 墓碑化 + 双 LRU | 核心 |
| `python/sglang/srt/configs/model_config.py` | is_hybrid_swa, sliding_window_size, 层 ID 解析 | 高 |
| `python/sglang/srt/managers/schedule_batch.py` | maybe_evict_swa(), Req SWA 字段 | 高 |
| `python/sglang/srt/model_executor/forward_batch_info.py` | ForwardBatch.out_cache_loc_swa | 高 |

---

## 3. NPU 端 SWA 架构

### 3.1 NPU Attention 实现位置

| 文件 | 角色 |
|------|------|
| `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` | 核心后端，等同 GPU 端的 flashinfer_backend.py |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py` | 基于 SDPA 的 fallback 实现 |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py` | NPU 版 GDN linear attention |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_hybrid_linear_attn_backend.py` | NPU 版混合 linear attention |
| `python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py` | NPU MLA 预处理 |

NPU 后端通过 `@register_attention_backend("ascend")` 注册，在 `attention_registry.py` 中选择。

### 3.2 NPU SWA 适配现状

#### 已有 SWA 支持

**a) Hybrid SWA Block Tables**：`ascend_backend.py` 构造函数中初始化 `is_hybrid_swa` 状态和 `full_to_swa_index_mapping`。在 `init_forward_metadata` 中构建 SWA 专用 block tables。

**b) Sinks Attention NPU 实现**：从 `sgl_kernel_npu.attention.sinks_attention` 导入 `attention_sinks_prefill_triton` 和 `attention_sinks_triton`，在三个调用点使用：
- Prefill/extend 路径
- Decode graph 路径
- Decode 非 graph 路径

**c) FIA 路径**：NPU 有两条主要路径：
- `use_fia=True`：使用 `torch.ops.npu.npu_fused_infer_attention_score`
- `use_fia=False`：使用 `torch_npu._npu_flash_attention_qlens`（extend）和 `torch_npu._npu_paged_attention`（decode）

### 3.3 NPU 与 GPU 的关键差异

1. **内核来源不同**：GPU 用 Triton 内核；NPU 用 `sgl_kernel_npu` 专用实现
2. **FIA 是可选路径**：NPU 后端通过环境变量 `ASCEND_USE_FIA=1` 控制
3. **无 Triton 路径**：GPU 的 TritonAttnBackend 在 NPU 不可用
4. **MLA 路径独立**：NPU 使用 `torch_npu._npu_paged_attention_mla` 和 `torch_npu.atb.npu_ring_mla`
5. **Graph Runner 不同**：NPU 使用 `NpuGraphRunner` 而非 `CudaGraphRunner`

### 3.4 NPU KV Cache

- **独立的 Memory Pool**：`NPUMHATokenToKVPool` 继承 GPU 版本，使用连续内存布局
- **独立的 Allocator**：`NPUPagedTokenToKVPoolAllocator` 覆盖 `alloc_extend()`，小批量用 NPU 专用 Triton 内核
- **SWA Cache 设备无关**：`swa_memory_pool.py` 和 `swa_radix_cache.py` 包含 NPU 分支但逻辑共享

### 3.5 NPU Attention 调用链

```
Model.forward()
  -> DecoderLayer.forward()
    -> Attention.forward()
      -> RadixAttention.__call__()
        -> AscendAttnBackend.forward_decode() 或 forward_extend()

Decode 路径分支:
  forward_decode()
    -> graph_mode? -> forward_decode_graph()
    -> use_mla? -> MLA decode path
    -> sinks? -> attention_sinks_triton()
    -> use_fia? -> npu_fused_infer_attention_score()
    -> else -> _npu_paged_attention()

Extend 路径分支:
  forward_extend()
    -> dllm? -> forward_dllm()
    -> nsa? -> forward_sparse()
    -> mtp? -> forward_mtp()
    -> use_mla? -> MLA extend path
    -> sinks? -> attention_sinks_prefill_triton()
    -> use_fia? -> npu_fused_infer_attention_score (per-bs loop)
    -> else -> _npu_flash_attention_qlens() 或 SDPA fallback
```

### 3.6 关键文件（NPU 端）

| 文件 | 角色 | 重要性 |
|------|------|--------|
| `ascend_backend.py` | NPU 主 attention backend，SWA/sinks 完整实现 | 核心 |
| `ascend_torch_native_backend.py` | NPU SDPA fallback | 高 |
| `memory_pool_npu.py` | NPU KV cache pool | 高 |
| `allocator_npu.py` | NPU paged allocator | 中 |
| `attention_registry.py` | Backend 注册和选择 | 关键 |

---

## 4. 同事 POC 代码分析

### 4.1 Commit 列表

| SHA | 日期 | 消息 | 修改文件 |
|-----|------|------|----------|
| `a5541b633d21` | 2026-04-10 | adapt for gemma | `memory_pool_npu.py` (+8) |
| `ff6b92226e84` | 2026-04-17 | fix graph problem | `gemma4_mm.py` (+10/-3) |
| `bc1be097c040` | 2026-04-20 | add sliding window attention | `ascend_backend.py` (+200/-24), `ascend_torch_native_backend.py` (+43/-15), `gemma4_causal.py` (+3/-1) |
| `336c8125b557` | 2026-04-10 | add profiling | `environ.py` (+5), `scheduler.py` (+54) |

### 4.2 各文件详细分析

#### 4.2.1 `gemma4_causal.py` -- 滑动窗口尺寸转换

**修改**：`sliding_window` 从 `config.sliding_window` 改为 `get_attention_sliding_window_size(config)`（即 `config.sliding_window - 1`）；非 SWA 层从 `None` 改为 `-1`。

**分析**：HF 使用 inclusive 窗口大小，sglang 假设 exclusive，因此减 1。这在 GPU 主线中由 `get_attention_sliding_window_size` 函数处理，但只在 model 级别的 `get_attention_sliding_window_size()` 方法中使用。POC 的修改方向正确，但引入了与主线不一致的行为差异——sink triton 内核接收的窗口大小可能不同。

#### 4.2.2 `ascend_torch_native_backend.py` -- SDPA SWA 窗口裁剪

**修改**：在 `run_sdpa_forward_extend` 和 `run_sdpa_forward_decode` 中新增 `sliding_window_size` 参数，对 KV 范围进行窗口裁剪。

**Extend 路径**：
```python
atten_start_kv = max(prefill_seq_len_q - sliding_window_size, atten_start_kv)
```
窗口锚定在 extend chunk 的第一个 query token，而非最终序列长度。

**Decode 路径**：
```python
atten_start_kv = max(atten_end_kv - (sliding_window_size + 1), atten_start_kv)
```
窗口大小为 `sliding_window_size + 1`（包括当前 token）。

**重要修改**：`per_req_query_redudant` 从 `torch.empty` 改为 `torch.zeros`，避免 SWA 裁剪后未填充区域的未定义行为。

#### 4.2.3 `ascend_backend.py` -- 主要 SWA 分页注意力逻辑

**新增方法**（约 150 行）：
- `_is_swa_layer()`：判断当前层是否为 SWA 层
- `_get_swa_page_aligned_starts()`：计算 SWA 窗口的页对齐起始位置
- `_get_swa_paged_seq_lens_cpu_int/list()`：计算窗口内有效序列长度
- `_build_swa_paged_block_tables()`：动态构建 SWA 块表
- `_get_paged_attention_inputs()`：统一接口，对 SWA 层返回裁剪后的块表

**四条路径的 SWA 处理**：

| 路径 | SWA 处理方式 |
|------|-------------|
| Sinks 路径（prefill/decode 有 sinks） | 使用预构建 `block_tables_swa`，走 `attention_sinks_triton` |
| FIA 路径 | 使用动态构建的 `paged_block_tables` 和裁剪后的 `paged_seq_lens` |
| SDPA 路径 | 动态构建块表 + 内部窗口裁剪 |
| Decode graph 路径 | 使用预构建 `block_tables_swa`（静态） |

#### 4.2.4 `gemma4_mm.py` -- NPU graph 兼容性修复

将布尔索引（`ple_ids[mask] = pad_id`）替换为 `torch.where`，解决 NPU graph 模式下 in-place boolean indexing 不兼容问题。

#### 4.2.5 `memory_pool_npu.py` -- SWA 参数透传

新增 `swa_head_num`, `swa_head_dim`, `swa_v_head_dim` 参数，透传给父类 `MHATokenToKVPool`。

---

## 5. 根因分析

### 5.1 Decode graph 静态 SWA 块表 [信心: HIGH]

**问题**：Decode graph 路径使用静态 `block_tables_swa`，窗口不随序列增长滑动。

```python
if self._is_swa_layer(layer):
    paged_block_tables = self.forward_metadata.block_tables_swa
    paged_seq_lens_cpu_int = self.forward_metadata.seq_lens_cpu_int
```

graph 捕获时 seq_len fill 值为 0，无法动态构建块表。但 `block_tables_swa` 在 metadata init 时根据 `full_to_swa_index_mapping` 构建后不再变化。

**影响**：decode 时 SWA 窗口无法正确滑动，一直看到同样的 KV pages。随着 decode 步数增加，越来越多的"过期" KV 被错误纳入注意力计算。

**这很可能是长跑精度退化的最大嫌疑。**

### 5.2 `full_to_swa_index_mapping` 映射陈旧 [信心: HIGH]

`_build_swa_paged_block_tables` 使用 `full_to_swa_index_mapping` 映射：

```python
block_tables = (
    self.full_to_swa_index_mapping[
        req_tokens.gather(1, gather_positions).long()
    ]
    // self.page_size
).to(torch.int32)
```

这个映射在 token 分配时建立，不会随序列位置变化。动态构建块表时，如果映射的某些位置没有被分配（值可能是 0 或无效值），会导致 gather 到错误的 page index。长跑中 token 分配/释放的累积可能产生越来越多的无效映射。

### 5.3 Extend 路径 anchor_lens 语义错误 [信心: HIGH]

在 `_get_paged_attention_inputs` 中：

```python
anchor_lens=forward_batch.extend_prefix_lens
```

SWA 窗口应基于最终序列长度计算可见范围，而不是 prefix 长度。当 extend chunk 较大（长 prompt prefill）时，SWA 窗口应锚定在当前 query 的最后一个 token，而非 prefix 末尾。

### 5.4 SDPA 与 FIA 路径窗口处理不一致 [信心: MEDIUM]

- SDPA extend 路径：`atten_start_kv = max(prefill_seq_len_q - sliding_window_size, atten_start_kv)`
- FIA extend 路径：`starts = clamp(anchor_lens - sliding_window_size, min=0)`

两者锚点不同，FIA 的 `context_lens` 被设为裁剪后的值，可能与 FIA 内核期望的语义不匹配。

### 5.5 Sink token 与 paged path 不一致 [信心: MEDIUM]

当 `sinks is not None` 时走 sinks triton 内核 + `block_tables_swa`，当 `sinks is None` 时走 FIA/SDPA + 动态构建块表。同一个 SWA 层在不同请求间使用不同块表构建逻辑。

### 5.6 `torch.empty` 改为 `torch.zeros` 不完全 [信心: LOW]

只在 SDPA extend 路径中修改，FIA 路径和其他路径没有类似修改。FIA 内核处理裁剪后块表时遇到无效页可能读到零初始化的 KV 数据。

### 5.7 综合推理链：长跑精度退化场景

```
初始状态
  -> seq_len 较小, SWA 窗口覆盖全部 KV, 一切正常

序列增长超过滑动窗口大小
  -> Decode graph: 静态 block_tables_swa, 窗口未正确滑动
  -> 动态构建: full_to_swa_index_mapping 可能访问已释放的 token slots

多轮请求交替
  -> Token pool 分配/释放导致映射陈旧
  -> 新请求复用已释放的 token slots, SWA 块表指向旧数据

精度累积
  -> 每步 decode 中错误 KV cache 导致 attention 输出微小偏差
  -> 偏差通过残差连接和后续层放大
  -> 逐步积累到"零点几"的级别
```

**最大嫌疑排序**：
1. Decode graph 中静态 SWA 块表（窗口不滑动）
2. `full_to_swa_index_mapping` 映射陈旧（token 复用后指向错误数据）
3. Extend 路径 anchor_lens 使用 prefix_lens 而非最终 seq_len

---

## 6. GPU vs NPU 对比

| 方面 | GPU 主线 (FlashInfer) | POC (NPU Ascend) |
|------|----------------------|-------------------|
| SWA 块表构建 | 预构建 `full_to_swa_index_mapping` + FlashInfer 在 metadata 阶段完成 | 运行时动态构建（`_build_swa_paged_block_tables`），每次 forward 都计算 |
| Extend 路径 | FlashInfer fused kernel 直接处理 SWA (`window_left`) | FIA 内核用裁剪后块表；SDPA fallback 用手动 KV 范围裁剪 |
| Decode 路径 | 预构建 SWA 块表 + FlashInfer kernel | Graph 模式用预构建块表；eager 模式动态构建 |
| Sink token | `attention_sinks_triton` 内核完整支持 | 同样使用该内核，接口一致 |
| 窗口大小 | `config.sliding_window`（HF inclusive）直接传入 | `config.sliding_window - 1`（exclusive 转换） |
| SWA KV 缓冲区 | `SWAKVPool` 独立分配 | 依赖父类 `MHATokenToKVPool` 的 SWA 分配机制 |
| KV 淘汰 | `maybe_evict_swa()` + SWA radix tree 墓碑化 | 依赖 GPU 的共享实现（设备无关） |
| 双 wrapper 调度 | FlashInfer 2 个 wrapper（SWA=0, Full=1） | 单 backend 内 if/else 分支 |

---

## 7. 实现建议

### 7.1 修复优先级

| 优先级 | 问题 | 建议修复方案 |
|--------|------|-------------|
| P0 | Decode graph 静态 SWA 块表 | 在 graph replay 时动态更新 `block_tables_swa`，或使用与 GPU 端类似的预构建 + offset 调整机制 |
| P0 | `full_to_swa_index_mapping` 陈旧 | 验证映射更新时机，确保在 token 分配/释放时同步更新 |
| P1 | Extend anchor_lens 使用 prefix_lens | 改用 `seq_lens - 1` 或 `extend_prefix_lens + extend_seq_lens - 1` 作为锚点 |
| P1 | 窗口大小转换不一致 | 统一使用 GPU 主线的 `get_attention_sliding_window_size` 函数 |
| P2 | SDPA vs FIA 路径不一致 | 统一两路径的窗口裁剪逻辑，确保锚点和范围计算一致 |
| P2 | `torch.empty` 改 `torch.zeros` 不完全 | 所有涉及裁剪后 tensor 的路径都使用 `zeros` 初始化 |

### 7.2 架构建议

1. **参考 GPU 端的 FlashInfer 双 wrapper 模式**：NPU 端的 if/else 分支散落在各路径中，建议抽象为统一的 SWA/Full 调度层
2. **SWA 块表构建时机**：GPU 端在 `init_forward_metadata` 阶段完成所有 SWA 块表准备，NPU 应保持一致
3. **Graph 模式 SWA 处理**：需要设计一种机制，在 graph replay 时能够更新 SWA 窗口的可见范围。可能的方案：
   - 在 graph capture 时使用最大可能的 block table 大小，replay 时用 mask 控制
   - 使用间接引用更新 block table 内容
4. **与主线保持一致**：窗口大小转换应复用 GPU 主线的 `get_attention_sliding_window_size` 函数，避免 NPU 独立维护

### 7.3 测试建议

1. **单请求短序列**：seq_len < sliding_window_size，验证窗口覆盖全部 KV
2. **单请求长序列**：seq_len > sliding_window_size，验证窗口正确滑动
3. **多请求交替**：验证 token pool 复用后 SWA 映射正确性
4. **长跑精度**：连续 decode 1000+ 步，对比 GPU 端输出
5. **Sink token 测试**：验证 sinks 与非 sinks 请求混合时行为一致
6. **Graph 模式测试**：捕获 graph 后长跑，验证块表正确更新

---

## 附录：关键文件完整索引

### GPU 端

| 文件路径 | 作用 |
|---------|------|
| `python/sglang/srt/layers/radix_attention.py` | 注意力层入口 |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | FlashInfer 双 wrapper + SWA |
| `python/sglang/srt/mem_cache/swa_memory_pool.py` | SWA 双 KV 池 |
| `python/sglang/srt/mem_cache/swa_radix_cache.py` | SWA 基数树 + 墓碑化 |
| `python/sglang/srt/mem_cache/unified_cache_components/swa_component.py` | 统一组件版 SWA |
| `python/sglang/srt/configs/model_config.py` | 模型配置 + SWA 检测 |
| `python/sglang/srt/managers/schedule_batch.py` | 调度 + SWA 淘汰 |
| `python/sglang/srt/model_executor/forward_batch_info.py` | ForwardBatch |
| `python/sglang/srt/model_executor/pool_configurator.py` | 池大小计算 |
| `python/sglang/srt/model_executor/model_runner.py` | 模型运行器 |
| `python/sglang/srt/models/mimo_v2.py` | Sink 参数示例模型 |

### NPU 端

| 文件路径 | 作用 |
|---------|------|
| `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py` | NPU 主 attention backend |
| `python/sglang/srt/hardware_backend/npu/attention/ascend_torch_native_backend.py` | SDPA fallback |
| `python/sglang/srt/hardware_backend/npu/memory_pool_npu.py` | NPU KV cache pool |
| `python/sglang/srt/hardware_backend/npu/allocator_npu.py` | NPU allocator |
| `python/sglang/srt/layers/attention/attention_registry.py` | Backend 注册 |
| `python/sglang/srt/models/gpt_oss.py` | 使用 sinks SWA 的模型 |
| `python/sglang/srt/models/gemma4_causal.py` | Gemma 4 模型 |
