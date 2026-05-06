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

`paged_kernel_lens` 是传给 FlashInfer 分页注意力内核的参数，表示**每个请求在本次注意力计算中实际需要关注的 KV cache 有效长度**。

内核不会读取整个 KV cache，而是根据 `paged_kernel_lens` 和 `kv_start_idx` 只访问窗口范围内的 KV 条目：

```
KV Cache:  [token_0, token_1, ..., token_N-1]
                                   ↑ kv_start_idx
              <--- paged_kernel_lens --->
```

**Decode 路径**（`flashinfer_backend.py` 第 1036-1037 行）：
```python
# wrapper_id == 0: SWA 层
paged_kernel_lens_tmp = torch.clamp(seq_lens, max=self.sliding_window_size + 1)
kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
```

- `seq_lens` = 当前序列总长度（含刚写入的 token）
- 例：序列长度 10000，窗口 4096 → `paged_kernel_lens = min(10000, 4097) = 4097`，`kv_start_idx = 10000 - 4097 = 5903`
- 内核只访问 KV cache 的 `[5903, 10000)` 范围，而非全部 10000 个 token
- **为什么 +1**：decode 生成了 1 个新 token，窗口应包含它，所以窗口大小 = `sliding_window_size + 1`

**Prefill 路径**（`flashinfer_backend.py` 第 1320-1322 行）：
```python
# wrapper_id == 0: SWA 层
paged_kernel_lens = torch.minimum(
    seq_lens,
    torch.tensor(self.sliding_window_size) + seq_lens - prefix_lens,
)
kv_start_idx = seq_lens - paged_kernel_lens
```

- `seq_lens - prefix_lens` = 本次 extend 新增的 token 数
- `sliding_window_size + (seq_lens - prefix_lens)` = 窗口 + 新 token 数，确保窗口覆盖新 token 可见的全部历史 token
- `min(seq_lens, ...)` 确保不超过实际序列长度（序列短于窗口时不截断）

#### 关键概念：`new_swa_evicted_seqlen`

`new_swa_evicted_seqlen` 表示**SWA KV cache 中可以安全释放的 token 序列位置上限**。位置 `[0, new_swa_evicted_seqlen)` 范围内的 token 保证不在当前滑动窗口内，可以从 SWA 池中释放。

计算逻辑（`schedule_batch.py` 第 2682-2685 行）：
```python
new_swa_evicted_seqlen = max(
    req.swa_evicted_seqlen,
    pre_len - sliding_window_size - self.tree_cache.page_size,
)
```

各部分含义：
- `pre_len`：当前序列长度（本次步骤之前）
- `pre_len - sliding_window_size`：超出滑动窗口的最早位置。窗口从 `pre_len - sliding_window_size` 到 `pre_len`，此位置之前的 token 不在窗口内
- 再减 `page_size`：安全裕量。确保至少保留一个非墓碑页面供基数树在多轮对话中复用（基数树需要非墓碑节点来合并前缀）
- `max(old, ...)`：淘汰只向前推进，不会回退。避免释放已释放过的 slot

示例（窗口 4096，page_size 16，pre_len 8000）：
```
序列位置: [0 ......... 3904 ......... 8000]
                       ↑ new_swa_evicted_seqlen = 8000 - 4096 - 16 = 3888
                       （实际会页对齐到 3888 // 16 * 16 = 3888）
[0, 3888) 范围的 SWA KV cache 可安全释放
[3888, 8000) 范围的 SWA KV cache 必须保留（在窗口内）
```

#### Sink Token

Sink token 是**可学习的、每头的标量偏置**，添加到注意力 softmax 中，补偿滑动窗口导致的被淘汰 KV 缓存条目。并非每个 SWA 模型都使用——由 `ModelConfig.has_attention_sinks` 控制。

**关键**：Sink token **不**存储在 KV 缓存中。它们是 FlashInfer/TensorRT-LLM 注意力内核应用于 softmax 计算的标量值。KV 缓存完全不知道 sink。

### 2.2 初始化阶段

#### 参数/数据结构

##### `ModelConfig`（`configs/model_config.py`）

| 字段 | 含义 |
|------|------|
| `is_hybrid_swa` | 布尔值，标识当前模型是否为混合 SWA 架构。通过 `is_hybrid_swa_model()` 检查 `hf_config.architectures` 是否包含已知 SWA 架构（Llama4, GptOss, MiMoV2, Step3.5, Gemma4） |
| `sliding_window_size` | 滑动窗口大小（token 数）。从 HF config 读取，优先级：`sliding_window_size` > `sliding_window` > `window_size` |
| `swa_attention_layer_ids` | 列表，哪些层使用滑动窗口注意力。例如 Gemma4 每隔 N 层交替 SWA/Full |
| `full_attention_layer_ids` | 列表，哪些层使用全局注意力（无窗口限制） |
| `has_attention_sinks` | 布尔值，标识模型是否使用可学习的 attention sink 偏置参数 |

```python
# model_config.py 第 431-434 行
self.is_hybrid_swa = (
    is_hybrid_swa_model(self.hf_config.architectures)
    and not self.disable_hybrid_swa
)

# model_config.py 第 989-994 行
def _get_sliding_window_size(self) -> Optional[int]:
    for key in ("sliding_window_size", "sliding_window", "window_size"):
        value = getattr(self.hf_text_config, key, None)
        if value is not None:
            return value
    return None
```

##### `ModelRunner`（`model_executor/model_runner.py`）

| 字段 | 含义 |
|------|------|
| `is_hybrid_swa` | 从 ModelConfig 透传，供 attention backend 和 memory pool 判断 |
| `sliding_window_size` | 最终解析的窗口大小。优先级：模型方法 > config > attention_chunk_size |
| `full_max_total_num_tokens` | full KV pool 最大 token 容量 |
| `swa_max_total_num_tokens` | SWA KV pool 最大 token 容量 |

```python
# model_runner.py 第 1495-1508 行
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

| 字段 | 含义 |
|------|------|
| `full_kv_pool` | 全局注意力层的 KV cache 池，存储所有 token 的 KV（包括已被 SWA 窗口淘汰的），供 full attention 层使用 |
| `swa_kv_pool` | 滑动窗口层的 KV cache 池，仅存储窗口内的 token KV。比 full pool 小得多 |
| `layers_mapping` | 字典 `{layer_id: (pool内偏移, 是否SWA层)}`，将全局 layer ID 映射到对应池内的 layer 索引 |
| `full_to_swa_index_mapping` | 张量，将 full pool 中的 token 索引映射到 swa pool 中的 token 索引。在 `alloc`/`alloc_extend`/`alloc_decode` 时更新 |

```python
# swa_memory_pool.py 第 28-101 行
class SWAKVPool(KVCache):
    def __init__(self, size, size_swa, page_size, ...,
                 swa_attention_layer_ids, full_attention_layer_ids, ...):
        # SWA 层专用池（小，只存窗口内 token）
        self.swa_kv_pool = token_to_kv_pool_class(size=size_swa, ...)
        # Full 层专用池（大，存全部 token）
        self.full_kv_pool = token_to_kv_pool_class(size=size, ...)
        # 路由映射：layer_id -> (池内偏移, 是否SWA层)
        self.layers_mapping: Dict[int, Tuple[int, bool]] = {}
        for full_id, global_id in enumerate(full_attention_layer_ids):
            self.layers_mapping[global_id] = (full_id, False)  # False = full 层
        for swa_id, global_id in enumerate(swa_attention_layer_ids):
            self.layers_mapping[global_id] = (swa_id, True)   # True = SWA 层
```

##### `SWATokenToKVPoolAllocator`（`mem_cache/swa_memory_pool.py`）

| 字段 | 含义 |
|------|------|
| `full_attn_allocator` | full KV pool 的分配器，管理 full pool 的 slot 分配/释放 |
| `swa_attn_allocator` | SWA KV pool 的分配器，管理 SWA pool 的 slot 分配/释放 |
| `full_to_swa_index_mapping` | 维护 full pool slot → swa pool slot 的映射张量。每次分配 token 时同时分配两个池的 slot 并记录映射 |

```python
# swa_memory_pool.py 第 231-280 行
class SWATokenToKVPoolAllocator(BaseTokenToKVPoolAllocator):
    def __init__(self, size, size_swa, page_size, ...):
        # Full pool 分配器
        self.full_attn_allocator = PagedTokenToKVPoolAllocator(size, page_size, ...)
        # SWA pool 分配器（独立管理，容量更小）
        self.swa_attn_allocator = PagedTokenToKVPoolAllocator(size_swa, page_size, ...)
```

##### `SWARadixCache`（`mem_cache/swa_radix_cache.py`）

| 字段 | 含义 |
|------|------|
| `sliding_window_size` | 窗口大小，用于计算淘汰边界 |
| `full_lru_list` | Full KV 的 LRU 链表，用于全局淘汰（淘汰 full 叶子节点时同时释放 SWA） |
| `swa_lru_list` | SWA KV 的 LRU 链表，用于仅 SWA 淘汰（保留 full KV，释放 SWA KV） |
| `swa_tombstone` | 字典，标记 SWA KV 已被释放但 full KV 仍保留的节点（墓碑节点） |

```python
# swa_radix_cache.py 第 336-382 行
class SWARadixCache(BasePrefixCache):
    def __init__(self, params):
        self.sliding_window_size = params.sliding_window_size
        self.reset()

    def reset(self):
        # 双 LRU 链表：full 和 SWA 独立管理淘汰顺序
        self.full_lru_list = LRUList(is_swa_list=False)
        self.swa_lru_list = LRUList(is_swa_list=True)
        self.swa_tombstone = {}  # 节点 -> 是否为墓碑（SWA KV 已释放）
```

##### `ForwardBatch`（`model_executor/forward_batch_info.py`）

| 字段 | 含义 |
|------|------|
| `out_cache_loc_swa` | SWA pool 中的输出位置索引。将 `out_cache_loc`（full pool 位置）翻译为 SWA pool 位置，供 SWA 层写入新 KV |

```python
# forward_batch_info.py 第 595-600 行
if model_runner.is_hybrid_swa and ret.out_cache_loc is not None:
    ret.out_cache_loc_swa = (
        model_runner.token_to_kv_pool_allocator.translate_loc_from_full_to_swa(
            ret.out_cache_loc
        )
    )
```

##### `Req`（`managers/schedule_batch.py`）

| 字段 | 含义 |
|------|------|
| `swa_evicted_seqlen` | 当前已释放的 SWA KV 最大序列位置。`[0, swa_evicted_seqlen)` 范围的 SWA KV 已被释放 |
| `swa_uuid_for_lock` | SWA 操作的锁 UUID，用于并发安全 |
| `cache_protected_len` | 基数树保护长度。`[0, cache_protected_len)` 范围的 KV 由基数树管理释放，不能在 `_evict_swa()` 中释放 |

```python
# schedule_batch.py Req 类字段
self.swa_evicted_seqlen = 0       # 初始未淘汰任何 token
self.swa_uuid_for_lock = None     # 并发控制
self.cache_protected_len = 0      # 基数树保护边界
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
    # 确保基数树保护范围内的 KV 不被释放
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
