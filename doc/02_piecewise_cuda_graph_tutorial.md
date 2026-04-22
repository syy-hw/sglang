# Piecewise CUDA Graph (PCG) 深入教程

> 生成日期: 2026-04-21
> 基于 commit: `c25f00630`
> 参考文档: [piecewise_cuda_graph.md](../docs/advanced_features/piecewise_cuda_graph.md), [pcg_vs_bcg_analysis.md](../docs/advanced_features/pcg_vs_bcg_analysis.md)

---

## 目录

- [一、核心创新点：split_points 机制](#一核心创新点split_points-机制)
  - [1. 问题：标准 CUDA Graph 无法处理动态 token 数](#1-问题标准-cuda-graph-无法处理动态-token-数)
  - [2. PCG 的解法：将模型按层切分](#2-pcg-的解法将模型按层切分)
  - [3. 代码入口](#3-代码入口)
- [二、torch.compile 在 PCG 中的角色](#二torchcompile-在-pcg-中的角色)
  - [1. 问题：如何自动化地将模型切分为子图](#1-问题如何自动化地将模型切分为子图)
  - [2. PCG 的解法：Dynamo tracing + 自定义 SGLangBackend](#2-pcg-的解法dynamo-tracing--自定义-sglangbackend)
  - [3. 代码入口](#3-代码入口-1)
- [三、Padding 策略](#三padding-策略)
  - [1. 问题：如何用有限个 capture size 覆盖无限的 token 数](#1-问题如何用有限个-capture-size-覆盖无限的-token-数)
  - [2. PCG 的解法：递增粒度的 capture size + bisect padding](#2-pcg-的解法递增粒度的-capture-size--bisect-padding)
  - [3. 代码入口](#3-代码入口-2)
- [四、完整数据流：从输入到输出](#四完整数据流从输入到输出)
  - [1. 初始化阶段（服务器启动时）](#1-初始化阶段服务器启动时)
  - [2. 运行时 Replay（每次推理请求）](#2-运行时-replay每次推理请求)

---

## 一、核心创新点：split_points 机制

### 1. 问题：标准 CUDA Graph 无法处理动态 token 数

**场景**：在 prefill/extend 阶段，每个 batch 的总 token 数在请求间变化：

```
Batch 1: [128 + 256 + 512] = 896 tokens
Batch 2: [64 + 32] = 96 tokens
Batch 3: [1024] = 1024 tokens
```

标准 CUDA Graph 要求固定 shape。如果为每种可能的 token 数都 capture 一个完整模型的 graph，显存开销将无法承受（一个 LLM-70B 的 graph 可能占用数百 MB）。

### 2. PCG 的解法：将模型按层切分

**核心洞察**：与其为整个模型捕获 N 种 shape，不如把模型切成 M 个 piece，每个 piece 独立捕获。这样只需 M x N 个小 graph，而不是 N 个大 graph。

**切分位置的选择** — 在以下操作处插入 split point：

| split point | 注册位置 | 为什么在这里切分 |
|-------------|----------|-----------------|
| `sglang.inplace_all_reduce` | `parallel_state.py:151` | TP 通信，依赖运行时状态 |
| `sglang.unified_attention_with_output` | `radix_attention.py:139` | Attention 计算涉及动态 seq_len |
| `sglang.unified_linear_attention_with_output` | `radix_linear_attention.py:105` | 线性 Attention 同理 |
| `sglang.moe_forward_piecewise_cuda_graph_impl` | `piecewise_cuda_graph_runner.py:183` | MoE dispatch（条件性注册） |

**切分后的结构**：

```
完整模型 forward:
  embed -> [layer_0: attn + mlp -> allreduce] -> [layer_1: attn + mlp -> allreduce] -> lm_head

切分后:
  submod_0: [embed, layer_0.mlp, layer_0.norm, ...]    <-- 可 CUDA Graph capture
  submod_1: [all_reduce / attention]                     <-- eager 执行 (splitting graph)
  submod_2: [layer_1.mlp, layer_1.norm, ...]             <-- 可 CUDA Graph capture
  submod_3: [attention]                                   <-- eager 执行
  ...
```

split ops 以 eager 模式执行，其余连续计算被 CUDA Graph 捕获。

### 3. 代码入口

| 组件 | 文件 | 行号 |
|------|------|------|
| `register_split_op()` 装饰器 | `compilation/compilation_config.py` | 5-14 |
| `SPLIT_OPS` 全局列表 | `compilation/compilation_config.py` | 5 |
| 动态添加 split op | `piecewise_cuda_graph_runner.py` | 183-186 |
| `split_graph()` 分割逻辑 | `compilation/backend.py` | 220-263 |

---

## 二、torch.compile 在 PCG 中的角色

### 1. 问题：如何自动化地将模型切分为子图

手动切分模型需要修改每个模型的 forward 代码，不现实。需要一种通用的、与模型无关的切分机制。

### 2. PCG 的解法：Dynamo tracing + 自定义 SGLangBackend

PCG 利用 PyTorch 的 `torch.compile` 基础设施，但完全自定义了编译 backend：

```
model.forward()
    |
    v
PyTorch Dynamo Tracing
+-------------------------------------------+
| 将 model.forward trace 成完整的 FX Graph   |
| (所有 op 变成 FX Graph 中的 node)          |
+--------------------+----------------------+
                     |
                     v
SGLangBackend.__call__(graph, example_inputs)     backend.py:402
+-------------------------------------------+
| 1. split_graph(graph, split_ops)              backend.py:427-430
|    -> 按 split_ops 切分为 submod 序列
|
| 2. PiecewiseCompileInterpreter.run()          backend.py:446-453
|    -> 用 fake tensor 模拟执行
|    -> 遇到可编译 submod -> compile + wrap
|    -> 遇到 splitting submod -> 跳过
|
| 3. 返回 split_gm (缝合模块)                   backend.py:473
+-------------------------------------------+
```

**trampoline 机制**（`compile.py:111-201`）：

```python
def trampoline(self, *args, **kwargs):
    use_compiled = is_in_piecewise_cuda_graph()       # :189
    if use_compiled:
        if not state["compiled"]:
            _ensure_compiled(self, *args, **kwargs)   # 首次调用触发 torch.compile
        return state["compiled_callable"](*args, **kwargs)
    else:
        return unbound_fwd(self, *args, **kwargs)     # 非 PCG 走原始 forward
```

trampoline 安装在 model.forward 上，作为"跳板"：
- PCG 模式（capture/replay 时）-> 走 compiled_callable -> split_gm
- 非 PCG 模式（正常推理回退时）-> 走原始 forward

**每个 submod 的包装**（`cuda_piecewise_backend.py:40-100`）：

每个 `CUDAPiecewiseBackend` 实例独立管理自己的 capture size 表：

```
CUDAPiecewiseBackend 实例
+-------------------------------------------+
| concrete_size_entries: {                   |
|   4:   ConcreteSizeEntry(cudagraph, out)  |
|   8:   ConcreteSizeEntry(cudagraph, out)  |
|   16:  ConcreteSizeEntry(cudagraph, out)  |
|   ...                                      |
|   4096: ConcreteSizeEntry(cudagraph, out) |
| }                                          |
+-------------------------------------------+
```

**CUDAPiecewiseBackend 的三阶段调度**（`cuda_piecewise_backend.py:107-206`）：

```
__call__(*args) 的执行路径:

Phase 1 - 首次运行 (first_run_finished = False)
  -> compiled_graph_for_general_shape(*args)     :108-111
  -> 通用形状编译结果，作为 warmup

Phase 2 - Capture 阶段 (cudagraph is None)
  -> warmup: num_finished_warmup < 1 -> 执行 runnable     :147-149
  -> capture: 创建 CUDAGraph, 用 torch.cuda.graph() 捕获  :156-194

Phase 3 - 稳态 Replay (cudagraph is not None)
  -> entry.cudagraph.replay()                              :205
  -> 返回 entry.output                                     :206
```

### 3. 代码入口

| 组件 | 文件 | 行号 |
|------|------|------|
| trampoline 安装 | `compilation/compile.py` | 111-201 |
| `_ensure_compiled()` 触发编译 | `compilation/compile.py` | 160-186 |
| `SGLangBackend.__call__()` | `compilation/backend.py` | 402-473 |
| `split_graph()` | `compilation/backend.py` | 220-263 |
| `PiecewiseCompileInterpreter` | `compilation/backend.py` | 302-343 |
| `CUDAPiecewiseBackend` | `compilation/cuda_piecewise_backend.py` | 40-206 |

---

## 三、Padding 策略

### 1. 问题：如何用有限个 capture size 覆盖无限的 token 数

不可能为每个可能的 token 数（1 到 4096+）都 capture 一个 graph。需要选择一个合理的 capture size 子集，在覆盖率和显存开销之间取得平衡。

### 2. PCG 的解法：递增粒度的 capture size + bisect padding

**capture size 生成**（`server_args.py:1498-1516`）：

| token 范围 | 步长 | size 数量 | 设计理由 |
|------------|------|-----------|---------|
| 4-32 | 4 | 8 | 小 batch 最常见，padding 浪费比例大 |
| 48-256 | 16 | 14 | 中等 batch，粒度适中 |
| 288-512 | 32 | 8 | 稍大 batch |
| 576-1024 | 64 | 8 | 较大 batch |
| 1280-4096 | 256 | 12 | 大 batch，padding 浪费比例小 |
| 4608+ | 512 | 视上限 | 很大 batch |

**递增粒度的直觉**：小 batch 时 padding 10 个 token 可能浪费 50%；大 batch 时 padding 256 个 token 只浪费 6%。所以小 size 需要更密的覆盖。

**运行时 padding 流程**（`piecewise_cuda_graph_runner.py:615-775`）：

```
1. 实际 token 数: num_tokens = 150

2. 二分搜索找最近的 capture size (上界):
   index = bisect.bisect_left([4, 8, ..., 4096], 150)
   static_num_tokens = capture_sizes[index]  -> 假设为 160

3. 零填充 padding 区域:
   buffers.input_ids[150:160].zero_()
   buffers.positions[150:160].zero_()
   buffers.out_cache_loc.zero_()      # 全部清零，防止脏数据

4. 拷贝实际数据到静态 buffer:
   buffers.input_ids[:150].copy_(实际数据)
   buffers.positions[:150].copy_(实际数据)
   buffers.out_cache_loc[:150].copy_(实际loc)

5. 构造 static_forward_batch:
   所有张量指向 buffers.xxx[:160]
```

**内存复用策略**（`piecewise_cuda_graph_runner.py:277-280`）：

```
全局共享 graph memory pool:
  capture 顺序: reversed(capture_sizes) -> 大 size 先 capture

  size=4096 -> 分配 ~4GB pool
  size=2048 -> 复用 4096 的 pool (够用)
  size=1024 -> 复用 4096 的 pool (够用)

  最后一个 submod 的输出用 weak_ref (cuda_piecewise_backend.py:176-182)
  -> 释放 tensor 引用，最大化 pool 内存的复用效率
```

### 3. 代码入口

| 组件 | 文件 | 行号 |
|------|------|------|
| capture size 生成 | `server_args.py` | 1498-1516 |
| max_tokens 默认值 | `server_args.py` | 1355-1375 |
| 运行时 padding | `piecewise_cuda_graph_runner.py` | 615-775 |
| 静态 buffer 分配 | `piecewise_cuda_graph_runner.py` | 214-270 |
| 内存池初始化 | `piecewise_cuda_graph_runner.py` | 277-280 |

---

## 四、完整数据流：从输入到输出

### 1. 初始化阶段（服务器启动时）

```
+------------------------------------------------------------------+
| Phase 0: 初始化判断 (model_runner.py:2606-2722)                    |
|                                                                    |
| init_piecewise_cuda_graphs()                                       |
|   +-- 检查禁用条件 (17 项自动禁用)                                  |
|   +-- 收集 attention_layers, moe_layers, moe_fusions              |
|   +-- PiecewiseCudaGraphRunner(self)                               |
+------------------------------+-----------------------------------+
                               |
+------------------------------v-----------------------------------+
| Phase 1: 分配静态 buffers (piecewise_cuda_graph_runner.py:214-270) |
|                                                                    |
| PrefillInputBuffers:                                               |
|   input_ids      [max_num_tokens]                                  |
|   positions      [max_num_tokens]                                  |
|   out_cache_loc   [max_num_tokens]                                 |
|   input_embeds    [max_num_tokens, hidden_size] (多模态)           |
+------------------------------+-----------------------------------+
                               |
+------------------------------v-----------------------------------+
| Phase 2: torch.compile 编译 (backend.py:364-473)                   |
|                                                                    |
| 2a. patch_model -> MultiPlatformOp 切换到 torch compile           |
| 2b. warmup_compile(num_tokens=capture_sizes[0])                   |
|     -> JIT kernel warmup + 首次 Dynamo trace                     |
| 2c. install_torch_compiled -> 安装 trampoline                     |
| 2d. 对 reversed(capture_sizes) 逐一 warmup_compile                |
|     -> 每个 size 触发一次 model.forward                           |
|     -> trampoline -> compiled_callable -> split_gm.forward        |
|     -> 每个 CUDAPiecewiseBackend 记录当前 shape 的编译结果        |
+------------------------------+-----------------------------------+
                               |
+------------------------------v-----------------------------------+
| Phase 3: CUDA Graph Capture (piecewise_cuda_graph_runner.py:450-613)|
|                                                                    |
| capture() -> reversed(capture_sizes)  [大->小复用内存]            |
|   +-- capture_one_batch_size(num_tokens)                           |
|       +-- 构造 ForwardBatch (bs=1, num_tokens)                    |
|       +-- run_once() x 2:                                         |
|           [1st] warmup -- CUDAPiecewiseBackend 首次执行            |
|           [2nd] capture -- torch.cuda.graph() 捕获                |
|                                                                    |
| 每个 CUDAPiecewiseBackend 的 capture 逻辑:                        |
|   if entry.cudagraph is None:                                      |
|     warmup: num_finished_warmup < 1 -> run runnable               |
|     capture: torch.cuda.graph(cudagraph, pool, stream)            |
+------------------------------------------------------------------+
```

### 2. 运行时 Replay（每次推理请求）

```
推理请求到达: batch 有 150 个 tokens
    |
    v
model_runner.forward_extend()                    model_runner.py:2791
    |
    +-- can_run_graph = pcg_runner.can_run(batch)  :420
    |     检查条件:
    |     - input_embeds == None?                   :423
    |     - not is_target_verify()                  :428
    |     - capture_hidden_mode 匹配                :433
    |     - replace_embeds == None                  :436
    |     - num_tokens <= max_num_tokens            :446-448
    |
    +-- [can_run = True]
    |     |
    |     v
    |   pcg_runner.replay(batch)                    :777
    |     |
    |     +-- replay_prepare(batch)                 :783
    |     |     num_tokens = 150
    |     |     bisect_left -> static_num_tokens = 160
    |     |     zero_: input_ids[150:160] = 0
    |     |     copy_: input_ids[:150] <- 实际数据
    |     |     返回 static_forward_batch
    |     |
    |     +-- model.forward(input_ids[:160], ...)   :794
    |           |
    |           v
    |         trampoline()                          compile.py:188
    |           | is_in_piecewise_cuda_graph() = True
    |           v
    |         split_gm.forward(*args)               缝合模块
    |           |
    |           +-- submod_0.__call__()   CUDAPiecewiseBackend
    |           |     entry(160).cudagraph.replay()
    |           |     return entry.output
    |           |
    |           +-- submod_1(*args)       split op (eager)
    |           |     attention / allreduce 正常执行
    |           |
    |           +-- submod_2.__call__()   CUDAPiecewiseBackend
    |           |     entry(160).cudagraph.replay()
    |           |
    |           +-- ... (交替进行)
    |           |
    |           +-- submod_N.__call__()   最后一个
    |                 cudagraph.replay() -> output (weak_ref)
    |
    +-- 截取输出: output[:150]                     :801-809
        LogitsProcessorOutput(next_token_logits[:150], ...)
        -> 返回给调用方
```

**Capture vs Replay 对比**：

```
             Capture 阶段                    Replay 阶段
             ============                    ============

触发时机:   服务器启动时，仅一次              每次推理请求
入口:       capture() (:450)                replay() (:777)
数据来源:   静态 buffer 的零值 (dummy)       实际请求数据
执行方式:   torch.cuda.graph() 捕获         cudagraph.replay() 重放
模型调用:   model.forward() x 2             model.forward() x 1
            (warmup + capture)              (直接 replay)

每个 submod:
  warmup:   num_finished_warmup < 1         N/A (已完成)
            -> 执行 runnable(*args)         entry.cudagraph.replay()
  capture:  torch.cuda.graph(cudagraph)     -> 返回 entry.output
```

---

## 总结：PCG 核心设计决策

| 维度 | 决策 | 理由 |
|------|------|------|
| **切分粒度** | 按层切分（attention/allreduce 为 split point） | 这些操作依赖动态状态，不适合 graph capture |
| **编译工具** | torch.compile + 自定义 SGLangBackend | 利用 Dynamo 的 tracing，但自己控制 split 和 capture |
| **形状处理** | 预定义 capture sizes + 运行时 padding | 避免动态形状的 CUDA Graph 限制 |
| **内存管理** | 全局共享 pool + 逆序 capture + weak_ref | 最大化显存复用 |
| **降级策略** | can_run() 检查 -> 不满足条件走 eager | 保证正确性优先 |
