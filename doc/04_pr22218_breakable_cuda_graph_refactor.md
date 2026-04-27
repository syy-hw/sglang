# PR #22218 Breakable CUDA Graph 重构深入分析

> 生成日期: 2026-04-27
> 基于 PR: [sgl-project/sglang#22218](https://github.com/sgl-project/sglang/pull/22218) (merged 2026-04-24)
> 作者: Oasis-Git, 灵感来自 #19102 (@cctry)
> 标题: [Experimental] Breakable Piecewise Cuda Graph

---

## 目录

- [一、PR 概览与动机](#一pr-概览与动机)
- [二、架构级重构：从继承到组合](#二架构级重构从继承到组合)
  - [1. BreakableCUDAGraph：从子类到容器](#1-breakablecudagraph从子类到容器)
  - [2. BreakableCUDAGraphCapture：从继承到独立上下文管理器](#2-breakablecudagraphcapture从继承到独立上下文管理器)
  - [3. 消失的底层 API 调用](#3-消失的底层-api-调用)
- [三、新增组件详解](#三新增组件详解)
  - [1. BreakableCudaGraphRunner](#1-breakablecudagraphrunner)
  - [2. context.py 运行时状态模块](#2-contextpy-运行时状态模块)
  - [3. weak_ref_tensor 内存优化](#3-weak_ref_tensor-内存优化)
- [四、模型集成点](#四模型集成点)
  - [1. radix_attention.py：Attention 层的 graph break](#1-radix_attentionpyattention-层的-graph-break)
  - [2. nemotron_h.py：Mamba 层的 graph break](#2-nemotron_hpy-mamba-层的-graph-break)
  - [3. model_runner.py：入口选择逻辑](#3-model_runnerpy-入口选择逻辑)
  - [4. server_args.py：CLI 参数](#4-server_argspy-cli-参数)
- [五、完整流程图与调用链](#五完整流程图与调用链)
  - [1. Capture 阶段](#1-capture-阶段)
  - [2. Replay 阶段](#2-replay-阶段)
  - [3. Eager Break 交替流程](#3-eager-break-交替流程)
- [六、新旧 BCG 对比表](#六新旧-bcg-对比表)
- [七、NPU 适配可行性评估](#七npu-适配可行性评估)
- [八、性能基准 (PR 提供)](#八性能基准-pr-提供)

---

## 一、PR 概览与动机

### 核心目标

PR #22218 将 Breakable CUDA Graph (BCG) 从一个 **decode 路径的调试/不兼容算子辅助工具** 重构为一个 **extend/prefill 路径的 piecewise graph runner**，功能上与 torch.compile-based PCG runner 并行。

### 动机

| 问题 | 解决方案 |
|------|----------|
| PCG 依赖 `torch.compile` + FX graph splitting，编译链长且复杂 | BCG 通过 `@eager_on_graph` 装饰器在运行时插入 break，无需 compile |
| PCG 的 split 点需在编译期通过 `register_split_op` 静态注册 | BCG 的 break 点由模型内部装饰器动态确定 |
| BCG 旧版依赖 7 个 NVIDIA CUDA C API 调用，NPU 不可用 | 新版仅剩 1 个 C API 调用（stream capture 状态查询），NPU 适配可行 |

### 一句话总结

> 新 BCG = **纯 PyTorch 高层 API** 构建的多段 CUDA Graph，以 `eager_on_graph` 装饰器替代 FX graph split，用共享 mempool + `weak_ref_tensor` 管理段间内存。

---

## 二、架构级重构：从继承到组合

### 1. BreakableCUDAGraph：从子类到容器

**旧版** — 继承 `torch.cuda.CUDAGraph`：

```python
# 旧版: 继承关系
class BreakableCUDAGraph(torch.cuda.CUDAGraph):
    def __new__(cls):
        return super().__new__(cls, True)  # multi-segment mode

    def capture_begin(self, pool=None, ...):
        super().capture_begin(pool, ...)   # PyTorch 启动 capture
        _end_capture_segment(stream)       # 立即结束 PyTorch 的
        _begin_capture_segment(stream)     # 开始 BCG 自己的

    def capture_end(self):
        self.last_graph = _end_capture_segment(stream)
        self.last_graph_exec = _instantiate_graph(self.last_graph)  # C API
        for break in breaks:
            graph_exec = _instantiate_graph(break.handle)           # C API
        _begin_capture_segment(stream)     # dummy capture 欺骗 PyTorch
        super().capture_end()

    def replay(self):
        for func, _, handle in self._exec:
            _replay_graph(handle, stream)  # C API: cudaGraphLaunch
            func()
        _replay_graph(self.last_graph_exec, stream)  # C API

    def __del__(self):
        _destroy_graph_exec(...)           # C API: cudaGraphExecDestroy
```

**新版** — 纯容器，不继承任何 PyTorch 类：

```python
# 新版: 纯容器
class BreakableCUDAGraph:
    """Container holding one torch.cuda.CUDAGraph per segment
    plus an eager break function between consecutive segments."""

    def __init__(self) -> None:
        self._segments: list[torch.cuda.CUDAGraph] = []    # 每段是独立的 CUDAGraph
        self._break_fns: list[Callable[[], Any]] = []      # 段之间的 eager 函数

    def replay(self) -> None:
        stream = torch.cuda.current_stream()
        token = _current_stream_var.set(stream)
        try:
            for i, seg in enumerate(self._segments):
                seg.replay()                # PyTorch 高层 API！
                if i < len(self._break_fns):
                    self._break_fns[i]()    # eager 执行
        finally:
            _current_stream_var.reset(token)
```

**关键变化**：

| 维度 | 旧版 | 新版 |
|------|------|------|
| 继承关系 | `torch.cuda.CUDAGraph` 子类 | 纯 Python 类，无继承 |
| 段存储 | 手动管理 `cudaGraphExec` 句柄 | `list[torch.cuda.CUDAGraph]` |
| replay | `rt.cudaGraphLaunch()` (C API) | `seg.replay()` (PyTorch API) |
| 实例化 | `rt.cudaGraphInstantiateWithFlags()` | `torch.cuda.CUDAGraph().capture_end()` 自动完成 |
| 析构 | `__del__` 手动调用 `cudaGraphExecDestroy` | Python GC 自动回收 |
| "三明治"技巧 | 需要 dummy capture 欺骗 PyTorch | 不需要，因为不继承 |

### 2. BreakableCUDAGraphCapture：从继承到独立上下文管理器

**旧版** — 继承 `torch.cuda.graph`：

```python
# 旧版: 继承 torch.cuda.graph 上下文管理器
class BreakableCUDAGraphCapture(torch.cuda.graph):
    def __init__(self, cuda_graph, pool, stream, ...):
        super().__init__(cuda_graph, pool=pool, stream=stream, ...)

    def __enter__(self):
        _install_wait_stream_hook()
        self._breaks_token = _captured_graphs_var.set([])
        return super().__enter__()  # -> torch.cuda.graph.__enter__()
        # -> 触发 BreakableCUDAGraph.capture_begin()（因为继承关系）

    def __exit__(self, *args):
        super().__exit__(*args)  # -> torch.cuda.graph.__exit__()
        # -> 触发 BreakableCUDAGraph.capture_end()
        ...
```

**新版** — 完全独立的上下文管理器：

```python
# 新版: 独立上下文管理器
class BreakableCUDAGraphCapture:
    def __init__(self, cuda_graph, pool, stream, capture_error_mode):
        assert isinstance(cuda_graph, BreakableCUDAGraph)
        self.cuda_graph = cuda_graph
        self._pool = pool if pool is not None else (0, 0)
        self._stream = stream
        self._capture_error_mode = capture_error_mode

    def __enter__(self):
        _install_wait_stream_hook()
        if self._stream is not None:
            self._stream_ctx = torch.cuda.stream(self._stream)
            self._stream_ctx.__enter__()
        self._capture_token = _current_capture_var.set(self)
        self._stream_token = _current_stream_var.set(...)
        self._forked_token = _forked_streams_var.set(set())
        self._begin_new_segment()       # 直接创建第一个 segment
        return self

    def __exit__(self, *args):
        try:
            self._end_current_segment()  # 结束最后一个 segment
        finally:
            # 清理所有 context vars 和 hook
            _forked_streams_var.reset(self._forked_token)
            _current_stream_var.reset(self._stream_token)
            _current_capture_var.reset(self._capture_token)
            if self._stream_ctx is not None:
                self._stream_ctx.__exit__(*args)
            _uninstall_wait_stream_hook()
        return False
```

**Segment 生命周期（新版核心）**：

```python
def _begin_new_segment(self) -> None:
    graph = torch.cuda.CUDAGraph()                    # PyTorch 高层 API
    graph.capture_begin(
        pool=self._pool,                               # 共享 mempool
        capture_error_mode=self._capture_error_mode
    )
    self.cuda_graph._segments.append(graph)

def _end_current_segment(self) -> None:
    # Auto-join forked side streams
    forked = _forked_streams_var.get()
    if forked:
        for side in list(forked):
            if _is_capturing(side.cuda_stream):
                _original_wait_stream(main_stream, side)
        forked.clear()
    self.cuda_graph._segments[-1].capture_end()       # PyTorch 高层 API
```

### 3. 消失的底层 API 调用

**旧版依赖的 7 个 CUDA C API**：

| API | 用途 | 新版状态 |
|-----|------|----------|
| `rt.cudaStreamBeginCapture` | 开始 stream capture | **消除** -> `torch.cuda.CUDAGraph().capture_begin()` |
| `rt.cudaStreamEndCapture` | 结束 stream capture | **消除** -> `segment.capture_end()` |
| `rt.cudaGraphInstantiateWithFlags` | 实例化图 | **消除** -> `capture_end()` 内部自动完成 |
| `rt.cudaGraphLaunch` | 启动图执行 | **消除** -> `seg.replay()` |
| `rt.cudaGraphDestroy` | 销毁图 | **消除** -> Python GC |
| `rt.cudaGraphExecDestroy` | 销毁执行句柄 | **消除** -> Python GC |
| `rt.cudaStreamGetCaptureInfo` | 查询 capture 状态 | **保留** -> `_is_capturing()` / `_capture_status()` |

**仅剩的 C API 依赖**（用于 `wait_stream` hook 中的 stream fork/join 追踪）：

```python
def _is_capturing(stream_ptr: int) -> bool:
    _, _, capture_status = checkCudaErrors(
        rt.cudaStreamGetCaptureInfo(stream_ptr)
    )
    return capture_status == rt.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

def _capture_status(stream_ptr: int):
    return checkCudaErrors(rt.cudaStreamGetCaptureInfo(stream_ptr))[2]
```

这两个函数仅在 `_hooked_wait_stream` 中被调用，用于在 segment 结束前判断 side stream 是否仍在 capture 状态。

---

## 三、新增组件详解

### 1. BreakableCudaGraphRunner

新增文件 `breakable_cuda_graph_runner.py`（402 行），功能上与 `PiecewiseCudaGraphRunner` 并行。

```
+--------------------------------------------------------------+
|              BreakableCudaGraphRunner 架构                    |
|                                                              |
|  +------------------+    +----------------------------+      |
|  | _init_buffers()  |--->| PrefillInputBuffers        |      |
|  | 分配静态 buffer  |    | (复用 PCG 的 buffer 类型)   |      |
|  +------------------+    +----------------------------+      |
|                                                              |
|  +------------------+    +----------------------------+      |
|  | _warmup()         |--->| 1 次 eager forward         |      |
|  | 预热模型          |    | 初始化 CUDA context         |      |
|  +------------------+    +----------------------------+      |
|                                                              |
|  +------------------+    +----------------------------+      |
|  | _capture_all()    |--->| for num_tokens in sizes:   |      |
|  | 逐 size 捕获      |    |   _capture_one() x N       |      |
|  | (大->小共享内存)   |    |   -> BreakableCUDAGraph()   |      |
|  +------------------+    |   -> BreakableCUDAGraphCapt |      |
|                          +----------------------------+      |
|                                                              |
|  +------------------+    +----------------------------+      |
|  | replay()          |--->| replay_prepare()           |      |
|  | 推理重放          |    | -> 拷贝数据到静态 buffer    |      |
|  |                   |    | -> graph.replay()           |      |
|  +------------------+    +----------------------------+      |
|                                                              |
|  +------------------+    +----------------------------+      |
|  | can_run()         |--->| batch_size == 1            |      |
|  | 判断能否走 graph   |    | && tokens <= max_tokens    |      |
|  +------------------+    +----------------------------+      |
|                                                              |
|  特殊设计:                                                    |
|  replay_prepare = PiecewiseCudaGraphRunner.replay_prepare    |
|  (绑定方法，不继承)                                            |
+--------------------------------------------------------------+
```

**关键设计决策**：

1. **绑定而非继承**：`replay_prepare = PiecewiseCudaGraphRunner.replay_prepare` — 直接绑定 PCG 的 buffer 填充方法，避免继承带来的耦合。

2. **EXTEND 模式 + batch_size=1**：
   - 使用 `ForwardMode.EXTEND`（prefill 模式）进行 capture
   - `batch_size=1` 限制（`can_run()` 中强制检查）
   - 原因：BCG 的 logits-gather / sampler 路径产生 bs=1 输出，多 batch 会返回错误形状

3. **共享 mempool**：所有 size 的 graph 共享同一个 `graph_pool_handle()`，从大到小 capture，最大化内存复用。

4. **静态 buffer 策略**：使用 `PrefillInputBuffers`（与 PCG 相同），但 `capture_hidden_mode=CaptureHiddenMode.NULL`（不捕获中间隐藏状态）。

### 2. context.py 运行时状态模块

新增文件 `context.py`（39 行），简洁的全局状态管理：

```python
# context.py — BCG 运行时状态
_in_breakable_cuda_graph = False

def is_in_breakable_cuda_graph() -> bool:
    return _in_breakable_cuda_graph

@contextmanager
def enable_breakable_cuda_graph():
    global _in_breakable_cuda_graph
    _in_breakable_cuda_graph = True
    try:
        yield
    finally:
        _in_breakable_cuda_graph = False
```

**设计意图**：

- 与 PCG 的 `piecewise_context_manager.py` 完全独立
- 使用全局变量（非 ContextVar），因为 BCG 的 capture/replay 在同一进程内
- 模型层通过 `is_in_breakable_cuda_graph()` 查询当前是否在 BCG 上下文中，决定走哪条代码路径

### 3. weak_ref_tensor 内存优化

新增的 `_weak_ref_if_tensor()` 函数替代了旧版的强引用 closure 捕获：

```python
def _weak_ref_if_tensor(x):
    """Return a weak-ref tensor view for tensors; pass-through for non-tensors."""
    if torch.is_tensor(x):
        from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors
        return weak_ref_tensors(x)
    return x
```

**为什么需要 weak_ref**：

```
旧版:
  captured_args = args          # 强引用 -> 阻止 GC 回收中间 tensor
  captured_output = output      # 强引用 -> 每层中间结果常驻内存

新版:
  captured_args = tuple(_weak_ref_if_tensor(a) for a in args)    # 弱引用
  captured_output = _weak_ref_if_tensor(output)                   # 弱引用
```

**内存管理机制**：

```
Segment 0 CUDAGraph  ----+
                         +--- 共享 mempool (use_count = 2)
Segment 1 CUDAGraph  ----+

当 segment 0 的 CUDAGraph 析构时:
  -> releasePool -> use_count -= 1
  -> use_count 仍为 1（segment 1 还活着）
  -> pool 不释放 -> weak_ref tensor 仍有效

当 segment 1 也析构时:
  -> use_count -= 1 -> use_count = 0
  -> pool 释放 -> 所有中间 tensor 内存回收
```

这意味着每层的中间 tensor 不需要 Python 级别的强引用来维持生命周期，mempool 的 `use_count` 自动管理。这在深层模型（如 Qwen3-235B）中显著减少内存占用。

---

## 四、模型集成点

### 1. radix_attention.py：Attention 层的 graph break

```python
# radix_attention.py — 新增 BCG 分支

from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.model_executor.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)

class RadixAttention:
    def forward(self, ...):
        if is_in_piecewise_cuda_graph() or is_in_breakable_cuda_graph():
            output = torch.empty_like(q)
            if is_in_breakable_cuda_graph():
                bcg_unified_attention_with_output(  # <- BCG 路径：会触发 graph break
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            else:
                unified_attention_with_output(      # <- PCG 路径：不 break
                    q, k, v, output, save_kv_cache, self.layer_id, **kwargs
                )
            return output
        ...

# 模块级：创建 BCG 版本的 attention 函数
bcg_unified_attention_with_output = eager_on_graph(True)(unified_attention_with_output)
```

**BCG vs PCG 在 attention 层的区别**：

| 路径 | 函数 | graph break? | 行为 |
|------|------|-------------|------|
| PCG | `unified_attention_with_output` | 否 | 作为 compiled submod 的一部分 |
| BCG | `bcg_unified_attention_with_output` | **是** | `@eager_on_graph(True)` 包裹，每次调用触发 segment split |
| Eager | `forward_batch.attn_backend.forward()` | 不适用 | 完全 eager 执行 |

### 2. nemotron_h.py：Mamba 层的 graph break

```python
# nemotron_h.py — 新增 BCG 分支

class NemotronHDecoderLayer:
    def forward(self, ...):
        if is_in_breakable_cuda_graph():
            output = torch.empty_like(hidden_states)
            breakable_nemotron_mamba2_with_output(hidden_states, output, self.layer_id)
            return output, residual
        # PCG path follows...

# 模块级
breakable_nemotron_mamba2_with_output = eager_on_graph(True)(
    nemotron_mamba2_with_output
)
```

**Break 点总览**（以 dense transformer 为例）：

```
model.forward()
  +-- [Segment 0] layers[0..N-1].forward() 的前半部分（embed, norm, linear...）
  +-- Graph Break -> bcg_unified_attention_with_output (layer 0)
  +-- [Segment 1] layer 0 的后半部分 + layers[1..N-1] 的前半部分
  +-- Graph Break -> bcg_unified_attention_with_output (layer 1)
  +-- ...
  +-- Graph Break -> bcg_unified_attention_with_output (layer N-1)
  +-- [Segment N] layers[N-1] 的后半部分 + lm_head

对于 Nemotron-H (hybrid mamba):
  每个 mamba mixer 层也会触发 graph break
```

### 3. model_runner.py：入口选择逻辑

```python
# model_runner.py — init_piecewise_cuda_graphs()

if self.server_args.enable_breakable_cuda_graph:
    # Experimental feature
    self.piecewise_cuda_graph_runner = BreakableCudaGraphRunner(self)
else:
    self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)
```

**注意**：BCG runner 被赋值给同一个 `piecewise_cuda_graph_runner` 属性，下游的 `forward_extend()` 逻辑无需修改。

### 4. server_args.py：CLI 参数

```python
# 新增 CLI 参数
parser.add_argument(
    "--enable-breakable-cuda-graph",
    action="store_true",
    help="Use breakable CUDA graph for piecewise capture instead of torch.compile-based splitting.",
)
```

**使用方式**：
```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3-8B \
    --enable-breakable-cuda-graph \
    --piecewise-cuda-graph-tokens 2048,4096,8192
```

---

## 五、完整流程图与调用链

### 1. Capture 阶段

```
ModelRunner.init_piecewise_cuda_graphs()                    model_runner.py:2729
  |
  +-- enable_breakable_cuda_graph=True?
  |   +-- YES -> BreakableCudaGraphRunner(model_runner)      breakable_cuda_graph_runner.py:120
  |
  +-- BreakableCudaGraphRunner.__init__()
       +-- _init_buffers()                                    :176
       |   +-- PrefillInputBuffers -> share_buffers()
       +-- 全局 graph memory pool
       +-- _warmup()                                          :250
       |   +-- 1 次 eager forward
       +-- _capture_all()                                     :260
            |
            +-- with enable_breakable_cuda_graph():
            +-- with graph_capture() -> stream, pool
            |
            +-- for num_tokens in reversed(capture_num_tokens):
                 |
                 +-- _capture_one(num_tokens, pool, stream)   :294
                      |
                      +-- _build_capture_forward_batch()      :208
                      |   +-- ForwardMode=EXTEND, bs=1
                      |
                      +-- run_once() x 2 (warmup)
                      |
                      +-- graph = BreakableCUDAGraph()        :305
                      |
                      +-- with BreakableCUDAGraphCapture(graph, pool, stream): :306
                            |
                            +-- __enter__()                   breakable_cuda_graph.py:284
                            |   +-- _install_wait_stream_hook()
                            |   +-- torch.cuda.stream(stream)
                            |   +-- _current_capture_var.set(self)
                            |   +-- _begin_new_segment()      :298
                            |       +-- torch.cuda.CUDAGraph()
                            |           .capture_begin(pool)  <- PyTorch 高层 API
                            |
                            +-- run_once() -> model.forward()
                            |   |
                            |   +-- [CUDA ops -> 录入 segment 0]
                            |   |
                            |   +-- 遇到 bcg_unified_attention_with_output (layer 0):
                            |   |   +-- capture._end_current_segment()
                            |   |   |   +-- segment_0.capture_end()  <- PyTorch 高层 API
                            |   |   +-- inner(*args)  <- eager 执行 attention
                            |   |   +-- weak_ref 捕获 args/output
                            |   |   +-- graph._break_fns.append(replay_fn)
                            |   |   +-- capture._begin_new_segment()
                            |   |       +-- segment_1 = torch.cuda.CUDAGraph()
                            |   |           .capture_begin(pool)
                            |   |
                            |   +-- [CUDA ops -> 录入 segment 1]
                            |   +-- ... 更多 break 点 ...
                            |   +-- [CUDA ops -> 录入 segment N]
                            |
                            +-- __exit__()                    :290
                                +-- _end_current_segment()
                                |   +-- segment_N.capture_end()  <- PyTorch 高层 API
                                +-- reset context vars
                                +-- _uninstall_wait_stream_hook()
```

### 2. Replay 阶段

```
ModelRunner.forward_extend()                                model_runner.py:2818
  |
  +-- piecewise_cuda_graph_runner.can_run(batch)
  |   +-- BreakableCudaGraphRunner.can_run()                 breakable_cuda_graph_runner.py:278
  |       +-- batch_size > 1 -> False
  |       +-- input_embeds is not None -> False
  |       +-- num_tokens <= max -> True
  |
  +-- piecewise_cuda_graph_runner.replay(batch)
      +-- BreakableCudaGraphRunner.replay()                  :309
          |
          +-- replay_prepare(forward_batch)                  (绑定自 PCG runner)
          |   +-- bisect_left 找最近的 capture size
          |   +-- padding: zero_ 多余部分
          |   +-- copy_: 拷贝实际数据到静态 buffer
          |
          +-- 更新静态 seq_lens / extend_* / req_pool_indices
          |
          +-- with enable_breakable_cuda_graph():
          |   +-- set_forward_context()
          |   +-- self.graphs[static_num_tokens].replay()    :350
          |       |
          |       +-- BreakableCUDAGraph.replay()            breakable_cuda_graph.py:243
          |           |
          |           +-- for i, seg in enumerate(_segments):
          |               +-- seg.replay()                   <- PyTorch 高层 API
          |               |   (重放 captured kernel)
          |               +-- if i < len(_break_fns):
          |                   _break_fns[i]()
          |                   +-- captured_inner(*captured_args)  <- 重执行 eager 函数
          |                   +-- _copy_output(captured_output, new_out)
          |
          +-- 返回 output buffer (LogitsProcessorOutput)
```

### 3. Eager Break 交替流程

```
Capture 时序:
================================================================

  segment_0.capture_begin(pool)
  |
  |  [embed + norm + qkv_proj + ...]  <- CUDA ops 被 graph 记录
  |
  +-- bcg_unified_attention_with_output (layer 0)
  |   +-- segment_0.capture_end()
  |   +-- attention(q, k, v, output, ...)  <- eager 执行
  |   +-- graph._break_fns.append(replay_fn_0)
  |   +-- segment_1.capture_begin(pool)    <- 共享 mempool!
  |
  |  [residual + norm + MLP + ...]  <- CUDA ops 被新 segment 记录
  |
  +-- bcg_unified_attention_with_output (layer 1)
  |   +-- segment_1.capture_end()
  |   +-- attention(...)
  |   +-- graph._break_fns.append(replay_fn_1)
  |   +-- segment_2.capture_begin(pool)
  |
  |  ...
  |
  segment_N.capture_end()


Replay 时序:
================================================================

  segment_0.replay()                      <- 重放前半部分 kernel
  |
  replay_fn_0()                           <- eager 重执行 attention
  |  +-- captured_inner(*weak_ref_args)   <- args 指向已更新的静态 buffer
  |  +-- _copy_output(captured_output, new_out)  <- 原地写回
  |
  segment_1.replay()                      <- 重放中间部分 kernel
  |
  replay_fn_1()
  |  +-- ...
  |
  ...
  |
  segment_N.replay()                      <- 重放最后一段
```

---

## 六、新旧 BCG 对比表

| 维度 | 旧版 BCG (pre-PR22218) | 新版 BCG (PR #22218) |
|------|------------------------|----------------------|
| **定位** | Decode 路径的调试/不兼容算子辅助 | Extend/Prefill 路径的 piecewise graph runner |
| **触发方式** | `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1` 环境变量 | `--enable-breakable-cuda-graph` CLI 参数 |
| **类继承** | `BreakableCUDAGraph(torch.cuda.CUDAGraph)` + `BreakableCUDAGraphCapture(torch.cuda.graph)` | 纯容器 + 独立上下文管理器，无任何继承 |
| **段存储** | `_exec: list[GraphBreakInfo]`（手动管理句柄） | `_segments: list[torch.cuda.CUDAGraph]`（PyTorch 管理） |
| **Capture API** | `rt.cudaStreamBeginCapture` / `rt.cudaStreamEndCapture` | `CUDAGraph.capture_begin()` / `CUDAGraph.capture_end()` |
| **实例化** | `rt.cudaGraphInstantiateWithFlags()` | `capture_end()` 内部自动完成 |
| **Replay API** | `rt.cudaGraphLaunch()` | `seg.replay()` |
| **资源释放** | `__del__` 手动调用 `cudaGraphExecDestroy` | Python GC 自动回收 |
| **"三明治"技巧** | 需要 dummy capture 欺骗 PyTorch | 不需要（不继承 PyTorch 类） |
| **CUDA C API 依赖** | 7 个 | 1 个（仅 `cudaStreamGetCaptureInfo` 在 hook 中） |
| **Break 信息** | `GraphBreakInfo` NamedTuple (func, output, handle) | `_break_fns: list[Callable]`（简化为函数列表） |
| **Context 变量** | `_captured_graphs_var` 存 break 列表 | `_current_capture_var` 存 capture 上下文 |
| **内存管理** | 强引用 closure 捕获 args/output | `weak_ref_tensor` 弱引用 |
| **独立 Runner** | 无，复用 `CudaGraphRunner` | `BreakableCudaGraphRunner`（402 行新文件） |
| **Batch Size** | decode (通常 bs=1) | extend prefill, 限制 bs=1 |
| **适用阶段** | Decode | Extend/Prefill |
| **NPU 可行性** | 不可能（7 个 C API 无替代） | 大体可行（仅 1 个 C API 待替换） |

---

## 七、NPU 适配可行性评估

### 模块逐项分析

| 模块 | NPU 适配 | 难度 | 说明 |
|------|----------|------|------|
| `BreakableCUDAGraph` | 直接可用 | 无 | 仅用 `torch.cuda.CUDAGraph`，NPU 有 `torch.npu.NPUGraph()` |
| `_begin_new_segment` | `capture_begin` | 低 | `NPUGraph().capture_begin()` 已有 |
| `_end_current_segment` | `capture_end` | 低 | `NPUGraph().capture_end()` 已有 |
| `seg.replay()` | `seg.replay()` | 低 | `NPUGraph().replay()` 已有 |
| `_weak_ref_if_tensor` | 需验证 | 中 | 需确认 NPU 上 `weak_ref_tensors` 的行为 |
| `_is_capturing` / `_capture_status` | **需替换** | **高** | 唯一剩余的 `cudaStreamGetCaptureInfo` C API 调用 |
| `_hooked_wait_stream` | 需适配 | 中 | 依赖 `_is_capturing`，NPU stream 语义可能不同 |
| `BreakableCudaGraphRunner` | 需适配 | 低 | `torch.get_device_module(device)` 已抽象设备 |
| `capture_error_mode` | 需确认 | 低 | NPU 的 `capture_begin` 是否支持该参数 |
| `context.py` | 直接可用 | 无 | 纯 Python 全局变量，无设备依赖 |

### NPU 适配路线图

```
Phase 1: 最小可行适配
-----------------------
1. 创建 NPUBreakableCUDAGraph / NPUBreakableCUDAGraphCapture
   - 替换 torch.cuda.CUDAGraph -> torch.npu.NPUGraph
   - 替换 capture_begin/end -> NPU 版本

2. 处理 _is_capturing 的 C API 依赖
   方案 A: NPU 不需要 stream fork/join 追踪（如 CANN 无此语义）
           -> 直接 stub 返回 False
   方案 B: torch_npu 提供 NPU 等价 API
           -> 用 NPU API 替换
   方案 C: 用 Python 标志位跟踪 capture 状态
           -> 在 capture_begin/end 时设置标志

3. 创建 NPUBreakableCudaGraphRunner
   - 继承 BreakableCudaGraphRunner
   - override 设备相关方法

Phase 2: 验证与优化
-----------------------
4. 验证 weak_ref_tensor 在 NPU mempool 上的行为
5. 验证 shared mempool 的 use_count 语义
6. 性能基准对比
```

### 与 PCG NPU 适配的对比

| 维度 | PCG NPU 适配 | BCG NPU 适配 (PR22218) |
|------|-------------|------------------------|
| 已有实现 | `NPUPiecewiseBackend` (生产可用) | 尚无，但架构已可行 |
| 适配工作量 | 已完成 | 预计 1-2 周（Phase 1） |
| 核心障碍 | 无（纯 PyTorch API） | `_is_capturing` 的 C API（可绕过） |
| 优势 | 经过生产验证 | 无 torch.compile 依赖，更简单 |

---

## 八、性能基准 (PR 提供)

PR 作者提供了 mGSM8K 基准测试（200 questions）：

| 模型配置 | PCG score | PCG tput | PCG cap_GB | BCG score | BCG tput | BCG cap_GB |
|----------|-----------|----------|------------|-----------|----------|------------|
| qwen3_8b_tp1 | 0.850 | 3352.6 | 1.43 | 0.815 | 3366.0 | 1.40 |
| qwen3_8b_tp2 | 0.835 | 4918.5 | 1.85 | 0.825 | 4989.9 | 1.93 |
| qwen3_32b_tp1 | 0.965 | 818.8 | 2.78 | 0.955 | 665.5 | 2.51 |
| qwen3_32b_tp4 | 0.975 | 2267.6 | 2.81 | 0.965 | 2284.1 | 2.84 |
| qwen3_30b_a3b_tp1 | 0.955 | 1689.8 | 1.37 | 0.955 | 1669.6 | 1.35 |
| qwen3_30b_a3b_tp2 | 0.955 | 2634.7 | 1.96 | 0.960 | 2560.5 | 2.04 |
| qwen3_30b_a3b_ep2 | 0.940 | 2452.3 | 2.06 | 0.950 | 2422.7 | 2.13 |
| qwen3_235b_tp8 | 0.980 | 901.2 | 3.53 | 0.985 | 892.2 | 3.54 |
| qwen3_235b_ep8 | 0.980 | 754.5 | 3.76 | 0.975 | 728.0 | 3.80 |
| nemotronh_8b_tp2 | 0.310 | 3610.5 | 1.66 | 0.300 | 3544.4 | 1.86 |

**观察**：

1. **吞吐量**：BCG 与 PCG 基本持平（差异 <5%），部分场景略优
2. **精度**：BCG score 与 PCG 非常接近，在可接受范围内
3. **显存占用**：BCG 与 PCG 相当，部分场景略高（如 nemotronh_8b: 1.86 vs 1.66 GB）
4. **大模型**：Qwen3-235B 上 BCG 几乎无性能损失（tput 差异 <2%）

---

## 附录：文件变更清单

| 文件 | 变更 | 行数 |
|------|------|------|
| `breakable_cuda_graph.py` | 重构（核心） | +123/-135 |
| `breakable_cuda_graph_runner.py` | 新增 | +402 |
| `context.py` | 新增 | +39 |
| `radix_attention.py` | 修改（集成） | +17/-3 |
| `nemotron_h.py` | 修改（集成） | +16/-1 |
| `model_runner.py` | 修改（入口） | +8/-1 |
| `server_args.py` | 修改（CLI） | +6 |
| `test_breakable_cuda_graph.py` | 重命名+扩展 | +41/-6 |

**净变更**: ~+652/-146 行
