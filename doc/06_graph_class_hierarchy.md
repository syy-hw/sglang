# SGLang Graph 特性 — 完整类层次结构

> 本文档梳理 sglang 中所有 graph 相关的类：定义位置、创建上下文、调用关系。
> 包含 mermaid 类图和流程图。

---

## 1. 类清单

所有路径相对于 `python/sglang/srt/`。

| 类 | 文件 | 行号 | 职责 |
|---|------|------|------|
| `CudaGraphRunner` | `model_executor/cuda_graph_runner.py` | 558 | Decode 模式 CUDA graph 捕获与重放 |
| `NPUGraphRunner` | `hardware_backend/npu/graph_runner/npu_graph_runner.py` | 73 | 昇腾 NPU 设备的 CudaGraphRunner 子类 |
| `CPUGraphRunner` | `model_executor/cpu_graph_runner.py` | 480 | CPU 设备 decode 路径，用 `torch.compile` |
| `PiecewiseCudaGraphRunner` | `model_executor/piecewise_cuda_graph_runner.py` | 145 | Extend 模式 PCG（torch.compile FX 分割） |
| `BreakableCudaGraphRunner` | `model_executor/breakable_cuda_graph_runner.py` | 71 | Extend 模式 BCG（eager break point） |
| `BreakableCUDAGraph` | `breakable_cuda_graph/breakable_cuda_graph.py` | 251 | 容器：CUDAGraph 段列表 + break 函数列表 |
| `BreakableCUDAGraphCapture` | `breakable_cuda_graph/breakable_cuda_graph.py` | 271 | 上下文管理器：管理多段捕获生命周期 |
| `eager_on_graph()` | `breakable_cuda_graph/breakable_cuda_graph.py` | 209 | 装饰器：在捕获中插入 graph break |
| `break_graph()` | `breakable_cuda_graph/breakable_cuda_graph.py` | 348 | 显式断点（函数体为空） |
| `DecodeInputBuffers` | `model_executor/cuda_graph_runner.py` | 133 | Decode 模式预分配 GPU 输入缓冲区 |
| `PrefillInputBuffers` | `model_executor/piecewise_cuda_graph_runner.py` | 72 | Extend 模式预分配 GPU 输入缓冲区 |
| `ForwardInputBuffers` | `model_executor/input_buffers.py` | 15 | 输入缓冲区管理基类 |

---

## 2. 类关系图

```mermaid
classDiagram
    direction TB

    class ModelRunner {
        +graph_runner
        +piecewise_cuda_graph_runner
        +init_device_graphs()
        +init_piecewise_cuda_graphs()
        +_forward_raw()
        +forward_extend()
    }

    class CudaGraphRunner {
        +graphs dict~bs, CUDAGraph~
        +output_buffers dict~bs, Output~
        +buffers DecodeInputBuffers
        +capture()
        +replay()
        +can_run()
        +_capture_graph()
        +_create_device_graph()
    }

    class NPUGraphRunner {
        +_create_device_graph() NPUGraph
        +_capture_graph()
        +replay()
    }

    class CPUGraphRunner {
        +graphs dict~bs, compiled_fn~
        +capture()
        +replay()
    }

    class PiecewiseCudaGraphRunner {
        +buffers PrefillInputBuffers
        +warmup_compile()
        +capture()
        +replay()
        +replay_prepare()
    }

    class BreakableCudaGraphRunner {
        +graphs dict~tokens, BreakableCUDAGraph~
        +buffers PrefillInputBuffers
        +_capture_all()
        +_capture_one()
        +replay()
        +replay_prepare() 从PiecewiseCudaGraphRunner借用的方法
    }

    class BreakableCUDAGraph {
        +_segments list~CUDAGraph~
        +_break_fns list~Callable~
        +replay()
    }

    class BreakableCUDAGraphCapture {
        +cuda_graph BreakableCUDAGraph
        +_pool MempoolId
        +__enter__()
        +__exit__()
        +_begin_new_segment()
        +_end_current_segment()
    }

    NPUGraphRunner --|> CudaGraphRunner : 继承

    ModelRunner *-- CudaGraphRunner : self.graph_runner
    ModelRunner *-- CPUGraphRunner : self.graph_runner CPU设备
    ModelRunner *-- PiecewiseCudaGraphRunner : self.piecewise_cuda_graph_runner
    ModelRunner *-- BreakableCudaGraphRunner : BCG启用时

    CudaGraphRunner o-- BreakableCUDAGraph : BCG启用时用于decode
    BreakableCudaGraphRunner ..> BreakableCUDAGraph : 创建
    BreakableCudaGraphRunner ..> BreakableCUDAGraphCapture : 作为上下文创建
    BreakableCUDAGraphCapture o-- BreakableCUDAGraph : 捕获到其中
```

---

## 3. 两套并行的 Graph 系统

sglang 同时运行 **两套独立的 graph 系统**，一套用于 decode，一套用于 extend（prefill）：

```mermaid
graph TB
    subgraph "Decode 路径"
        MR[ModelRunner._forward_raw] --> GR{设备类型?}
        GR -->|CUDA| CGR[CudaGraphRunner]
        GR -->|NPU| NGR[NPUGraphRunner]
        GR -->|CPU| CPUR[CPUGraphRunner]
    end

    subgraph "Extend / Prefill 路径"
        ME[ModelRunner.forward_extend] --> PCG{BCG启用?}
        PCG -->|否| PW[PiecewiseCudaGraphRunner]
        PCG -->|是| BCG[BreakableCudaGraphRunner]
    end
```

| 系统 | Runner | Forward 模式 | 键 | Graph 技术 |
|------|--------|-------------|-----|-----------|
| Decode | `CudaGraphRunner` / `NPUGraphRunner` / `CPUGraphRunner` | DECODE, TARGET_VERIFY, IDLE | batch_size | `torch.cuda.CUDAGraph` / `torch.npu.NPUGraph` / `torch.compile` |
| Extend | `PiecewiseCudaGraphRunner` / `BreakableCudaGraphRunner` | EXTEND, MIXED | num_tokens | torch.compile FX 分割 **或** BreakableCUDAGraph 分段 |

---

## 4. 实例化 — 谁创建谁

所有 graph runner 由 `ModelRunner` 在初始化时创建。

```mermaid
sequenceDiagram
    participant MR as ModelRunner
    participant CGR as CudaGraphRunner
    participant NGR as NPUGraphRunner
    participant CPUR as CPUGraphRunner
    participant PW as PiecewiseCudaGraphRunner
    participant BCG as BreakableCudaGraphRunner

    Note over MR: init_device_graphs() — model_runner.py:2591
    alt CUDA/MUSA 设备
        MR->>CGR: CudaGraphRunner(self)
    else NPU 设备
        MR->>NGR: NPUGraphRunner(self)
    else CPU 设备
        MR->>CPUR: CPUGraphRunner(self)
    end
    Note over MR: self.graph_runner = runner

    Note over MR: init_piecewise_cuda_graphs() — model_runner.py:2643
    alt --enable-breakable-cuda-graph
        MR->>BCG: BreakableCudaGraphRunner(self)
    else --piecewise-cuda-graph-tokens
        MR->>PW: PiecewiseCudaGraphRunner(self)
    end
    Note over MR: self.piecewise_cuda_graph_runner = runner
```

---

## 5. Capture（捕获）流程

### 5A. Decode 捕获（CudaGraphRunner）

```mermaid
flowchart TD
    A["CudaGraphRunner.__init__"] --> B[capture]
    B --> C["遍历 batch_size（从大到小）"]
    C --> D[capture_one_batch_size]
    D --> E["构建 ForwardBatch + DecodeInputBuffers"]
    E --> F["Warmup: run_once x2"]
    F --> G{BCG启用?}
    G -->|是| H["_create_device_graph → BreakableCUDAGraph"]
    G -->|否| I["_create_device_graph → torch.cuda.CUDAGraph"]
    H --> J["_capture_graph: BreakableCUDAGraphCapture 上下文"]
    I --> K["_capture_graph: CUDAGraph 上下文"]
    J --> L["存入 self.graphs / self.output_buffers"]
    K --> L
    L --> C
```

### 5B. Extend 捕获 — BCG（BreakableCudaGraphRunner）

```mermaid
flowchart TD
    A["BreakableCudaGraphRunner.__init__"] --> B["_warmup: 一次前向"]
    B --> C[_capture_all]
    C --> D["进入 enable_breakable_cuda_graph 上下文"]
    D --> E["遍历 num_tokens（从大到小）"]
    E --> F[_capture_one]
    F --> G["构建 ForwardBatch + PrefillInputBuffers"]
    G --> H["Warmup: run_once x2"]
    H --> I["创建 BreakableCUDAGraph"]
    I --> J["进入 BreakableCUDAGraphCapture 上下文"]
    J --> K["__enter__ → _begin_new_segment → CUDAGraph.capture_begin"]
    K --> L["run_once（模型前向）"]
    L --> M{遇到 @eager_on_graph?}
    M -->|否| N["继续在当前段捕获"]
    N --> T["__exit__ → _end_current_segment"]
    M -->|是| O["_end_current_segment: CUDAGraph.capture_end"]
    O --> P["eager 执行该函数"]
    P --> Q["weak-ref 参数和输出"]
    Q --> R["注册 replay_fn 到 break_fns"]
    R --> S["_begin_new_segment: 新 CUDAGraph.capture_begin"]
    S --> L
    T --> U["存入 self.graphs"]
    U --> E
```

### 5C. Extend 捕获 — PCG（PiecewiseCudaGraphRunner）

```mermaid
flowchart TD
    A["PiecewiseCudaGraphRunner.__init__"] --> B["install_torch_compiled: torch.compile 编译模型"]
    B --> C["warmup_compile: 按token数预热编译"]
    C --> D[capture]
    D --> E["遍历 num_tokens（从大到小）"]
    E --> F[capture_one_batch_size]
    F --> G["构建 ForwardBatch + PrefillInputBuffers"]
    G --> H["Warmup run_once x2"]
    H --> I["torch.compile 通过 FX graph partitioner 自动分割"]
    I --> J["存入 self.graphs"]
    J --> E
```

---

## 6. Replay（重放）流程

### 6A. Decode 重放

```mermaid
sequenceDiagram
    participant MR as ModelRunner._forward_raw
    participant GR as CudaGraphRunner
    participant Buf as DecodeInputBuffers
    participant Graph as CUDAGraph / BreakableCUDAGraph

    MR->>GR: can_run(forward_batch)?
    GR-->>MR: True
    MR->>GR: replay(forward_batch)
    GR->>Buf: replay_prepare: 将真实数据拷贝到静态缓冲区
    GR->>GR: 二分查找最近的已捕获 batch_size
    GR->>Graph: replay()

    alt BreakableCUDAGraph
        loop 遍历每个 segment i
            Graph->>Graph: segments[i].replay()
            Graph->>Graph: break_fns[i]() — eager 函数
        end
    else 单个 CUDAGraph
        Graph->>Graph: CUDAGraph.replay()
    end

    GR-->>MR: 返回 output_buffers[key]，按实际 batch_size 截取
```

### 6B. Extend 重放 — BCG

```mermaid
sequenceDiagram
    participant MR as ModelRunner.forward_extend
    participant BCR as BreakableCudaGraphRunner
    participant Buf as PrefillInputBuffers
    participant BGraph as BreakableCUDAGraph

    MR->>BCR: can_run(forward_batch)?
    BCR-->>MR: True
    MR->>BCR: replay(forward_batch)
    BCR->>BCR: 进入 enable_breakable_cuda_graph 上下文
    BCR->>Buf: replay_prepare: 拷贝到静态缓冲区
    BCR->>BCR: 更新 seq_lens、extend 字段等
    BCR->>BGraph: self.graphs[num_tokens].replay()

    loop 遍历每个 segment i
        BGraph->>BGraph: _segments[i].replay()
        opt i < len(break_fns)
            BGraph->>BGraph: _break_fns[i]() — 重放 eager 函数
        end
    end

    BCR-->>MR: 返回 output，按实际 token 数截取
```

---

## 7. BreakableCUDAGraph 内部工作机制

```mermaid
flowchart LR
    subgraph "BreakableCUDAGraphCapture（上下文管理器）"
        E[__enter__] --> B1[_begin_new_segment]
        B1 --> S1["CUDAGraph #1 .capture_begin(pool)"]
        S1 --> RUN["run_once() — 模型前向"]
        RUN --> HIT{遇到break?}
        HIT -->|是| END1[_end_current_segment]
        END1 --> S1E["CUDAGraph #1 .capture_end()"]
        S1E --> EAGER["eager 执行函数"]
        EAGER --> B2[_begin_new_segment]
        B2 --> S2["CUDAGraph #2 .capture_begin(同一个pool)"]
        S2 --> RUN2["继续模型前向"]
        RUN2 --> HIT2{又一个break?}
        HIT2 -->|是| END2["...重复..."]
        HIT2 -->|否| EXIT[__exit__]
        EXIT --> LAST["_end_current_segment: CUDAGraph #N .capture_end()"]
    end

    subgraph "产物：BreakableCUDAGraph"
        SEGS["_segments: [Graph#1, Graph#2, ..., Graph#N]"]
        FNS["_break_fns: [replay_fn_1, ..., replay_fn_(N-1)]"]
    end

    LAST --> SEGS
    EAGER -.-> FNS
```

### BreakableCUDAGraph 重放

```mermaid
flowchart TD
    START["replay()"] --> S1["segments[0].replay()"]
    S1 --> F1["break_fns[0]()"]
    F1 --> S2["segments[1].replay()"]
    S2 --> F2["break_fns[1]()"]
    F2 --> DOTS["..."]
    DOTS --> SN["segments[N-1].replay()"]
    SN --> DONE["结束"]

    style F1 fill:#f9f,stroke:#333
    style F2 fill:#f9f,stroke:#333
```

紫色节点 = 在 GPU graph 重放之间执行的 eager Python 函数。

---

## 8. 关键设计点

### 8.1 共享内存池

`BreakableCUDAGraph` 中所有段共享同一个 CUDA 内存池（`MempoolId_t`）。
- 段 N 分配的中间张量可以被段 N+1 复用
- 只要任意段的 CUDAGraph 存活，内存池就保持锁定（通过 `use_count`）
- `weak_ref_tensor` 视图在多次重放间保持有效

### 8.2 Stream 捕获 Hook

BCG 在捕获期间全局 hook `torch.cuda.Stream.wait_stream`，追踪 side stream 的
fork/join。在结束每个段之前（`_end_current_segment`），自动 join 所有已 fork 但
未 rejoin 的 stream，因为 `capture_end()` 在有参与捕获的 side stream 时会失败。

### 8.3 Weak-Ref 优化

`_weak_ref_if_tensor()` 对捕获的参数和输出创建弱引用视图，避免 Python 引用计数
阻止中间张量释放。存储生命周期由共享内存池的 `use_count` 管理。

### 8.4 方法借用（BreakableCudaGraphRunner）

`BreakableCudaGraphRunner` **不继承** `PiecewiseCudaGraphRunner`，而是通过
方法绑定借用 `replay_prepare`：

```python
replay_prepare = PiecewiseCudaGraphRunner.replay_prepare
```

避免深层继承，同时共享缓冲区准备逻辑。

### 8.5 Decode + BCG 组合

Decode 的 `CudaGraphRunner` 在设置 `SGLANG_USE_BREAKABLE_CUDA_GRAPH` 环境变量
时也可以使用 `BreakableCUDAGraph`。此时 `_create_device_graph()` 返回
`BreakableCUDAGraph` 而非 `torch.cuda.CUDAGraph`。

---

## 9. 文件索引

| 文件 | 包含的类/函数 |
|------|-------------|
| `model_executor/cuda_graph_runner.py` | `CudaGraphRunner`, `DecodeInputBuffers`, `DeepEPCudaGraphRunnerAdapter` |
| `model_executor/cpu_graph_runner.py` | `CPUGraphRunner` |
| `model_executor/piecewise_cuda_graph_runner.py` | `PiecewiseCudaGraphRunner`, `PrefillInputBuffers` |
| `model_executor/breakable_cuda_graph_runner.py` | `BreakableCudaGraphRunner` |
| `model_executor/model_runner.py:2591` | `init_device_graphs()` — 创建 decode runner |
| `model_executor/model_runner.py:2643` | `init_piecewise_cuda_graphs()` — 创建 extend runner |
| `breakable_cuda_graph/breakable_cuda_graph.py` | `BreakableCUDAGraph`, `BreakableCUDAGraphCapture`, `eager_on_graph()`, `break_graph()` |
| `breakable_cuda_graph/context.py` | `enable_breakable_cuda_graph()`, `is_in_breakable_cuda_graph()` |
| `breakable_cuda_graph/cuda_utils.py` | CUDA 运行时绑定辅助工具 |
| `breakable_cuda_graph/npu_utils.py` | NPU（昇腾 ACL）捕获状态检测 |
| `hardware_backend/npu/graph_runner/npu_graph_runner.py` | `NPUGraphRunner` |
| `compilation/piecewise_context_manager.py` | `enable_piecewise_cuda_graph()`, `ForwardContext` |
| `model_executor/input_buffers.py` | `ForwardInputBuffers`（基类） |
