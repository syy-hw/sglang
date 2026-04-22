# PCG vs BCG 架构对比深度研究报告

*生成日期: 2026-04-21 | 代码源: sglang main branch | 外部源: 15+ | 置信度: High*

---

## Executive Summary

sglang 项目中存在两种 CUDA Graph 加速策略：**Piecewise CUDA Graph (PCG)** 和 **Breakable CUDA Graph (BCG)**。PCG 采用编译时 FX 图分割策略，在注册的 split points 处将模型切分为多个子图，每个子图独立捕获为 CUDA Graph，通过 padding + 二分查找处理动态形状。BCG 采用运行时流捕获策略，利用底层 CUDA Runtime C API（`cudaStreamBeginCapture/EndCapture`）在 `@eager_on_graph` 装饰器处动态插入断点，实现可中断的 CUDA Graph。

**关键发现**：
- PCG 具有良好的硬件抽象，已成功适配华为 Ascend NPU（`NPUPiecewiseBackend`）
- BCG 依赖 `cuda-python` 包的低级 CUDA C API，**无法移植到 NPU**
- Ascend NPU 的 `torch.npu.NPUGraph` API 与 `torch.cuda.CUDAGraph` 几乎等价，ACL Graph 设计与 CUDA Graph 高度一致
- PyTorch 社区正在推进 `torch.accelerator.Graph` 统一抽象（RFC #158827, PR #171285），Ascend 已参与

---

## 1. 代码实现对比

### 1.1 核心文件与类层次

#### PCG (Piecewise CUDA Graph)

```
文件结构:
├── python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py
│   └── PiecewiseCudaGraphRunner          # 主 Runner，管理捕获/回放
├── python/sglang/srt/compilation/cuda_piecewise_backend.py
│   └── CUDAPiecewiseBackend              # CUDA 后端，每个子图的捕获/回放
├── python/sglang/srt/compilation/npu_piecewise_backend.py
│   └── NPUPiecewiseBackend               # NPU 后端，继承 CUDA 后端
├── python/sglang/srt/compilation/backend.py
│   ├── SGLangBackend                     # 编译后端，FX 图分割
│   └── PiecewiseCompileInterpreter       # FX 解释器，替换子模块
└── python/sglang/srt/compilation/piecewise_context_manager.py
    └── 全局上下文管理                     # PCG 模式标志追踪
```

#### BCG (Breakable CUDA Graph)

```
文件结构:
├── python/sglang/srt/model_executor/breakable_cuda_graph/
│   ├── breakable_cuda_graph.py
│   │   ├── BreakableCUDAGraph            # 继承 torch.cuda.CUDAGraph
│   │   ├── BreakableCUDAGraphCapture     # 上下文管理器
│   │   ├── GraphBreakInfo (NamedTuple)   # 断点信息
│   │   ├── eager_on_graph()              # 装饰器，标记断点
│   │   └── break_graph()                # 显式断点插入
│   ├── cuda_utils.py                     # CUDA Runtime C 绑定
│   └── __init__.py
└── python/sglang/srt/model_executor/cuda_graph_runner.py
    └── CudaGraphRunner                   # 集成 BCG 的主 Runner
```

### 1.2 核心数据结构

#### PCG 关键数据结构

```python
# 预填充输入缓冲区
@dataclass
class PrefillInputBuffers:
    input_ids: torch.Tensor              # Token IDs
    out_cache_loc: torch.Tensor          # KV cache 位置
    positions: torch.Tensor              # 位置编码
    input_embeds: Optional[torch.Tensor] # 多模态嵌入
    mrope_positions: Optional[torch.Tensor]  # M-RoPE 位置
    ...

# 捕获大小条目
@dataclass
class ConcreteSizeEntry:
    runtime_shape: int                    # 该大小对应的 token 数
    need_to_compile: bool                 # 是否需要编译
    use_cudagraph: bool                   # 是否使用 CUDA Graph
    compiled: bool = False                # 编译状态
    runnable: Callable = None             # 编译后的函数
    num_finished_warmup: int = 0          # 预热次数
    cudagraph: Optional[torch.cuda.CUDAGraph] = None  # 捕获的图
    output: Optional[Any] = None          # 输出引用
```

#### BCG 关键数据结构

```python
# 断点信息
class GraphBreakInfo(NamedTuple):
    func: Callable       # 断点处需要 eager 执行的函数
    output: Any          # 输出缓冲区引用
    graph_handle: Any    # CUDA graph 句柄

# 上下文变量（运行时追踪）
_captured_graphs_var: ContextVar    # 存储断点
_current_stream_var: ContextVar     # 当前捕获流
_forked_streams_var: ContextVar     # 分支流追踪
```

### 1.3 调用链对比

#### PCG 调用链

```
ModelRunner.forward_extend()
    │
    ├─► PiecewiseCudaGraphRunner.can_run()
    │       └─ 检查: 无 input_embeds, 正确的 hidden_mode, token 数在范围内
    │
    ├─► PiecewiseCudaGraphRunner.replay()
    │       │
    │       ├─► replay_prepare()
    │       │       ├─ bisect_left() 找到最近的捕获大小
    │       │       ├─ 零填充输入到捕获大小
    │       │       └─ 设置 forward batch
    │       │
    │       ├─► set_forward_context()
    │       │
    │       └─► model.forward()  ──► 触发编译后的子图
    │               │
    │               ├─► CUDAPiecewiseBackend.__call__()
    │               │       ├─ 选择合适大小的 compiled runnable
    │               │       └─ cuda_graph.replay()
    │               │
    │               └─► [Split Point 处] Eager 执行 MoE dispatch
    │
    └─► 切片输出到实际 token 数
```

#### BCG 调用链

```
ModelRunner.forward_decode() / forward_extend()
    │
    ├─► CudaGraphRunner.can_run()
    │       └─ 检查: batch size 在捕获范围内
    │
    ├─► CudaGraphRunner.replay()
    │       │
    │       ├─► 准备 DecodeInputBuffers
    │       ├─► 选择捕获大小 (二分查找)
    │       │
    │       └─► BreakableCUDAGraph.replay()
    │               │
    │               ├─► Segment 1: cuda_graph_launch(handle_1)
    │               ├─► Eager: GraphBreakInfo.func()  ── 执行断点函数
    │               ├─► Segment 2: cuda_graph_launch(handle_2)
    │               ├─► Eager: GraphBreakInfo.func()
    │               ├─► ...
    │               └─► Segment N: cuda_graph_launch(handle_N)
    │
    └─► _copy_output() ── 写回输出
```

---

## 2. 架构对比

### 2.1 综合对比表格

| 维度 | PCG (Piecewise CUDA Graph) | BCG (Breakable CUDA Graph) |
|------|---------------------------|---------------------------|
| **分段策略** | 编译时 FX 图分割，在注册的 split points 切分 | 运行时流捕获，在 `@eager_on_graph` 处动态断开 |
| **分割粒度** | 模型层级（如 MoE dispatch 前后） | 算子层级（任意 CUDA 可捕获操作之间） |
| **编译方式** | `torch.compile` + FX 图分割 + `split_module()` | `cudaStreamBeginCapture/EndCapture` 流捕获 |
| **底层依赖** | `torch.cuda.CUDAGraph` (PyTorch 高级 API) | `cuda-python` 包 (CUDA Runtime C API) |
| **内存管理** | 全局共享 graph pool，反向捕获复用内存 | 全局 graph pool + 跨段输出拷贝 |
| **动态形状** | 预分配多个大小缓冲区 + padding + 二分查找 | 预分配固定大小缓冲区 + padding + 二分查找 |
| **条件/分支** | 仅在 split points 处切换 eager/compiled | `@eager_on_graph` 装饰器标记任意断点 |
| **调试能力** | 有限（编译后子图不可逐步调试） | 优秀（可逐步执行，`--debug-cuda-graph`） |
| **NPU 兼容性** | 已适配（`NPUPiecewiseBackend`） | 不兼容（缺少底层 C API） |
| **启用方式** | 默认启用（满足条件时自动启用） | `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1` |
| **适用场景** | Prefill/Extend 阶段 | Decode 阶段 |
| **平台支持** | CUDA + Ascend NPU | 仅 NVIDIA CUDA |

### 2.2 捕获流程对比

#### PCG 三阶段流程

```
┌─────────────────────────────────────────────────────────────┐
│                    PCG Capture Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: COMPILE                                           │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Dummy     │───►│ torch.compile │───►│ FX Graph     │     │
│  │ Forward   │    │ + SGLangBackend│   │ Split        │     │
│  └───────────┘    └──────────────┘    └──────┬───────┘     │
│                                               │             │
│                    ┌──────────────────────────┤             │
│                    ▼                          ▼             │
│              SubGraph 0                  SubGraph 1         │
│            (pre-MoE ops)              (post-MoE ops)       │
│                    │                          │             │
│  Phase 2: CAPTURE  │                          │             │
│                    ▼                          ▼             │
│  For each capture_size (reverse order: large → small):     │
│    ┌──────────────────────────────────────────────┐        │
│    │ 1. Warmup: run forward twice                  │        │
│    │ 2. Capture: torch.cuda.graph(g, pool=pool)    │        │
│    │    ├─ SubGraph 0 → CUDAGraph_0[size]          │        │
│    │    ├─ [Eager: MoE dispatch]                    │        │
│    │    └─ SubGraph 1 → CUDAGraph_1[size]          │        │
│    └──────────────────────────────────────────────┘        │
│                                                             │
│  Phase 3: REPLAY                                            │
│    ┌──────────────────────────────────────────────┐        │
│    │ 1. bisect_left() → find nearest size          │        │
│    │ 2. Pad inputs → static buffers                │        │
│    │ 3. CUDAGraph_0.replay() → Eager MoE →         │        │
│    │    CUDAGraph_1.replay()                        │        │
│    │ 4. Slice outputs to actual token count         │        │
│    └──────────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### BCG 单阶段捕获流程

```
┌─────────────────────────────────────────────────────────────┐
│                    BCG Capture Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Single Phase: CAPTURE (stream capture)                     │
│                                                             │
│  cudaStreamBeginCapture(stream)                             │
│       │                                                     │
│       ├─ Graphable ops (captured as segment 1)              │
│       │                                                     │
│       ├─ @eager_on_graph encountered:                       │
│       │   ├─ cudaStreamEndCapture() → graph_handle_1        │
│       │   ├─ cudaGraphInstantiateWithFlags(handle_1)        │
│       │   ├─ Run function eagerly                          │
│       │   └─ cudaStreamBeginCapture() → start segment 2    │
│       │                                                     │
│       ├─ Graphable ops (captured as segment 2)              │
│       │                                                     │
│       ├─ [Another @eager_on_graph]                          │
│       │   ├─ cudaStreamEndCapture() → graph_handle_2        │
│       │   ├─ cudaGraphInstantiateWithFlags(handle_2)        │
│       │   ├─ Run function eagerly                          │
│       │   └─ cudaStreamBeginCapture() → start segment 3    │
│       │                                                     │
│       └─ Final graphable ops                                │
│                                                             │
│  cudaStreamEndCapture() → graph_handle_N                    │
│  cudaGraphInstantiateWithFlags(handle_N)                    │
│                                                             │
│  REPLAY:                                                    │
│    for each segment:                                        │
│      cudaGraphLaunch(handle_i)                              │
│      eager_func_i()                                         │
│    cudaGraphLaunch(handle_N)  # final segment               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 内存管理对比

```
PCG Memory Layout:
┌────────────────────────────────────────────┐
│           Global Graph Memory Pool          │
│  ┌────────────────────────────────────────┐ │
│  │ Capture Size 4096 (captured first)     │ │
│  │ ┌──────────┐  ┌──────────┐            │ │
│  │ │SubGraph 0│  │SubGraph 1│            │ │
│  │ └──────────┘  └──────────┘            │ │
│  ├────────────────────────────────────────┤ │
│  │ Capture Size 2048 (reuses pool)        │ │
│  │ ┌──────────┐  ┌──────────┐            │ │
│  │ │SubGraph 0│  │SubGraph 1│            │ │
│  │ └──────────┘  └──────────┘            │ │
│  ├────────────────────────────────────────┤ │
│  │ ... (smaller sizes)                    │ │
│  └────────────────────────────────────────┘ │
│  Weak refs for outputs → immediate free     │
└────────────────────────────────────────────┘

BCG Memory Layout:
┌────────────────────────────────────────────┐
│           Global Graph Memory Pool          │
│  ┌────────────────────────────────────────┐ │
│  │ Batch Size N (captured first)          │ │
│  │ ┌─────┐ ┌─────┐ ... ┌─────┐            │ │
│  │ │Seg 1│ │Seg 2│     │Seg N│            │ │
│  │ └─────┘ └─────┘     └─────┘            │ │
│  │  ↑ break   ↑ break        ↑            │ │
│  │  output    output      (no break)      │ │
│  │  copy-back copy-back                   │ │
│  └────────────────────────────────────────┘ │
└────────────────────────────────────────────┘
```

---

## 3. Ascend NPU 适配性分析

### 3.1 NPU Piecewise Backend 实现

sglang 的 NPU 适配采用 **仅 PCG** 策略：

```
CUDA → NPU API 映射表:
┌──────────────────────────┬──────────────────────────────┐
│ CUDA API                 │ NPU 等效 API                 │
├──────────────────────────┼──────────────────────────────┤
│ torch.cuda.CUDAGraph()   │ torch.npu.NPUGraph()        │
│ torch.cuda.graph(g, ...) │ torch.npu.graph(g, ...)     │
│ torch.cuda.Stream()      │ torch.npu.Stream()          │
│ torch.cuda.empty_cache() │ torch.npu.empty_cache()     │
│ torch.cuda.graph_pool    │ torch.npu.graph_pool        │
│ .replay()                │ .replay()                    │
│ .capture_begin()         │ .capture_begin()             │
│ .capture_end()           │ .capture_end()               │
└──────────────────────────┴──────────────────────────────┘

NPUPiecewiseBackend 继承关系:
    CUDAPiecewiseBackend
         │
         ├── __init__()     → 替换 torch.cuda → torch.npu
         ├── capture()      → 使用 torch.npu.NPUGraph
         └── replay()       → 使用 NPUGraph.replay()
              │
              └── NPUPiecewiseBackend
                    └── 最小化改动，仅替换 API 调用
```

### 3.2 BCG 在 NPU 上不可用的原因

```
BCG 依赖的底层 CUDA Runtime C API:
┌──────────────────────────────────┬─────────────────────────────┐
│ CUDA Runtime API (cuda-python)   │ NPU 等效                     │
├──────────────────────────────────┼─────────────────────────────┤
│ cudaStreamBeginCapture()         │ ❌ 无等效公开 API            │
│ cudaStreamEndCapture()           │ ❌ 无等效公开 API            │
│ cudaGraphInstantiateWithFlags()  │ ❌ 无等效公开 API            │
│ cudaGraphLaunch()                │ ❌ 无等效公开 API            │
│ cudaGraphDestroy()               │ ❌ 无等效公开 API            │
└──────────────────────────────────┴─────────────────────────────┘

原因分析:
1. BCG 通过 cuda-python 包直接调用 CUDA Driver/Runtime C API
2. 这些 API 没有被 torch_npu 暴露为 Python 接口
3. ACL Graph 内部使用 aclmdlRICaptureMode，但未暴露底层控制
4. 要支持 BCG，需要华为在 torch_npu 中实现等效的低级流捕获 API
```

### 3.3 Ascend NPU 的 CUDA Graph 等效 API 支持情况

基于网络调研的发现：

#### 3.3.1 torch.npu.NPUGraph (已支持)

```python
# torch_npu/npu/graphs.py 中的实现
class NPUGraph(torch_npu._C._NPUGraph):
    def capture_begin(self, pool=None, capture_error_mode="global"):
        """开始捕获 NPU 工作流 — 等效于 CUDAGraph.capture_begin()"""

    def capture_end(self):
        """结束捕获 — 等效于 CUDAGraph.capture_end()"""

    def replay(self):
        """重放捕获的图 — 等效于 CUDAGraph.replay()"""

    def reset(self):
        """重置图 — 等效于 CUDAGraph.reset()"""

class graph:
    """上下文管理器，等效于 torch.cuda.graph()"""
```

关键特性：
- API 与 `torch.cuda.CUDAGraph` 几乎 1:1 对应
- 支持 `capture_error_mode` 参数（global/thread_local/relaxed）
- 支持图间内存池共享
- 来源: [torch_npu graphs.py](https://gitee.com/ascend/pytorch/blob/3d25eb5c5a0e932bddd413c52553fba7659ca1f5/torch_npu/npu/graphs.py)

#### 3.3.2 ACL Graph 底层能力

- **aclgraph** (Ascend Computing Library Graph) 是华为 Ascend 的图执行方案
- 设计与 CUDAGraph 几乎一致（来自 PyTorch RFC #158827 的 Ascend 开发者确认）
- 集成在 torch_npu 和 vllm-ascend 中
- 来源: [PyTorch RFC #158827](https://github.com/pytorch/pytorch/issues/158827)

#### 3.3.3 vllm-ascend 的 ACLGraph 实践

```python
# vllm-ascend 的 ACLGraphWrapper
class ACLGraphWrapper:
    """使用 torch.npu.NPUGraph 进行图捕获和回放"""
    aclgraph: torch.npu.NPUGraph | None = None

    # 支持 FULL 和 PIECEWISE 两种模式
    # FULL: 整图捕获（使用 torch.npu.graph_task_update_begin/end）
    # PIECEWISE: 分段捕获（与 sglang PCG 类似）

    # 限制: 最多 1800 个图（受 ACL stream 限制）
```
- 来源: [vllm-ascend acl_graph.py](https://github.com/vllm-project/vllm-ascend/blob/44ef9a36/vllm_ascend/compilation/acl_graph.py)

#### 3.3.4 torchair npugraph_ex 后端

华为 torchair 项目提供了 `npugraph_ex` 编译后端：

```python
# 通过 torch.compile 使用
model = torch.compile(model, backend="npugraph_ex")

# 功能:
# - FX 图优化 Pass
# - 算子融合 Pass
# - aclgraph 间内存复用
# - 静态 Kernel 编译
# - 集合通信入图
# - 多流表达
```
- 来源: [torchair npugraph_ex](https://github.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/quick_start.md)
- 约束: 与 `torch.cuda.CUDAGraph` 原生接口一致（不支持 stream sync、动态控制流等）

#### 3.3.5 PyTorch 统一 Graph API 进展

```
PyTorch 社区正在推进 torch.accelerator.Graph:
┌─────────────────────────────────────────────────────────┐
│ RFC: pytorch/pytorch#158827 (Graph Generalization)       │
│ PR:  pytorch/pytorch#171285 (torch.accelerator.Graph)    │
│                                                          │
│ 目标:                                                    │
│   torch.accelerator.Graph  → 统一前端接口               │
│   ├── CUDA 后端                                           │
│   ├── XPU 后端                                            │
│   └── NPU (Ascend) 后端 ← 华为已参与讨论                 │
│                                                          │
│ 状态: PR #171285 已提交 (2025-12-25), 活跃开发中         │
│ Ascend 开发者确认: aclgraph 设计与 CUDAGraph 几乎一致     │
└─────────────────────────────────────────────────────────┘
```

### 3.4 NPU 适配性总结表

| 能力维度 | PCG on NPU | BCG on NPU | 备注 |
|---------|------------|------------|------|
| 高级 Graph API | ✅ `torch.npu.NPUGraph` | ❌ 无等效 | PCG 仅依赖高级 API |
| 流捕获 (Stream Capture) | ✅ `torch.npu.graph()` | ❌ 无 `cudaStreamBeginCapture` | BCG 需要低级 C API |
| 图实例化/启动 | ✅ 内置 `replay()` | ❌ 无 `cudaGraphLaunch` | BCG 手动管理图句柄 |
| FX 图分割 | ✅ 已适配 | N/A | PCG 核心能力 |
| 内存池共享 | ✅ 支持 | ❌ | NPU 图池行为一致 |
| 动态形状处理 | ✅ padding + 二分查找 | N/A | PCG 方案可直接复用 |
| vllm-ascend 验证 | ✅ 已在生产环境使用 | N/A | ACLGraphWrapper 参考实现 |

---

## 4. 架构图

### 4.1 PCG 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      sglang Serving Layer                        │
│                                                                 │
│  ┌────────────────────┐        ┌──────────────────────┐         │
│  │  ModelRunner        │        │ ServerArgs            │         │
│  │  ├─ forward_extend()│───┐    │ ├─ use_piecewise_cg   │         │
│  │  └─ forward_decode()│   │    │ └─ cuda_graph_max_bs  │         │
│  └────────────────────┘   │    └──────────────────────┘         │
│                            │                                     │
│  ┌─────────────────────────▼──────────────────────────┐         │
│  │      PiecewiseCudaGraphRunner                       │         │
│  │  ┌───────────────────────────────────────────────┐  │         │
│  │  │ init_piecewise_cuda_graphs()                   │  │         │
│  │  │  ├─ 收集 attention/MoE 层                      │  │         │
│  │  │  └─ 创建 PiecewiseCudaGraphRunner              │  │         │
│  │  ├───────────────────────────────────────────────┤  │         │
│  │  │ capture()                                      │  │         │
│  │  │  ├─ Phase 1: torch.compile + SGLangBackend    │  │         │
│  │  │  │   └─ split_graph() at MoE split points     │  │         │
│  │  │  ├─ Phase 2: Warmup (2 runs per size)         │  │         │
│  │  │  └─ Phase 3: torch.{cuda|npu}.graph() capture │  │         │
│  │  ├───────────────────────────────────────────────┤  │         │
│  │  │ replay()                                       │  │         │
│  │  │  ├─ can_run() → 检查条件                      │  │         │
│  │  │  ├─ replay_prepare() → padding + 二分查找      │  │         │
│  │  │  └─ model.forward() → 子图回放                │  │         │
│  │  └───────────────────────────────────────────────┘  │         │
│  └─────────────────────────────────────────────────────┘         │
│                            │                                     │
│           ┌────────────────┴────────────────┐                    │
│           ▼                                  ▼                    │
│  ┌─────────────────┐               ┌──────────────────┐         │
│  │ CUDAPiecewise    │               │ NPUPiecewise      │         │
│  │ Backend          │               │ Backend           │         │
│  │ ├─ torch.cuda.  │               │ ├─ torch.npu.    │         │
│  │ │  CUDAGraph     │               │ │  NPUGraph       │         │
│  │ └─ capture on   │               │ └─ capture on    │         │
│  │    NVIDIA GPU    │               │    Ascend NPU     │         │
│  └─────────────────┘               └──────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 BCG 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      sglang Serving Layer                        │
│                                                                 │
│  ┌────────────────────┐                                         │
│  │  ModelRunner        │                                         │
│  │  └─ forward_decode()│──────────────────┐                      │
│  └────────────────────┘                   │                      │
│                                            ▼                      │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  CudaGraphRunner (with BCG support)                   │        │
│  │  ┌──────────────────────────────────────────────────┐│        │
│  │  │  BreakableCUDAGraph (extends torch.cuda.CUDAGraph)│        │
│  │  │                                                    │        │
│  │  │  Capture:                                          │        │
│  │  │  ┌──────────┐  ┌──────────┐       ┌──────────┐   │        │
│  │  │  │ Segment 1│──│ eager()  │──...──│ Segment N│   │        │
│  │  │  │ (captured)│  │(break pt)│       │ (captured)│   │        │
│  │  │  └──────────┘  └──────────┘       └──────────┘   │        │
│  │  │       │                                  │         │        │
│  │  │       ▼                                  ▼         │        │
│  │  │  cudaGraphLaunch()               cudaGraphLaunch() │        │
│  │  ├──────────────────────────────────────────────────┤│        │
│  │  │  Replay:                                          │        │
│  │  │  for break in breaks:                             │        │
│  │  │    cudaGraphLaunch(segment_i)                     │        │
│  │  │    break.func()  # eager execution                │        │
│  │  │  cudaGraphLaunch(final_segment)                   │        │
│  │  └──────────────────────────────────────────────────┘│        │
│  └──────────────────────────────────────────────────────┘        │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────┐                                    │
│  │  cuda_utils.py            │  ← cuda-python 绑定              │
│  │  ├─ cudaStreamBeginCapture│                                    │
│  │  ├─ cudaStreamEndCapture  │                                    │
│  │  ├─ cudaGraphInstantiate  │                                    │
│  │  └─ cudaGraphLaunch       │                                    │
│  └──────────────────────────┘                                    │
│       │                                                          │
│       ▼  仅 NVIDIA CUDA                                         │
│  ┌──────────────────────────┐                                    │
│  │  NVIDIA GPU Driver        │                                    │
│  │  (CUDA Runtime API)       │                                    │
│  └──────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 硬件抽象层架构

```
                    ┌────────────────────┐
                    │   sglang Runtime    │
                    │ (ModelRunner, etc.) │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Graph Runner Layer │
                    │ ├─ CudaGraphRunner  │ ← BCG 在此层
                    │ └─ PiecewiseCGRunner│ ← PCG 在此层
                    └─────────┬──────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼───────┐ ┌────▼─────────┐ ┌──▼──────────────┐
     │  CUDA Backend   │ │ NPU Backend  │ │ (Other Backend) │
     │                 │ │              │ │                  │
     │ torch.cuda.     │ │ torch.npu.   │ │ ...              │
     │ ├─ CUDAGraph    │ │ ├─ NPUGraph  │ │                  │
     │ ├─ Stream       │ │ ├─ Stream    │ │                  │
     │ └─ graph()      │ │ └─ graph()   │ │                  │
     │                 │ │              │ │                  │
     │ cuda-python:    │ │ ❌ 无等效:   │ │                  │
     │ ├─ StreamCapture│ │ ├─ BeginCap  │ │                  │
     │ ├─ EndCapture   │ │ ├─ EndCap    │ │                  │
     │ └─ GraphLaunch  │ │ └─ GraphLnc  │ │                  │
     └────────┬────────┘ └──────┬───────┘ └──────────────────┘
              │                  │
     ┌────────▼────────┐ ┌──────▼───────────┐
     │  NVIDIA CUDA    │ │  Huawei Ascend   │
     │  Runtime API    │ │  ACL (CANN)      │
     └─────────────────┘ └──────────────────┘
```

---

## 5. 关键发现与建议

### 5.1 关键发现

1. **PCG 是 NPU 的唯一可行路径**：PCG 仅依赖 PyTorch 高级 API（`torch.npu.NPUGraph`），NPU 已完整支持这些 API

2. **BCG 的底层 CUDA C API 依赖是根本障碍**：BCG 通过 `cuda-python` 直接调用 `cudaStreamBeginCapture` 等函数，这些函数在 torch_npu 中没有公开等效接口

3. **ACL Graph 设计与 CUDA Graph 高度一致**：华为开发者在 PyTorch RFC 中明确表示 "aclgraph is almost identical to CUDAGraph"

4. **vllm-ascend 已验证 NPU 图模式可行性**：vllm-ascend 的 `ACLGraphWrapper` 成功使用了 `torch.npu.NPUGraph` 实现 FULL 和 PIECEWISE 两种模式

5. **PyTorch 社区正在统一 Graph API**：`torch.accelerator.Graph` (PR #171285) 将为所有加速器提供统一接口

### 5.2 NPU 适配建议

| 优先级 | 建议 | 理由 |
|--------|------|------|
| P0 | 继续使用 PCG 策略 | 已验证可行，API 完整 |
| P1 | 关注 `torch.accelerator.Graph` 进展 | 统一接口将简化硬件抽象 |
| P2 | 评估 torchair `npugraph_ex` 后端 | 可提供额外 FX 图优化 |
| P3 | 向华为请求暴露低级流捕获 API | 如需 BCG 能力，需华为配合 |

---

## Sources

1. [torch_npu/npu/graphs.py](https://gitee.com/ascend/pytorch/blob/3d25eb5c5a0e932bddd413c52553fba7659ca1f5/torch_npu/npu/graphs.py) — NPUGraph 类实现
2. [CUDA to NPU Compatibility Layer | DeepWiki](https://deepwiki.com/Ascend/pytorch/7.1-cuda-to-npu-compatibility-layer) — CUDA→NPU API 映射
3. [PyTorch RFC #158827: Graph Generalization](https://github.com/pytorch/pytorch/issues/158827) — 统一 Graph API 提案，含 Ascend 开发者确认
4. [PyTorch PR #171285: torch.accelerator.Graph](https://github.com/pytorch/pytorch/pull/171285) — 统一前端 Graph 接口实现
5. [PyTorch Issue #166205: Graph Capture for PrivateUse1](https://github.com/pytorch/pytorch/issues/166205) — 第三方加速器图捕获需求
6. [vllm-ascend: ACLGraph Implementation](https://github.com/vllm-project/vllm-ascend/blob/44ef9a36/vllm_ascend/compilation/acl_graph.py) — vllm-ascend 图模式实现
7. [vllm-ascend: Graph Execution Documentation](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/Design_Documents/ACL_Graph.html) — ACL Graph 工作原理
8. [torchair: npugraph_ex Quick Start](https://github.com/Ascend/torchair/blob/master/docs/zh/npugraph_ex/quick_start.md) — npugraph_ex 后端使用指南
9. [vllm-ascend RFC #4715: npugraph_ex backend](https://github.com/vllm-project/vllm-ascend/issues/4715) — npugraph_ex 后端提案
10. [DeepWiki: vllm-ascend Graph Execution](https://deepwiki.com/vllm-project/vllm-ascend/5-graph-execution-and-optimization) — vllm-ascend 图执行总览
11. [Ascend/pytorch GitHub](https://github.com/Ascend/PyTorch) — torch_npu 官方仓库
12. sglang 源码: `piecewise_cuda_graph_runner.py`, `cuda_piecewise_backend.py`, `npu_piecewise_backend.py`, `breakable_cuda_graph.py`, `cuda_utils.py`

## Methodology

- **代码分析**: 使用 Explore agent 对 sglang 代码库中 PCG/BCG/NPU 相关的 12+ 文件进行深度分析
- **网络搜索**: 使用 Exa 搜索引擎对 torch_npu、ACL Graph、vllm-ascend 等关键词进行了 3 组搜索，获取 24 条结果
- **源码阅读**: 深度阅读了 torch_npu graphs.py、vllm-ascend acl_graph.py、torchair npugraph_ex 文档
- **子问题**: PCG 实现、BCG 实现、NPU 适配、Ascend API 支持状况


 ---
  1. BCG 是在同一个 stream 里分段捕获？PCG 不是？

  是的，你的理解正确。

  BCG — 同一个 stream 上连续 begin/end capture 分段：
  - capture_begin() → 先让 torch.cuda.graph 开始一个 dummy capture，然后 _end_capture_segment() 结束它，再 _begin_capture_segment() 开始真正的第一段
  - 每个 @eager_on_graph 装饰的函数触发时：_end_capture_segment() → 执行 eager 函数 → _begin_capture_segment()
  - capture_end() → _end_capture_segment() 最后一段，实例化所有分段
  - 关键：全在同一个 stream 上，通过 rt.cudaStreamBeginCapture/EndCapture 反复切分

  PCG — 不是同一个 stream 上连续分段：
  - PCG 用 torch.fx.passes.split_module.split_module() 在编译时把 FX graph 按注册的 split 点拆成多个子图
  - 每个 CUDAPiecewiseBackend 实例管理一个子图，独立调用 torch.cuda.graph(cudagraph, pool=..., stream=stream) 做 capture
  - 所有子图共享同一个 capture stream（get_pcg_capture_stream()），但每个子图是独立的 torch.cuda.CUDAGraph 对象，各自独立 capture 和 replay

  2. BCG 依赖 cuda-python 的低级 CUDA C API — 是不是最大 gap？

  是的，这是 NPU 移植的最大 gap。

  BCG 直接依赖的 cuda-python API：

  ┌──────────────────────────────────┬──────────────────┬──────────────────┐
  │               API                │       用途       │    NPU 等价物    │
  ├──────────────────────────────────┼──────────────────┼──────────────────┤
  │ rt.cudaStreamBeginCapture        │ 开始 stream 捕获 │ torch_npu 未暴露 │
  ├──────────────────────────────────┼──────────────────┼──────────────────┤
  │ rt.cudaStreamEndCapture          │ 结束 stream 捕获 │ torch_npu 未暴露 │
  ├──────────────────────────────────┼──────────────────┼──────────────────┤
  │ rt.cudaGraphInstantiateWithFlags │ 实例化 graph     │ torch_npu 未暴露 │
  ├──────────────────────────────────┼──────────────────┼──────────────────┤
  │ rt.cudaGraphLaunch               │ 回放 graph       │ torch_npu 未暴露 │
  ├──────────────────────────────────┼──────────────────┼──────────────────┤
  │ rt.cudaGraphDestroy              │ 销毁 graph       │ torch_npu 未暴露 │
  ├──────────────────────────────────┼──────────────────┼──────────────────┤
  │ rt.cudaStreamGetCaptureInfo      │ 查询捕获状态     │ torch_npu 未暴露 │
  └──────────────────────────────────┴──────────────────┴──────────────────┘

  而 PCG 只依赖：
  - torch.cuda.CUDAGraph() / torch.cuda.graph() — NPU 有 torch.npu.NPUGraph() / torch.npu.graph()
  - torch.compile + FX graph — 与硬件无关

  所以 PCG 的 NPUPiecewiseBackend 只需把 torch.cuda.* 替换为 torch.npu.* 即可（仅 ~3 行改动），而 BCG 需要华为暴露 aclGraph 的 Python 低级绑定。

  3. 分段入图方式快速总结

  ┌─────────────────────────────────────────────────────────────────────┐
  │                    分段入图方式对比                                    │
  ├───────────────┬─────────────────────┬───────────────────────────────┤
  │               │  PCG (Piecewise)    │  BCG (Breakable)              │
  ├───────────────┼─────────────────────┼───────────────────────────────┤
  │ 分段时机       │ 编译时 (FX graph)    │ 运行时 (stream capture)       │
  │ 分段依据       │ 注册的 split 点      │ @eager_on_graph 装饰器        │
  │               │ (如 MoE dispatch)    │ + break_graph() 显式中断      │
  ├───────────────┼─────────────────────┼───────────────────────────────┤
  │ 捕获方式       │ 每个子图独立         │ 同一 stream 连续切分           │
  │               │ torch.cuda.graph()  │ BeginCapture/EndCapture 循环  │
  │ Graph 对象     │ N 个独立 CUDAGraph  │ N+1 个 graph_exec handle     │
  │               │ (每子图每shape一个)  │ (stream上连续分段实例化)       │
  ├───────────────┼─────────────────────┼───────────────────────────────┤
  │ 回放方式       │ 每个子图独立 replay  │ 顺序 replay:                  │
  │               │ cudagraph.replay()  │ graph_launch → eager_fn →     │
  │               │                     │ graph_launch → ... → 最后段   │
  ├───────────────┼─────────────────────┼───────────────────────────────┤
  │ API 层级       │ PyTorch 高级 API    │ CUDA Runtime C API            │
  │               │ (可移植到 NPU)       │ (cuda-python 绑定, 不可移植)  │
  ├───────────────┼─────────────────────┼───────────────────────────────┤
  │ 中间段输出处理  │ 子图间通过 FX 传参   │ eager 函数返回 output tensor  │
  │               │ 自动连接             │ 下一段可直接引用               │
  ├───────────────┼─────────────────────┼───────────────────────────────┤
  │ NPU 适配难度   │ 低 (~3行替换)       │ 高 (需华为暴露 ACL graph 绑定) │
  └───────────────┴─────────────────────┴───────────────────────────────┘

  一句话总结：PCG 是编译时拆图 + 每段独立 capture（torch.cuda.graph），BCG 是运行时在同一 stream 上连续 begin/end capture 分段（cuda-python C
  API）。两者的核心区别在于分段发生在编译时还是运行时，以及 API 是可移植的高级 PyTorch API 还是不可移植的低级 CUDA C API