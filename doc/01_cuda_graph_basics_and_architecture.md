# CUDA Graph 基础与 sglang 架构全景

> 生成日期: 2026-04-21
> 基于 commit: `c25f00630`
> 参考文档: [piecewise_cuda_graph.md](../docs/advanced_features/piecewise_cuda_graph.md), [breakable_cuda_graph.md](../docs/advanced_features/breakable_cuda_graph.md), [pcg_vs_bcg_analysis.md](../docs/advanced_features/pcg_vs_bcg_analysis.md)

---

## 目录

- [一、CUDA Graph 概念入门](#一cuda-graph-概念入门)
  - [1. 什么是 Kernel Launch Overhead](#1-什么是-kernel-launch-overhead)
  - [2. CUDA Graph 的核心思想：录制与回放](#2-cuda-graph-的核心思想录制与回放)
  - [3. 静态 Shape 与静态内存地址约束](#3-静态-shape-与静态内存地址约束)
  - [4. 10 行伪代码：最基本的 Capture-Replay 流程](#4-10-行伪代码最基本的-capture-replay-流程)
  - [5. CUDA Graph 的限制](#5-cuda-graph-的限制)
- [二、sglang CUDA Graph 架构全景](#二sglang-cuda-graph-架构全景)
  - [1. Server 启动时的初始化链路](#1-server-启动时的初始化链路)
  - [2. CudaGraphRunner 核心状态机](#2-cudagraphrunner-核心状态机)
  - [3. Decode 路径：标准 CUDA Graph](#3-decode-路径标准-cuda-graph)
  - [4. Extend 路径：为什么标准 CUDA Graph 不够用](#4-extend-路径为什么标准-cuda-graph-不够用)
  - [5. 整体架构图](#5-整体架构图)

---

## 一、CUDA Graph 概念入门

### 1. 什么是 Kernel Launch Overhead

**类比解释**：想象一个厨师（CPU）给助手（GPU）下达指令。每道菜要分 50 步，每一步厨师都要：
1. 写一张指令卡（参数打包）
2. 走到助手工位递卡（驱动调用，用户态→内核态切换）
3. 等助手接过去（命令入队）

如果每道工序只需 5 微秒执行，但传达指令也要 5-10 微秒，那**一半时间都花在了传达上**。这就是 kernel launch overhead。

**技术原理**：在 LLM 推理的 decode 阶段：
- batch size 通常很小（1-32）
- 每个 token 的 forward pass 包含**几十到上百个 kernel launch**
- 每个 kernel 的执行时间可能只有几微秒
- kernel launch overhead 本身也是 ~5-10 微秒
- 累计下来，CPU 端的 kernel launch overhead 可能占总延迟的 **30-50%**

### 2. CUDA Graph 的核心思想：录制与回放

**类比解释**：与其每步都传达指令，不如**把整套流程录成录像带**。下次做同样的菜，直接把录像带丢给助手，助手自己按录像操作。CPU 只需要做一件事：按下播放键。

**技术原理**：CUDA Graph 将整个 GPU 操作序列"预编译"为一个执行图：
- 所有 kernel launch 参数在 capture 时确定
- replay 时，整个图通过**一次 `cudaGraphLaunch` 调用**提交给 GPU
- GPU 端自行按依赖关系执行图中的节点，无需 CPU 逐个发起 kernel launch
- CPU 只需一次调用，之后可以立即做其他工作

无论图中有 50 个还是 500 个 kernel，CPU 端的开销都是一次 `cudaGraphLaunch`（约几微秒）。

### 3. 静态 Shape 与静态内存地址约束

**类比解释**：录像带录的是"在第 3 号锅里翻炒、在第 5 号碗里搅拌"。如果下次换了锅或碗的编号，录像带就失效了。所以你必须**永远用同一套锅碗**，只是每次放的食材不同。

**技术原理**：

**(a) 静态内存地址**：
- CUDA Graph 在 capture 阶段记录的是每个操作的**参数值（包括指针地址）**，而非对变量的引用
- `g.replay()` 时，CUDA driver 直接将录制好的命令序列提交给 GPU，**完全绕过 CPU 端的参数解析和内存分配**
- 如果 tensor 的 data pointer 改变了，replay 仍然读写旧地址 → 未定义行为或显存损坏
- 解决方案：预分配静态 buffer，replay 时只通过 `copy_()` 更新 buffer 内容

**(b) 静态 Shape**：
- kernel launch 参数（grid size、block size）在 capture 时固定
- shape 变化意味着需要不同的 kernel 配置，但 graph 中已锁定这些参数
- 内部临时显存的分配量也由 shape 决定
- 解决方案：为每种 batch size 分别 capture 一个 graph

**代码对应**：sglang 在 `cuda_graph_runner.py` 中预分配 `DecodeInputBuffers`（静态 buffer），capture 时锁定 shape，replay 时只做 `copy_()` 更新内容。

### 4. 10 行伪代码：最基本的 Capture-Replay 流程

```python
import torch

# 1. 预分配静态 buffer（地址永远不变）
static_input = torch.randn(10, device='cuda')

# 2. Warmup：预热 CUDA context、确定 kernel 选择（必须！）
for _ in range(2):
    static_output = static_input * 2 + 1

# 3. Capture：将 GPU 操作"录像"
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = static_input * 2 + 1   # 这行操作被录进 graph

# 4. Replay：换食材（数据），但锅碗（地址）不变
static_input.copy_(torch.ones(10, device='cuda'))  # 只改内容，不改地址
g.replay()                                           # 按录像重放
print(static_output)  # tensor([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.])
```

### 5. CUDA Graph 的限制

| 限制 | 原因 | sglang 的应对 |
|------|------|--------------|
| 输入 tensor 地址必须固定 | graph 记录的是指针值 | 预分配 `DecodeInputBuffers`，用 `copy_()` 更新 |
| Shape 必须固定 | kernel launch 参数在 capture 时固化 | 为每种 batch size 分别 capture；padding 到最近大小 |
| 不支持动态控制流 | graph 录制的是固定执行路径 | 使用 Breakable CUDA Graph 插入 eager 段 |
| Capture 中禁止 CPU-GPU 同步 | 同步会中断 capture 上下文 | 所有同步在 capture 外完成 |
| Capture 中的错误难以调试 | CUDA capture mode 限制错误报告 | `--debug-cuda-graph` 参数 + BCG |

---

## 二、sglang CUDA Graph 架构全景

### 1. Server 启动时的初始化链路

从启动到 CUDA Graph 就绪的完整调用链：

```
用户启动命令
  python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct
    |
    v
ServerArgs 解析
  python/sglang/srt/server_args.py
    |  关键参数:
    |  - cuda_graph_max_bs: 最大捕获 batch size
    |  - cuda_graph_bs: 指定捕获的 batch size 列表
    |  - disable_cuda_graph: 是否禁用
    |  - debug_cuda_graph: 调试模式（启用 BCG）
    |  - disable_piecewise_cuda_graph: 是否禁用 PCG
    |
    v
ModelRunner.__init__()                    model_runner.py:295-492
    |  初始化模型、tokenizer 等
    |
    v
ModelRunner.initialize()                  model_runner.py:646-740
    |  初始化 attention backend
    |  init_aux_hidden_state_capture()
    |
    v
ModelRunner.init_device_graphs()          model_runner.py:2554-2603
    |  根据 device 类型创建 runner:
    |  - CUDA/MUSA -> CudaGraphRunner
    |  - CPU        -> CPUGraphRunner
    |  - NPU        -> NPUGraphRunner
    |
    +--------------------------------------------------------------+
    v                                                              v
CudaGraphRunner.__init__()              PiecewiseCudaGraphRunner.__init__()
  cuda_graph_runner.py:515-656            piecewise_cuda_graph_runner.py
    |                                       |
    v                                       v
  CudaGraphRunner.capture()             PCG 初始化（编译+捕获）
    cuda_graph_runner.py:761-822
    |  逆序遍历 batch size（大->小，内存复用）
    v
  capture_one_batch_size() x N
    cuda_graph_runner.py:864-1078
    |  每个 batch size:
    |  1. 创建 ForwardBatch
    |  2. warmup x 2
    |  3. capture graph
    v
  CUDA Graph 就绪，等待推理请求
```

### 2. CudaGraphRunner 核心状态机

```
                    +------------------+
                    |  __init__()      |
                    |  分配静态 buffers |
                    |  确定 capture_bs  |
                    +--------+---------+
                             |
                             v
                    +------------------+
              +---->|    capture()     |<-----------------+
              |     |  逆序遍历 bs     |                   |
              |     +--------+---------+                   |
              |              |                             |
              |              v                             |
              |     +------------------+                   |
              |     |capture_one_bs()  |                   |
              |     |  warmup x 2      |                   |
              |     |  _capture_graph() |                   |
              |     |  存储 graph+buf   |                   |
              |     +------------------+                   |
              |              |                             |
              |              v                             |
              |     +------------------+                   |
              |     |  can_run() ?     |                    |
              |     |  检查 batch size  |                   |
              |     |  检查兼容性条件  |                    |
              |     +---+--------+----+                   |
              |         |        |                         |
              |    YES  |        |  NO                      |
              |         v        v                         |
              |   +----------+ +--------------+            |
              |   | replay() | |forward_decode |            |
              |   |  prepare | |  (普通执行)    |            |
              |   |  replay  | +--------------+            |
              |   +----------+                              |
              |                                             |
              +---------------------------------------------+
```

**核心状态变量**（`cuda_graph_runner.py:515-656`）：

| 变量 | 类型 | 作用 |
|------|------|------|
| `graphs` | `Dict[int, CUDAGraph]` | 每个 batch size 对应一个 captured graph |
| `output_buffers` | `Dict[int, Tensor]` | 每个 graph 的输出 buffer |
| `capture_bs` | `List[int]` | 需要捕获的 batch size 列表 |
| `buffers` | `DecodeInputBuffers` | 预分配的静态输入 buffer |
| `graph_memory_pool` | `int` | 跨 graph 共享的内存池 |

### 3. Decode 路径：标准 CUDA Graph

Decode 阶段（每次生成一个 token）是 CUDA Graph 的最佳适用场景：
- batch size 在一个 step 内固定
- 每个请求只处理 1 个 token
- kernel launch overhead 占比高

**执行流程**（`model_runner.py:2999-3004`）：

```
ModelRunner.forward_decode()
  |
  v
can_run = self.graph_runner.can_run(forward_batch)
  |  检查: batch size <= max captured bs
  |       无特殊条件（encoder_lens, hidden_mode 等）
  |
  +-- YES --> CudaGraphRunner.replay()
  |            |
  |            +-- replay_prepare()
  |            |    选择匹配的 captured bs（实际 bs <= captured bs）
  |            |    padding 输入到 captured bs
  |            |    copy 数据到静态 buffer
  |            |
  |            +-- graph.replay()          <-- 一次调用，重放所有 kernel
  |            |
  |            +-- 返回 output（slice 到实际 bs）
  |
  +-- NO ---> self.forward_decode()      <-- 普通 eager 执行（fallback）
```

### 4. Extend 路径：为什么标准 CUDA Graph 不够用

Extend（prefill）阶段有一个核心难题：**token 数量在每次请求间变化**。

```
请求 A: 128 tokens  --+
请求 B: 256 tokens  --+--- 同一个 batch，总 token 数 = 128+256+512 = 896
请求 C: 512 tokens  --+

下一个 batch:
请求 D: 64 tokens   --+
请求 E: 32 tokens   --+--- 总 token 数 = 96

-> shape 完全不同，标准 CUDA Graph 无法处理！
```

sglang 的解决方案：**Piecewise CUDA Graph (PCG)**
- 将模型按层切分为多个 piece
- 每个 piece 独立 capture 多种 token 长度
- 运行时 padding 到最近的已捕获长度
- 详见 [02_piecewise_cuda_graph_tutorial.md](./02_piecewise_cuda_graph_tutorial.md)

### 5. 整体架构图

```
+---------------------------------------------------------------------+
|                          sglang Server                               |
|                                                                      |
|  +--------------------------------------------------------------+  |
|  |                      ModelRunner                              |  |
|  |                                                               |  |
|  |   Decode 路径                    Extend 路径                   |  |
|  |   +---------------------+     +---------------------------+  |  |
|  |   |  CudaGraphRunner    |     | PiecewiseCudaGraphRunner  |  |  |
|  |   |                     |     |                           |  |  |
|  |   |  标准 CUDA Graph   |     |  按 layer 切分            |  |  |
|  |   |  或                 |     |  torch.compile 编译       |  |  |
|  |   |  BreakableCudaGraph |     |  多种 token length 捕获   |  |  |
|  |   |  (debug 模式)       |     |  padding + 分段 replay    |  |  |
|  |   |                     |     |                           |  |  |
|  |   |  适用: 固定 bs      |     |  适用: 动态 token 数      |  |  |
|  |   |  每次 1 token       |     |  prefill/extend           |  |  |
|  |   +---------------------+     +---------------------------+  |  |
|  |                                                               |  |
|  |   Fallback: forward_decode() / forward_extend() (eager)      |  |
|  +--------------------------------------------------------------+  |
|                                                                      |
|  配置入口:                                                           |
|  - --cuda-graph-bs / --cuda-graph-max-bs    (标准 CG)              |
|  - --disable-cuda-graph                      (禁用)                |
|  - --debug-cuda-graph                        (BCG debug)           |
|  - --disable-piecewise-cuda-graph            (禁用 PCG)            |
+---------------------------------------------------------------------+
```
