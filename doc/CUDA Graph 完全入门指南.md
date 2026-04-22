# CUDA Graph 完全入门指南

> 本指南面向从未接触过 CUDA Graph 的读者，从最基础的概念讲起，
> 结合 PyTorch API 和 sglang 项目中的真实代码进行讲解。

---

## 1. 什么是 CUDA Graph？

### 1.1 生活中的比喻：录音机

想象你在厨房做一道菜，每次都要翻菜谱：

```
第1步：切菜 → 查菜谱（CPU 发指令给 GPU）
第2步：热油 → 查菜谱（CPU 发指令给 GPU）
第3步：翻炒 → 查菜谱（CPU 发指令给 GPU）
第4步：调味 → 查菜谱（CPU 发指令给 GPU）
```

**每次执行一步，CPU 都要停下来"告诉 GPU 下一步干什么"**。
这个"告诉"的过程就是 **kernel launch overhead**（内核启动开销），
虽然单次只有几微秒，但在 LLM 推理中，一次 forward pass 可能有
上百个 GPU kernel，这些开销累加起来就很可观了。

**CUDA Graph 就像一台录音机：**

```
第一次做菜时：按下"录音"按钮，正常做一遍菜（Capture）
              录音机记住了所有步骤和顺序

之后每次做菜：按下"播放"按钮（Replay）
              录音机自动按录好的顺序执行所有步骤
              CPU 只需发一次"播放"指令，无需逐步指挥
```

### 1.2 它解决了什么问题？

```
+----------------------- 没有 CUDA Graph ------------------------+
|                                                                 |
|  CPU: --launch--launch--launch--launch--launch--launch----     |
|           |       |       |       |       |       |           |
|  GPU:   [K1]    [K2]    [K3]    [K4]    [K5]    [K6]         |
|         ----+------+------+------+------+------+------> 时间  |
|             gap    gap    gap    gap    gap                     |
|           (CPU调度开销，GPU空闲等待)                            |
|                                                                 |
+-----------------------------------------------------------------+

+------------------------ 有 CUDA Graph -------------------------+
|                                                                 |
|  CPU: --launch--(去干别的事)--------------------------------     |
|           |                                                     |
|  GPU:   [K1->K2->K3->K4->K5->K6]  (一次 Replay 播放全部)      |
|         --------------------------------------------------> 时间|
|         无间隙，GPU 全速执行                                    |
|                                                                 |
+-----------------------------------------------------------------+

K = Kernel（GPU 计算单元）
gap = CPU->GPU 调度造成的空闲间隙
```

**核心收益：**
- 消除 CPU→GPU 的逐个 kernel 启动开销
- GPU kernel 之间无缝衔接，提高 GPU 利用率
- 对 LLM decode 阶段（小 batch、大量 kernel）提速尤为显著

### 1.3 核心工作流程：Capture → Instantiate → Replay

CUDA Graph 的生命周期分三个阶段：

```
+-------------+     +-------------+     +-------------+
|   Capture    | --> | Instantiate | --> |   Replay    |
|   (录制)     |     |  (实例化)    |     |  (回放)     |
+-------------+     +-------------+     +-------------+
    录下 GPU 操作       优化并固化          重复执行
    记录依赖关系        分配显存            只需一次 API 调用
```

**详细流程：**

```
阶段1: Capture (录制)
+------------------------------------------------------+
|  1. 分配固定的输入/输出缓冲区 (Buffer)                |
|  2. 进入录制模式                                      |
|  3. 正常执行一遍模型 forward                          |
|  4. CUDA 驱动记录所有 GPU kernel 及其依赖关系         |
|  5. 退出录制模式                                      |
+------------------------------------------------------+
         |
阶段2: Instantiate (实例化)
+------------------------------------------------------+
|  1. CUDA 驱动优化录制的 kernel 执行顺序              |
|  2. 预分配所有需要的显存                              |
|  3. 生成可执行的计算图 (Executable Graph)             |
+------------------------------------------------------+
         |
阶段3: Replay (回放) <-- 可重复执行无数次
+------------------------------------------------------+
|  1. 将新的输入数据复制到预分配的输入缓冲区            |
|  2. 调用 graph.replay()                              |
|  3. CUDA 驱动一次性执行所有 kernel                   |
|  4. 从输出缓冲区读取结果                              |
+------------------------------------------------------+
```

---

## 2. CUDA Graph 的基本约束

### 2.1 为什么只能录制固定的 GPU 操作？

CUDA Graph 的本质是**提前确定整个执行计划**。
就像录音机录的是声音波形，不是"某种类型的音乐"——
录下来的就是固定的波形，播放时不能改变。

```
可以录制的：                         不能录制的：
-------------------------             ------------------------
+ 固定的数学运算 (矩阵乘法)          - Python if/else 分支
+ 固定的 kernel 调用序列             - 动态循环次数
+ 固定的内存地址(预分配Buffer)        - 运行时决定 tensor 大小
+ 固定的数据流依赖关系               - CPU 端的逻辑判断
```

**根本原因：** CUDA Graph 在 Capture 阶段记录的是 GPU 硬件层面的
操作序列（kernel + 参数地址），而不是高层逻辑。Replay 时只是
机械地重复这些硬件操作，无法在中途插入新的决策。

### 2.2 为什么不支持条件分支和动态形状？

```
+------ 理想世界 ------+     +------ 现实世界 ------+
|                       |     |                       |
|  if (x > 0):          |     |  录制时 x = 5         |
|      路径A: kernel1   |     |  -> 录下了 kernel1     |
|  else:                |     |                       |
|      路径B: kernel2   |     |  回放时 x = -3        |
|                       |     |  -> 仍然执行 kernel1   |
|                       |     |  -> 路径B 永远不会执行  |
+-----------------------+     +-----------------------+

  逻辑上的分支            实际录下来的是单条路径
```

**动态形状的问题类似：**

```
录制时: matmul(shape=[4, 128, 128])  -> 记录了这个特定形状的 kernel
回放时: 想执行 matmul(shape=[8, 128, 128])  -> 不行！形状不匹配
```

**sglang 的解决方案：为不同 batch size 录制不同的 Graph**

```python
# sglang 中的真实做法（简化）
# python/sglang/srt/model_executor/cuda_graph_runner.py

class CudaGraphRunner:
    def capture(self):
        # 为每个预定义的 batch size 单独录制一张图
        for bs in self.cuda_graph_bs:  # 例如 [1, 2, 4, 8, 16, 32]
            graph = torch.cuda.CUDAGraph()
            # ... 准备 bs 大小的输入缓冲区 ...
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = model.forward(input_ids, positions, forward_batch)
            self.graphs[bs] = graph  # 存储起来，replay 时按 bs 查找
```

```
录制多张 Graph：

  bs=1  ->  Graph_1   (录制 batch_size=1 时的所有 kernel)
  bs=2  ->  Graph_2   (录制 batch_size=2 时的所有 kernel)
  bs=4  ->  Graph_4   (录制 batch_size=4 时的所有 kernel)
  ...
  bs=32 ->  Graph_32  (录制 batch_size=32 时的所有 kernel)

运行时：

  实际 batch_size=3
  -> 找到最接近的已录制大小 (bs=4)
  -> 将 3 条数据 padding 到 4 条
  -> Replay Graph_4
```

### 2.3 "所有操作必须在同一个 CUDA Stream 上"意味着什么？

**CUDA Stream 是什么？**

可以把 Stream 想象成一条**传送带**：
- 同一个 Stream 上的任务**按顺序执行**
- 不同 Stream 上的任务**可以并行执行**

```
Stream 0 (默认): --[K1]--[K2]--[K3]-->  (串行)
Stream 1:        --[K4]--[K5]-------->  (与 Stream 0 并行)
```

**CUDA Graph 要求录制在单个 Stream 上：**

```
+----------- 可以 ------------+     +----------- 不行 ------------+
|                              |     |                              |
|  Stream 0 (录制):            |     |  Stream 0 (录制):            |
|  --[K1]--[K2]--[K3]-->     |     |  --[K1]--[K3]-->            |
|                              |     |                              |
|  所有 kernel 在同一个        |     |  Stream 1:                   |
|  stream 上，依赖关系清晰     |     |  --[K2]-->                  |
|                              |     |  跨 stream 录制会混乱       |
+------------------------------+     +------------------------------+
```

**原因：** CUDA Graph 需要精确知道 kernel 之间的依赖关系。
如果 kernel 分布在多个 stream 上，依赖关系由 stream 间的
同步事件决定，录制时很难完整捕获。

---

## 3. PyTorch 中的 CUDA Graph API

### 3.1 核心对象：`torch.cuda.CUDAGraph`

```python
import torch

# CUDAGraph 是一个容器，用来存储录制的计算图
g = torch.cuda.CUDAGraph()

# 它本质上是一个不透明的句柄 (opaque handle)
# 你不需要关心它的内部结构
# 只需要知道两件事：
#   1. 用 torch.cuda.graph() 往里面录制
#   2. 用 g.replay() 来回放
```

### 3.2 `torch.cuda.graph()` 上下文管理器

这是录制 CUDA Graph 的核心 API：

```python
with torch.cuda.graph(graph, pool=None, stream=None):
    # 在这个 with 块里的所有 GPU 操作都会被录制
    output = model(input)
```

**参数说明：**
- `graph`: 一个 `torch.cuda.CUDAGraph` 实例，用来存储录制结果
- `pool`: 显存池句柄，多个 Graph 共享同一池可以节省显存
- `stream`: 录制使用的 CUDA Stream

### 3.3 最小可运行 Demo

以下是一个完整的、可直接运行的 CUDA Graph 示例：

```python
import torch

# ===== 准备模型和数据 =====
# 一个极简的"模型"：一个线性层 + ReLU
model = torch.nn.Linear(4, 2).cuda()
model.eval()

# ===== 步骤1: 预分配固定的输入/输出缓冲区 =====
# 关键：这些缓冲区的地址在 capture 和 replay 时必须相同
static_input = torch.randn(4, device="cuda")    # 固定的输入缓冲区
static_output = None                            # 将在 capture 时确定

# ===== 步骤2: 预热 (Warm-up) =====
# 必须在 capture 前跑几次，让 PyTorch 完成各种内部初始化
# (JIT 编译、内存分配策略选择等)
with torch.no_grad():
    for _ in range(3):
        static_output = model(static_input)

# ===== 步骤3: Capture (录制) =====
g = torch.cuda.CUDAGraph()

with torch.no_grad():
    with torch.cuda.graph(g):
        static_output = model(static_input)  # 这里的操作会被录制

print("Capture 完成!")
print(f"录制时的输出: {static_output}")

# ===== 步骤4: Replay (回放) =====
# 回放时只需要把新数据拷贝到 static_input，然后 replay
new_data = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
static_input.copy_(new_data)  # 关键：用 copy_ 写入固定缓冲区，不能重新赋值

g.replay()  # 一次性执行所有录制的 kernel

print(f"Replay 后的输出: {static_output}")  # 结果已经更新到 static_output

# ===== 验证正确性 =====
with torch.no_grad():
    expected = model(new_data)
print(f"直接计算的输出: {expected}")
print(f"结果是否一致: {torch.allclose(static_output, expected)}")
```

### 3.4 完整流程图解

```
                 +-----------------------------+
                 |     预分配 Buffer            |
                 |  static_input (GPU 显存)     |
                 |  static_output (GPU 显存)    |
                 +-----------------------------+
                              |
                 +-----------------------------+
                 |     Warm-up (预热)           |
                 |  跑 2-3 次 forward           |
                 |  让 PyTorch 初始化完成       |
                 +-----------------------------+
                              |
              +-------------------------------+
              |         Capture (录制)          |
              |                                |
              |   graph = CUDAGraph()          |
              |   with torch.cuda.graph(graph):|
              |       output = model(input)    |
              |                                |
              |   此时所有 GPU kernel 被记录   |
              +-------------------------------+
                              |
              +-------------------------------+
              |       Replay (回放) x N        |
              |                                |
              |   for each new_request:        |
              |     static_input.copy_(data)   |
              |     graph.replay()             |
              |     result = static_output     |
              |                                |
              |   CPU 只发一次 replay 指令     |
              |   GPU 连续执行所有 kernel      |
              +-------------------------------+
```

### 3.5 常见陷阱

**陷阱 1：在 capture 块中创建新 tensor**

```python
# [X] 错误：在 capture 中创建新 tensor
with torch.cuda.graph(g):
    x = torch.randn(4, device="cuda")  # 每次 replay 都会分配新内存！
    output = model(x)

# [V] 正确：使用预分配的 buffer
static_input = torch.randn(4, device="cuda")  # 在 capture 外分配
with torch.cuda.graph(g):
    output = model(static_input)               # 使用固定地址
```

**陷阱 2：忘记 warm-up**

```python
# [X] 错误：没有 warm-up 就直接 capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input)  # 第一次运行，PyTorch 内部可能有惰性初始化

# [V] 正确：先 warm-up 再 capture
with torch.no_grad():
    for _ in range(3):
        _ = model(input)  # 预热
g = torch.cuda.CUDAGraph()
with torch.no_grad():
    with torch.cuda.graph(g):
        output = model(input)  # 现在可以安全录制了
```

**陷阱 3：capture 后修改 tensor 的引用**

```python
static_output = None
with torch.cuda.graph(g):
    static_output = model(static_input)

# [X] 错误：重新赋值会断开与 graph 的连接
static_output = some_other_tensor  # graph 内部仍指向旧地址

# [V] 正确：使用 in-place 操作或直接读取
result = static_output  # 直接读取，不要重新赋值
```

---

## 4. sglang 中的 CUDA Graph 架构

### 4.1 整体架构

sglang 针对不同的推理阶段，使用了不同的 CUDA Graph 策略：

```
+-------------------------------------------------------------+
|                   sglang CUDA Graph 架构                     |
|                                                              |
|  +---------------------+    +------------------------------+ |
|  |  CudaGraphRunner    |    |  PiecewiseCudaGraphRunner    | |
|  |  (Decode 阶段)       |    |  (Prefill/Extend 阶段)       | |
|  |                      |    |                              | |
|  |  按 batch size 录制  |    |  按 token 数量分段录制       | |
|  |  文件:               |    |  文件:                       | |
|  |  cuda_graph_runner.py|    |  piecewise_cuda_graph_       | |
|  |                      |    |  runner.py                   | |
|  +---------------------+    +------------------------------+ |
|                                                              |
|  +---------------------+    +------------------------------+ |
|  |  BreakableCUDAGraph |    |  ViTCudaGraphRunner          | |
|  |  (可中断的 Graph)    |    |  (视觉模型专用)              | |
|  |                      |    |                              | |
|  |  支持 graph break    |    |  为 Vision Transformer       | |
|  |  可在录制中插入      |    |  单独录制 CUDA Graph         | |
|  |  CPU 操作            |    |                              | |
|  +---------------------+    +------------------------------+ |
+-------------------------------------------------------------+
```

### 4.2 CudaGraphRunner 核心流程（Decode 阶段）

这是 sglang 中最核心的 CUDA Graph 实现，用于 **decode（逐 token 生成）** 阶段。

源文件: `python/sglang/srt/model_executor/cuda_graph_runner.py`

```
Capture 阶段：
+----------------------------------------------------------+
|                                                           |
|  for bs in [1, 2, 4, 8, 16, 32, ...]:                   |
|      1. 预分配 bs 大小的固定 Buffer (DecodeInputBuffers) |
|      2. warm-up: 同步 + 跑 2 次 forward                  |
|      3. graph = torch.cuda.CUDAGraph()                   |
|      4. with torch.cuda.graph(graph, pool, stream):      |
|             output = model.forward(input_ids, positions)  |
|      5. self.graphs[bs] = graph                          |
|                                                           |
+----------------------------------------------------------+

Replay 阶段 (每次有新请求时)：
+----------------------------------------------------------+
|                                                           |
|  实际 batch_size = 当前请求数                              |
|  graph_bs = 找到 >= 实际 bs 的最小已录制大小               |
|                                                           |
|  1. 将实际数据 copy 到预分配的 Buffer                     |
|     buffers.input_ids[:bs].copy_(real_input_ids)          |
|     buffers.positions[:bs].copy_(real_positions)          |
|     buffers.seq_lens[:bs].copy_(real_seq_lens)            |
|                                                           |
|  2. 多余位置 padding (如果 graph_bs > 实际 bs)            |
|     buffers.seq_lens[bs:graph_bs].fill_(dummy_value)      |
|                                                           |
|  3. self.graphs[graph_bs].replay()  <-- 一次性执行!       |
|                                                           |
|  4. 从 output buffer 读取结果                              |
|     logits = buffers.next_token_logits[:bs]               |
|                                                           |
+----------------------------------------------------------+
```

### 4.3 Buffer 预分配策略

sglang 中的 `DecodeInputBuffers` 为每个 batch size 录制时都预分配了
**最大 batch size** 的缓冲区，然后通过切片使用：

```python
# python/sglang/srt/model_executor/cuda_graph_runner.py
# DecodeInputBuffers.create() 简化版

@classmethod
def create(cls, *, device, max_bs, hidden_size, vocab_size, dtype, ...):
    with torch.device(device):
        input_ids = torch.zeros((max_bs,), dtype=torch.int64)        # 最大容量
        req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64)
        seq_lens = torch.full((max_bs,), fill_value, dtype=torch.int32)
        positions = torch.zeros((max_bs,), dtype=torch.int64)
        next_token_logits_buffer = torch.zeros((max_bs, vocab_size), dtype=torch.float)
        # ... 其他 buffer ...

# Capture 时：只使用前 bs 个元素
input_ids = buffers.input_ids[:bs]            # 切片，地址不变
req_pool_indices = buffers.req_pool_indices[:bs]
seq_lens = buffers.seq_lens[:bs]

# Replay 时：copy 实际数据到 buffer 的前 raw_bs 个位置
buffers.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
buffers.positions[:raw_num_token].copy_(forward_batch.positions)
```

```
Buffer 内存布局 (max_bs = 32):

  input_ids:  [0][0][0][0][0]...[0][0][0][0]   共 32 个位置
               |-- bs=4 使用 --|

  Capture bs=4 时：录制使用 input_ids[:4]
  Replay 时：      copy 数据到 input_ids[:4]，replay，读取 output[:4]

  +-------------------------------------------------------------+
  | [实际数据][实际数据][实际数据][实际数据][pad]|...[未使用]    |
  |  <-- raw_bs=3 ---><-- padding -->                            |
  |  <----------- graph_bs=4 ------------->                     |
  +-------------------------------------------------------------+
```

### 4.4 显存池共享

多个 Graph 共享同一个显存池，避免重复分配：

```python
# python/sglang/srt/model_executor/cuda_graph_runner.py
# 简化版

# 全局显存池，所有 graph 共享
global_graph_memory_pool = None

def capture_one_batch_size(self, bs, forward, stream_idx=None):
    global global_graph_memory_pool

    graph = torch.cuda.CUDAGraph()

    # Warm-up: 跑两次确保所有内部状态稳定
    for _ in range(2):
        torch.cuda.synchronize()
        run_once()

    # 首次 capture 时，获取显存池句柄
    if global_graph_memory_pool is None:
        global_graph_memory_pool = torch.cuda.graph_pool_handle()

    # 使用共享池进行 capture
    with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
        output = run_once()

    self.graphs[bs] = graph
    return output
```

```
显存池共享示意：

  +---------------------------------------------+
  |           共享显存池 (Memory Pool)            |
  |                                              |
  |  +---------+ +---------+ +---------+        |
  |  | Graph_1 | | Graph_2 | | Graph_4 |        |
  |  | bs=1    | | bs=2    | | bs=4    |        |
  |  +---------+ +---------+ +---------+        |
  |                                              |
  |  所有 Graph 共享这块显存区域                 |
  |  只有当前正在 Replay 的 Graph 占用显存       |
  |  (同一时刻只有一个 Graph 在执行)             |
  +---------------------------------------------+

  好处：总显存 ~= 最大的那个 Graph 的显存
        而不是所有 Graph 显存之和
```

---

## 5. 总结

```
+----------------------------------------------------+
|                CUDA Graph 关键要点                  |
|                                                     |
|  1. 本质：提前录制 GPU 操作序列，回放时一次性执行   |
|                                                     |
|  2. 收益：消除 kernel launch overhead               |
|     对小 batch、多 kernel 的场景（如 decode）最有效  |
|                                                     |
|  3. 约束：                                          |
|     - 操作必须固定（不能有动态分支）                |
|     - 形状必须固定（不能有动态 tensor 大小）        |
|     - 必须在同一个 CUDA stream 上 TODO?                 |
|     - 输入/输出使用预分配的固定地址 buffer          |
|                                                     |
|  4. 三步流程：                                      |
|     预分配 Buffer -> Capture (录制) -> Replay (回放)|
|                                                     |
|  5. sglang 中的策略：                               |
|     - Decode 阶段：按 batch size 录制多个 Graph     |
|     - Prefill 阶段：按 token 数分段录制             |
|     - 显存池共享：所有 Graph 共用一块显存           |
|     - Padding：实际 bs 小于录制 bs 时填充           |
+----------------------------------------------------+
```

---

## 6. 深入分析：sglang 原始 CUDA Graph 实现

> 本节深入分析 `python/sglang/srt/model_executor/cuda_graph_runner.py` 中的原始
> CUDA Graph 实现（不涉及 PCG Piecewise 和 BCG Breakable 扩展），
> 逐阶段讲解初始化、录制、重放的完整流程。

### 6.0 核心类关系总览

```
+------------------------------------------------------------------+
|                        CudaGraphRunner                            |
|                     (cuda_graph_runner.py:512)                    |
|                                                                   |
|  self.graphs: Dict[int, CUDAGraph]   -- bs -> 录制好的计算图      |
|  self.output_buffers: Dict[int, Obj] -- bs -> 录制时的输出引用    |
|  self.buffers: DecodeInputBuffers     -- 预分配的 GPU 缓冲区     |
|  self.capture_bs: List[int]           -- 需要录制的 batch size   |
|                                                                   |
|  主要方法:                                                        |
|    __init__()         -> 初始化 + 触发 capture                   |
|    capture()          -> 遍历所有 bs 批量录制                    |
|    capture_one_batch_size() -> 录制单个 bs 的 graph              |
|    _capture_graph()   -> 底层 capture 调用                       |
|    replay()           -> 查找 graph + 数据拷贝 + replay          |
|    replay_prepare()   -> 数据拷贝到预分配 buffer                  |
+------------------------------------------------------------------+
         |
         | 使用
         v
+------------------------------------------------------------------+
|                      DecodeInputBuffers                           |
|                    (cuda_graph_runner.py:128)                     |
|                                                                   |
|  预分配的 GPU 张量:                                               |
|    input_ids:      [max_num_token]           int64               |
|    req_pool_indices: [max_bs]                int64               |
|    seq_lens:       [max_bs]                 int32               |
|    positions:      [max_num_token]           int64               |
|    out_cache_loc:  [max_num_token]           int64               |
|    next_token_logits_buffer: [max_num_token, vocab_size] float  |
|    ... 其他 buffer ...                                            |
|                                                                   |
|  核心方法:                                                        |
|    create()            -> 类方法，在 GPU 上分配所有 tensor        |
|    populate_from_forward_batch() -> 把真实数据 copy 到 buffer    |
+------------------------------------------------------------------+
```

### 6.1 初始化阶段 (`__init__`)

> 源码位置: `cuda_graph_runner.py:515-655`

#### 6.1.1 `self.graphs` 和 `self.output_buffers`

```python
# cuda_graph_runner.py:520-521
self.graphs = {}          # 类型: Dict[int, torch.cuda.CUDAGraph]
self.output_buffers = {}  # 类型: Dict[int, LogitsProcessorOutput]
```

**作用：** 这是两张查找表，key 是 batch size：

```
self.graphs:
  +------+------------------+
  | key  | value            |
  +------+------------------+
  |  1   | CUDAGraph (bs=1) |
  |  2   | CUDAGraph (bs=2) |
  |  4   | CUDAGraph (bs=4) |
  |  8   | CUDAGraph (bs=8) |
  | ...  | ...              |
  | 256  | CUDAGraph (bs=256)|
  +------+------------------+

self.output_buffers:
  +------+---------------------------+
  | key  | value                     |
  +------+---------------------------+
  |  1   | LogitsProcessorOutput     |
  |  2   | LogitsProcessorOutput     |
  | ...  | ...                       |
  | 256  | LogitsProcessorOutput     |
  +------+---------------------------+

  output_buffers 存储的是 capture 时 forward 的返回值引用。
  由于 capture 时使用了预分配的 buffer，replay 后这些引用
  指向的 GPU 内存会自动被更新为最新的计算结果。
```

**为什么 key 是 int 而不是字符串？** 因为 decode 阶段的 graph
完全由 batch size 决定——同样的 bs，GPU kernel 序列完全相同。

#### 6.1.2 `capture_bs` 是什么？为什么需要多个 batch size？

```python
# cuda_graph_runner.py:572-574
self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
    model_runner, self.num_tokens_per_bs
)
# 例: capture_bs = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
```

**`capture_bs` 的来源** (`get_batch_sizes_to_capture`, `cuda_graph_runner.py:462`):

```
输入: server_args.cuda_graph_bs (用户配置或默认值)
      例: [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]

处理:
  1. 过滤掉超过 max_running_requests 的 bs
  2. 确保每个 bs * num_tokens_per_bs 是并行度(mul_base)的倍数
  3. 必要时追加 num_max_requests 到列表
  4. 去重并排序

输出: capture_bs = [1, 2, 4, 8, 16, 32, 48, 64, ...]
```

**为什么需要多个 batch size？** 这是 CUDA Graph 最核心的限制：

```
CUDA Graph 只能录制"固定形状"的操作。
batch_size=4 时的 kernel 序列和 batch_size=8 时完全不同
(矩阵维度不同，计算量不同)。

所以必须为每个可能的 batch size 单独录制一张图。

但不可能为每个整数 bs 都录制(太耗显存和时间)，
所以选择一组"代表性"的 batch size，
运行时通过 padding 机制匹配到最近的已录制大小。
```

**`compile_bs`** 是其中需要 `torch.compile` 优化的子集。

#### 6.1.3 `DecodeInputBuffers.create(...)` 做了什么？

```python
# cuda_graph_runner.py:624-644
self.buffers: DecodeInputBuffers = DecodeInputBuffers.create(
    device=self.device,
    max_bs=self.max_bs,           # 所有 capture_bs 中的最大值
    max_num_token=self.max_num_token,  # max_bs * num_tokens_per_bs
    hidden_size=...,
    vocab_size=...,
    ...
)
```

**`create()` 方法内部** (`cuda_graph_runner.py:150-260`):

```python
@classmethod
def create(cls, *, device, max_bs, max_num_token, ...):
    with torch.device(device):  # 所有 tensor 创建在 GPU 上
        input_ids = torch.zeros((max_num_token,), dtype=torch.int64)
        input_embeds = torch.zeros((max_num_token, hidden_size), dtype=dtype)
        req_pool_indices = torch.zeros((max_bs,), dtype=torch.int64)
        seq_lens = torch.full((max_bs,), seq_len_fill_value, dtype=torch.int32)
        out_cache_loc = torch.zeros((max_num_token,), dtype=cache_loc_dtype)
        positions = torch.zeros((max_num_token,), dtype=torch.int64)
        mrope_positions = torch.zeros((3, max_num_token), dtype=torch.int64)
        num_token_non_padded = torch.zeros((1,), dtype=torch.int32)
        next_token_logits_buffer = torch.zeros(
            (max_num_token, vocab_size), dtype=torch.float)
        # ... mamba, encoder_lens, pp_proxy, ngram 等 ...
    return cls(input_ids=input_ids, ...)
```

**为什么要预分配 buffer？**

```
+----------- CUDA Graph 的刚性约束 ---------------------------------+
|                                                                     |
|  Capture 时录制的 kernel 记录的是"内存地址"，不是"变量名"。       |
|  Replay 时必须在相同的内存地址上读写数据。                        |
|                                                                     |
|  所以：                                                             |
|  - 输入数据必须写入 capture 时使用的那块 GPU 内存                  |
|  - 输出数据也会出现在 capture 时确定的 GPU 内存位置               |
|  - 不能在 replay 时分配新的 tensor (地址会变)                     |
|                                                                     |
|  DecodeInputBuffers 就是一块"永久固定地址"的 GPU 内存。          |
|  所有 batch size 的 graph 共享这一块 buffer:                      |
|                                                                     |
|  buffers.input_ids (max_num_token=256)                             |
|  +--+--+--+--+--+--+...+--+--+--+--+--+...+--+--+                |
|  |  |  |  |  |  |  |...|  |  |  |  |  |...|  |  |                |
|  +--+--+--+--+--+--+...+--+--+--+--+--+...+--+--+                |
|  |<- bs=4 录制时用 ->|                                             |
|  |<- bs=8 录制时用 --------->|                                     |
|  |<- bs=256 录制时用 ---------------------------------->|          |
|                                                                     |
|  不同 bs 的 graph 录制时切片同一块 buffer 的不同前缀部分          |
+---------------------------------------------------------------------+
```

#### 6.1.4 初始化的最后一步：触发 capture

```python
# cuda_graph_runner.py:648-655
try:
    with model_capture_mode():  # 设置全局标志 is_capture_mode = True
        self.capture()          # 录制所有 batch size 的 graph
except RuntimeError as e:
    raise Exception(
        f"Capture cuda graph failed: {e}\n{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
    )
```

**`model_capture_mode()`** (`cuda_graph_runner.py:373-380`):
一个简单的上下文管理器，设置全局变量 `is_capture_mode = True`，
让模型内部知道当前处于录制状态（可以跳过某些不需要录制的逻辑）。

---

### 6.2 录制阶段 (`capture` -> `capture_one_batch_size` -> `_capture_graph`)

#### 6.2.1 `capture()` 方法 -- 从大到小录制

```python
# cuda_graph_runner.py:761-822
def capture(self) -> None:
    def _capture_one_stream(stream_idx=None):
        avail_mem = get_available_gpu_memory(...)
        # 关键: reversed! 从大到小录制
        capture_range = (
            tqdm.tqdm(list(reversed(self.capture_bs)))
            ...
        )
        for i, bs in enumerate(capture_range):
            with patch_model(...) as forward:
                graph, output_buffers = self.capture_one_batch_size(
                    bs, forward, stream_idx
                )
                key = bs if stream_idx is None else f"{stream_idx}_{bs}"
                self.graphs[key] = graph
                self.output_buffers[key] = output_buffers

    with freeze_gc(...):      # 冻结 GC 避免录制期间垃圾回收
        with graph_capture() as ctx:  # 创建专用的 CUDA stream
            self.stream = ctx.stream
            _capture_one_stream()
```

**为什么从大到小录制？**

```
原因: 显存池共享机制

所有 graph 共享同一个显存池 (global_graph_memory_pool)。
显存池的大小 = 历史上分配过的最大峰值。

录制顺序 1: bs=1,2,4,...,256 (从小到大)
  bs=1   -> 显存池峰值 100MB
  bs=2   -> 显存池峰值 200MB
  bs=4   -> 显存池峰值 400MB
  ...
  bs=256 -> 显存池峰值 10GB
  总耗时: 每个阶段都需要分配新的显存

录制顺序 2: bs=256,...,4,2,1 (从大到小)  <-- 实际采用
  bs=256 -> 显存池峰值 10GB (一次性分配到位)
  bs=128 -> 复用显存池，无需新分配 (10GB 内有足够空间)
  bs=64  -> 复用显存池
  ...
  bs=1   -> 复用显存池
  好处: 小 bs 完全复用大 bs 已分配的显存，无需额外分配

+-------------------------------------------------+
|  global_graph_memory_pool (10GB)                |
|  +---------------------------+                  |
|  | bs=256 录制时分配的显存    |  <- 首次分配     |
|  +---------------------------+                  |
|  | bs=128 直接复用其中一部分  |  <- 无需新分配   |
|  +---+                       |                  |
|  |bs=1|                      |  <- 无需新分配   |
|  +---+-----------------------+                  |
+-------------------------------------------------+
```

#### 6.2.2 `capture_one_batch_size()` -- 核心录制方法

> 源码位置: `cuda_graph_runner.py:864-1078`

这个方法是录制阶段的核心，整个流程分 5 步：

```
capture_one_batch_size(bs, forward, stream_idx)
  |
  +---> 步骤 1: 从 self.buffers 切片出当前 bs 大小的 view
  |     (cuda_graph_runner.py:873-884)
  |
  +---> 步骤 2: 构建 ForwardBatch 对象
  |     (cuda_graph_runner.py:972-1003)
  |
  +---> 步骤 3: 初始化 attention backend 的录制元数据
  |     (cuda_graph_runner.py:1019-1027)
  |
  +---> 步骤 4: 定义 run_once()，然后 warm-up 2 遍
  |     (cuda_graph_runner.py:1030-1068)
  |
  +---> 步骤 5: 真正的 capture (录制)
        (cuda_graph_runner.py:1070-1078)
```

**步骤 1: 切片 buffer**

```python
# cuda_graph_runner.py:873-884
num_tokens = bs * self.num_tokens_per_bs

# 关键：切片操作不会复制数据！
# input_ids 仍然指向 self.buffers.input_ids 的同一块 GPU 内存
# 只是 view 的范围缩小到 [:num_tokens]
input_ids = buffers.input_ids[:num_tokens]
req_pool_indices = buffers.req_pool_indices[:bs]
seq_lens = buffers.seq_lens[:bs]
out_cache_loc = buffers.out_cache_loc[:num_tokens]
positions = buffers.positions[:num_tokens]
next_token_logits_buffer = buffers.next_token_logits_buffer[:num_tokens]
```

```
self.buffers.input_ids (max_num_token = 256):
+---+---+---+---+---+---+---+---+...+---+---+
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |...| 0 | 0 |
+---+---+---+---+---+---+---+---+...+---+---+
|<- input_ids[:4] when bs=4 ->|
|<- input_ids[:8] when bs=8 -------->|

切片不创建新 tensor，只创建一个新的 view。
底层的数据指针(data_ptr)不变，只是 shape/stride 不同。
这对 CUDA Graph 至关重要：replay 时写入同一个地址。
```

**步骤 2: 构建 ForwardBatch**

```python
# cuda_graph_runner.py:972-1003
forward_batch = ForwardBatch(
    forward_mode=self.capture_forward_mode,  # DECODE
    batch_size=bs,
    input_ids=input_ids,                     # 切片 view
    req_pool_indices=req_pool_indices,        # 切片 view
    seq_lens=seq_lens,                       # 切片 view
    positions=positions,                     # 切片 view
    next_token_logits_buffer=next_token_logits_buffer,  # 切片 view
    attn_backend=attn_backend,
    # ... 更多字段 ...
)
```

这一步把切片后的 buffer view 组装成一个完整的 `ForwardBatch`，
它和正常运行时的 `ForwardBatch` 结构完全相同，只是数据是空的(dummy)。

**步骤 3: attention backend 初始化**

```python
# cuda_graph_runner.py:1019-1027
attn_backend.init_forward_metadata_capture_cuda_graph(
    bs, num_tokens, req_pool_indices, seq_lens,
    encoder_lens, forward_batch.forward_mode, forward_batch.spec_info,
)
```

Attention backend 需要在录制前准备好自己的 CUDA graph 状态
（如预分配的 workspace buffer），确保录制时所有 GPU 操作都使用固定地址。

**步骤 4: `run_once()` + warm-up**

```python
# cuda_graph_runner.py:1030-1061
def run_once():
    # 清理 DP attention 的缓存
    forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
    set_dp_buffer_len(global_dp_buffer_len, num_tokens, ...)
    set_is_extend_in_batch(False)

    kwargs = {}
    if self.pp_size > 1:
        kwargs["pp_proxy_tensors"] = PPProxyTensors(...)

    # 核心调用：执行模型 forward
    logits_output = forward(input_ids, forward_batch.positions, forward_batch, **kwargs)
    return logits_output
```

**warm-up 循环** (`cuda_graph_runner.py:1065-1068`):

```python
for _ in range(2):
    self.device_module.synchronize()       # 等 GPU 完成所有操作
    self.model_runner.tp_group.barrier()   # 等所有 TP rank 同步
    run_once()                             # 跑一次 forward
```

**为什么 warm-up 要跑两遍？**

```
第1遍 warm-up (关键):
  - PyTorch 首次运行时会进行 JIT 编译
  - CUDA kernel 的 lazy initialization (首次调用才加载)
  - 内部内存分配器确定内存分配策略
  - cuBLAS/cuDNN 自动调优 (选择最快的 kernel 实现)
  - torch.compile 的图编译 (如果启用)

第2遍 warm-up:
  - 确保第1遍产生的所有副作用稳定
  - 验证运行结果的一致性
  - 让 CUDA driver 的内部缓存预热

如果跳过 warm-up 直接 capture:
  - capture 过程中可能触发新的内存分配
  - 某些 lazy op 的首次执行会导致 graph 结构变化
  - 结果：capture 的 graph 可能不完整或不稳定
```

**步骤 5: 真正的 capture**

```python
# cuda_graph_runner.py:1070-1078
if get_global_graph_memory_pool() is None:
    set_global_graph_memory_pool(self.device_module.graph_pool_handle())
set_graph_pool_id(get_global_graph_memory_pool())

out = self._capture_graph(
    graph,                    # 空 CUDAGraph 对象
    get_global_graph_memory_pool(),  # 共享显存池
    stream,                   # 专用 CUDA stream
    run_once                  # 要录制的函数
)
return graph, out
```

#### 6.2.3 `_capture_graph()` -- 底层录制调用

```python
# cuda_graph_runner.py:824-855
def _capture_graph(self, graph, pool, stream, run_once_fn):
    # 选择录制方式
    if envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.get():
        graph_ctx = BreakableCUDAGraphCapture   # BCG 扩展
    else:
        graph_ctx = self.device_module.graph    # 标准 torch.cuda.graph

    captured_fn = run_once_fn

    # 进入录制模式
    with graph_ctx(cuda_graph=graph, pool=pool, stream=stream):
        out = captured_fn()   # run_once() 被调用，但这次被录制
    return out
```

**`with torch.cuda.graph(graph, pool, stream):` 这一行发生了什么？**

```
1. CUDA 驱动进入"stream capture"模式
   - 从此时起，在该 stream 上提交的所有 GPU 操作不再执行
   - 而是被记录到一个 CUDA Graph 对象中

2. 执行 run_once() (即 model.forward)
   - forward 中发出的每一个 kernel launch 被记录
   - 每一个内存操作被记录
   - kernel 之间的依赖关系被记录
   - 但这些 kernel 并不真正执行 (或执行后回滚)

3. with 块结束时，CUDA 驱动自动:
   - 退出 stream capture 模式
   - 对录制的 kernel 序列进行优化 (instantiate)
   - 生成可执行图 (executable graph)
   - 将 graph 句柄存入 graph 对象

时间线:
  with torch.cuda.graph(g):  <-- 开始录制
    |
    |  model.forward()
    |    -> layer1: kernel_A  被记录
    |    -> layer2: kernel_B  被记录
    |    -> layer3: kernel_C  被记录
    |    -> ...
    |    -> layerN: kernel_Z  被记录
    |
  # with 结束               <-- 录制完成 + 自动 instantiate
```

#### 6.2.4 录制完成后 `graph` 和 `output_buffers` 里存了什么？

```python
# capture_one_batch_size 返回后 (cuda_graph_runner.py:798-802)
graph, output_buffers = self.capture_one_batch_size(bs, forward, stream_idx)
self.graphs[bs] = graph
self.output_buffers[bs] = output_buffers
```

```
self.graphs[bs]:
  +-----------------------------------------------------+
  |  torch.cuda.CUDAGraph 对象                           |
  |                                                      |
  |  内部持有:                                           |
  |  - CUDA 驱动层的可执行图句柄 (cudaGraphExec_t)      |
  |  - 所有 kernel 的执行顺序和参数                      |
  |  - kernel 间的依赖关系图                             |
  |  - 预分配的中间张量显存地址                          |
  |                                                      |
  |  可以调用 .replay() 来重复执行                       |
  +-----------------------------------------------------+

self.output_buffers[bs]:
  +-----------------------------------------------------+
  |  LogitsProcessorOutput 对象                           |
  |                                                      |
  |  包含:                                               |
  |  - next_token_logits: Tensor [bs, vocab_size]       |
  |    (指向 buffers.next_token_logits_buffer[:bs])      |
  |  - full_logits: Tensor (可能为 None)                 |
  |  - hidden_states: Tensor (可能为 None)               |
  |                                                      |
  |  关键: 这些 tensor 指向预分配的 buffer 内存         |
  |  replay() 后，这些内存被自动更新                     |
  |  直接读取 output_buffers[bs] 就能拿到新结果         |
  +-----------------------------------------------------+
```

---

### 6.3 重放阶段 (`replay` -> `replay_prepare`)

> 源码位置: `cuda_graph_runner.py:1112-1249`

#### 6.3.1 `replay_prepare()` -- 数据拷贝到预分配 buffer

```python
# cuda_graph_runner.py:1112-1191
def replay_prepare(self, forward_batch, pp_proxy_tensors=None):
    buffers = self.buffers

    raw_bs = forward_batch.batch_size          # 实际请求数
    raw_num_token = raw_bs * self.num_tokens_per_bs

    # Padding: 找到 >= raw_bs 的最小已录制 batch size
    index = bisect.bisect_left(self.capture_bs, raw_bs)
    bs = self.capture_bs[index]  # 例: raw_bs=3 -> bs=4
```

**Padding 机制详解：**

```
bisect_left 的作用：

  capture_bs = [1, 2, 4, 8, 16, 32, 64, 128, 256]

  raw_bs = 3
    bisect_left([1,2,4,8,...], 3) = 2  (3 应该插入的位置)
    capture_bs[2] = 4                  -> 使用 bs=4 的 graph

  raw_bs = 5
    bisect_left([1,2,4,8,...], 5) = 3
    capture_bs[3] = 8                  -> 使用 bs=8 的 graph

  raw_bs = 32
    bisect_left([1,2,4,8,...], 32) = 5
    capture_bs[5] = 32                 -> 精确匹配
```

**数据拷贝 -- `populate_from_forward_batch()`**

```python
# cuda_graph_runner.py:1138-1151
buffers.populate_from_forward_batch(
    forward_batch=forward_batch,
    raw_bs=raw_bs,          # 实际请求数 (如 3)
    raw_num_token=raw_num_token,
    bs=bs,                  # padding 后的 bs (如 4)
    ...
)
```

`populate_from_forward_batch()` 内部 (`cuda_graph_runner.py:262-363`):

```python
def populate_from_forward_batch(self, *, forward_batch, raw_bs, raw_num_token, bs, ...):
    # 步骤 1: 如果需要 padding (bs > raw_bs)，先清空 padding 区域
    if bs != raw_bs:
        self.seq_lens.fill_(seq_len_fill_value)   # 填充 dummy 值
        self.out_cache_loc.zero_()                 # 清零
        # ... mamba, encoder 等同理 ...

    # 步骤 2: 构建 copy 列表 (批量拷贝)
    dsts = [
        self.input_ids[:raw_num_token],         # 目标: 预分配 buffer 的前 N 个
        self.req_pool_indices[:raw_bs],
        self.seq_lens[:raw_bs],
        self.out_cache_loc[:raw_num_token],
        self.positions[:raw_num_token],
    ]
    srcs = [
        forward_batch.input_ids,                # 源: 真实数据
        forward_batch.req_pool_indices,
        forward_batch.seq_lens,
        forward_batch.out_cache_loc,
        forward_batch.positions,
    ]

    # 步骤 3: 批量 GPU 拷贝 (按 dtype 分组优化)
    _grouped_foreach_copy_(dsts, srcs)

    # 步骤 4: CPU tensor 单独拷贝
    self.seq_lens_cpu[:raw_bs].copy_(forward_batch.seq_lens_cpu)
```

```
数据拷贝示意 (raw_bs=3, bs=4):

  forward_batch.input_ids:  [101, 205, 340]  (3 个真实 token id)

  buffers.input_ids:
  拷贝前: [0, 0, 0, 0, 0, 0, ...]  (max_num_token=256)
  拷贝后: [101, 205, 340, 0, 0, 0, ...]
           |<- raw_bs=3 ->|  |<- padding 区域: 保持为 0 (不会参与计算) ->|

  buffers.seq_lens:
  清空后: [fill, fill, fill, fill, 0, 0, ...]  (先全部填 dummy 值)
  拷贝后: [15, 42, 8, fill, 0, 0, ...]         (前 3 个是真实值)
           |<- raw_bs=3 ->|  |<- 第 4 个保持 dummy ->|

  这样 replay 时，Graph_4 会在 4 个位置上执行，
  但第 4 个位置的 dummy 数据不会影响最终输出。
  最终只取 output[:raw_bs] 作为有效结果。
```

#### 6.3.2 `replay()` -- 执行回放

```python
# cuda_graph_runner.py:1193-1249
def replay(self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None):
    self.deepep_adapter.replay()           # DeepEP MoE 适配器

    if not skip_attn_backend_init:
        self.replay_prepare(forward_batch, pp_proxy_tensors)  # 数据拷贝

    # 确定 graph key
    if self.enable_pdmux:
        graph_key = f"{get_current_stream_idx()}_{self.bs}"
    else:
        graph_key = self.bs               # 通常就是 bs 值

    # 核心操作: 一行代码完成整个 forward pass！
    self.graphs[graph_key].replay()       # <-- 这里就是魔法发生的地方

    # 读取输出
    output = self.output_buffers[graph_key]  # 直接引用预分配 buffer

    # 截取有效部分 (去除 padding)
    if isinstance(output, LogitsProcessorOutput):
        next_token_logits = output.next_token_logits[:self.raw_num_token]
        return LogitsProcessorOutput(
            next_token_logits=next_token_logits,
            hidden_states=output.hidden_states[:self.raw_num_token] if ... else None,
            ...
        )
```

#### 6.3.3 `graph.replay()` 为什么能加速？

```
普通 forward 执行 (无 CUDA Graph):
+-----------------------------------------------------------------+
| CPU 线程                     GPU                                 |
|                                                               |
| launch kernel_A ------->  [kernel_A 执行]                      |
|  (等返回)                     |                                 |
| launch kernel_B ------->       [kernel_B 执行]                  |
|  (等返回)                           |                           |
| launch kernel_C ------->             [kernel_C 执行]            |
|  (等返回)                                   |                   |
| ...                                          [kernel_N 执行]    |
|                                                               |
| 总时间 = 所有 kernel 执行时间 + N * launch_overhead            |
| launch_overhead 约 5-10us, N 可达数百                          |
+-----------------------------------------------------------------+

graph.replay() 执行:
+-----------------------------------------------------------------+
| CPU 线程                     GPU                                 |
|                                                               |
| graph.replay() ------->  [A->B->C->...->N 连续执行]             |
|  (一次调用)                无间隙，GPU 全速运转                  |
|                                                               |
| 总时间 = 所有 kernel 执行时间 (无额外开销)                      |
| 节省 = N * launch_overhead                                     |
| 当 N=200, overhead=8us 时: 节省 ~1.6ms/次                      |
| decode 阶段每秒可能执行 100+ 次: 节省 ~160ms/s                  |
+-----------------------------------------------------------------+
```

---

### 6.4 完整调用链

#### 6.4.1 从请求到 CUDA Graph replay 的完整路径

```
用户请求 (generate/sampling)
  |
  v
Scheduler.get_next_batch_to_run()          # 调度器组装 batch
  |
  v
Scheduler.run_batch()
  |
  v
ModelRunner.forward(forward_batch)         # model_runner.py:2882
  |
  +-- self.forward_pass_id += 1            # 递增 forward 计数
  +-- expert_distribution_recorder 记录
  |
  v
ModelRunner._forward_raw(forward_batch)    # model_runner.py:2941
  |
  +-- 检查 forward_mode.is_cuda_graph()    # 是否 decode 模式
  +-- 检查 self.graph_runner 存在          # CUDA Graph Runner 是否初始化
  +-- self.graph_runner.can_run(batch)     # 当前 batch 能否用 graph
  |     |
  |     +-- 检查 batch_size 是否在支持范围   # cuda_graph_runner.py:666
  |     +-- 检查 encoder_lens 是否支持
  |     +-- 检查 capture_hidden_mode 匹配
  |     +-- 检查 TBO/ngram 等特性兼容性
  |
  v (can_run_graph = True)
CudaGraphRunner.replay(forward_batch)      # cuda_graph_runner.py:1193
  |
  +-- deepep_adapter.replay()
  +-- replay_prepare(forward_batch)        # cuda_graph_runner.py:1112
  |     |
  |     +-- recapture_if_needed(batch)     # 检查是否需要重新录制
  |     +-- bisect_left 找到 padding bs   # 确定使用哪个 graph
  |     +-- populate_from_forward_batch()  # 数据拷贝到预分配 buffer
  |     +-- attn_backend.init_forward_metadata_replay_cuda_graph()
  |
  +-- self.graphs[bs].replay()             # <-- 一行代码执行整个 forward!
  |
  +-- 截取 output[:raw_bs] 有效部分
  |
  v
返回 LogitsProcessorOutput
  |
  v
ModelRunnerOutput(logits_output=ret, can_run_graph=True)
  |
  v
Scheduler 继续后续处理 (sampling, detokenize)
```

#### 6.4.2 调用链代码位置速查

```
阶段           方法                            文件:行号
-----------    ----------------------------    -----------------------
调度入口       Scheduler.run_batch()            scheduler.py
Forward 入口  ModelRunner.forward()            model_runner.py:2882
分发判断       ModelRunner._forward_raw()       model_runner.py:2941
  |            can_run_graph 检查               model_runner.py:2954-2958
  |            graph_runner.replay() 调用       model_runner.py:2967
  v
Graph Replay   CudaGraphRunner.replay()         cuda_graph_runner.py:1193
  |            deepep_adapter.replay()          cuda_graph_runner.py:1199
  |            replay_prepare()                 cuda_graph_runner.py:1202
  v
Replay 准备    replay_prepare()                 cuda_graph_runner.py:1112
  |            recapture_if_needed()            cuda_graph_runner.py:1118
  |            bisect_left 找 padding bs        cuda_graph_runner.py:1133-1136
  |            populate_from_forward_batch()    cuda_graph_runner.py:1138-1151
  |            attn_backend.init_replay          cuda_graph_runner.py:1174-1183
  v
核心 Replay    self.graphs[bs].replay()         cuda_graph_runner.py:1221
  |
  v
结果截取       output[:raw_num_token]           cuda_graph_runner.py:1234-1248
```

---

### 6.5 完整生命周期流程图

```
======================== 初始化阶段 (服务启动时，一次性) ========================

ModelRunner.__init__()
  |
  v
CudaGraphRunner.__init__()                         [cuda_graph_runner.py:515]
  |
  +-- self.graphs = {}                             [cuda_graph_runner.py:520]
  +-- self.output_buffers = {}                     [cuda_graph_runner.py:521]
  |
  +-- capture_bs = get_batch_sizes_to_capture()    [cuda_graph_runner.py:572]
  |     例: [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
  |
  +-- max_bs = max(capture_bs)                     [cuda_graph_runner.py:584]
  +-- self.buffers = DecodeInputBuffers.create(    [cuda_graph_runner.py:624]
  |     max_bs=256, max_num_token=256, ...
  |   ) -> 在 GPU 上分配所有固定 buffer
  |
  +-- self.capture()                               [cuda_graph_runner.py:651]
        |
        +-- freeze_gc()                            [cuda_graph_runner.py:807]
        +-- graph_capture() 获取 stream            [cuda_graph_runner.py:809]
        |
        +-- for bs in REVERSED(capture_bs):        [cuda_graph_runner.py:778]
        |     [256, 192, 128, 96, 64, 48, 32, 16, 8, 4, 2, 1]
        |     |
        |     +-- capture_one_batch_size(bs)        [cuda_graph_runner.py:798]
        |           |
        |           +-- buffers.input_ids[:bs]      [cuda_graph_runner.py:873]
        |           +-- 构建 ForwardBatch            [cuda_graph_runner.py:972]
        |           +-- attn_backend.init_capture   [cuda_graph_runner.py:1019]
        |           |
        |           +-- 定义 run_once()             [cuda_graph_runner.py:1030]
        |           |     -> forward(input_ids, positions, batch)
        |           |
        |           +-- Warm-up: 跑 2 遍            [cuda_graph_runner.py:1065]
        |           |     synchronize() + barrier()
        |           |     run_once()
        |           |     synchronize() + barrier()
        |           |     run_once()
        |           |
        |           +-- 获取全局显存池              [cuda_graph_runner.py:1070]
        |           |
        |           +-- _capture_graph()            [cuda_graph_runner.py:1074]
        |           |     with torch.cuda.graph(g, pool, stream):
        |           |         out = run_once()      [cuda_graph_runner.py:853]
        |           |         ^-- 所有 GPU kernel 被录制
        |           |
        |           +-- self.graphs[bs] = graph     [cuda_graph_runner.py:801]
        |           +-- self.output_buffers[bs] = out [cuda_graph_runner.py:802]
        |
        +-- 所有 bs 录制完成，服务就绪

======================== 推理阶段 (每次请求循环) ========================

每次 decode step:
  |
  v
ModelRunner.forward(forward_batch)                 [model_runner.py:2882]
  |
  v
ModelRunner._forward_raw(forward_batch)            [model_runner.py:2941]
  |
  +-- can_run_graph = graph_runner.can_run(batch)  [model_runner.py:2954]
  |     检查: bs 在范围内, 模式匹配, 特性兼容
  |
  v (can_run = True)
CudaGraphRunner.replay(forward_batch)              [cuda_graph_runner.py:1193]
  |
  +-- replay_prepare(forward_batch)                [cuda_graph_runner.py:1202]
  |     |
  |     +-- raw_bs = batch.batch_size (如 3)       [cuda_graph_runner.py:1120]
  |     +-- index = bisect_left(capture_bs, 3) = 2 [cuda_graph_runner.py:1135]
  |     +-- bs = capture_bs[2] = 4 (padding)       [cuda_graph_runner.py:1136]
  |     |
  |     +-- populate_from_forward_batch()           [cuda_graph_runner.py:1138]
  |     |     - 清空 padding 区域 (seq_lens.fill_)
  |     |     - copy 真实数据到 buffer[:raw_bs]
  |     |     - _grouped_foreach_copy_ 批量 GPU 拷贝
  |     |
  |     +-- attn_backend.init_replay_metadata       [cuda_graph_runner.py:1174]
  |
  +-- self.graphs[4].replay()                       [cuda_graph_runner.py:1221]
  |     ^-- GPU 一次性执行 bs=4 录制的所有 kernel
  |     ^-- CPU 只发这一个指令，无需逐步 launch
  |
  +-- output = self.output_buffers[4]               [cuda_graph_runner.py:1222]
  |     ^-- 指向预分配 buffer，replay 后自动更新
  |
  +-- 截取 output[:raw_bs=3] 作为有效结果           [cuda_graph_runner.py:1234]
  |     丢弃 padding 位置的输出
  |
  v
返回 LogitsProcessorOutput(next_token_logits, ...)

--- 一次 decode step 完成，耗时 ~1ms (比无 graph 快 ~30%) ---
```

---

## 参考资料

- [PyTorch CUDA Graph 官方文档](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- [CUDA Graph API 官方文档](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html)
- sglang 源码: `python/sglang/srt/model_executor/cuda_graph_runner.py`
- sglang 源码: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`
- sglang 源码: `python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`
- sglang 源码: `python/sglang/srt/model_executor/model_runner.py`
