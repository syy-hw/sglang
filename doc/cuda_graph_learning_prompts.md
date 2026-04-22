# CUDA Graph 渐进式学习 Prompt

> 这是一套为 CUDA Graph 零基础开发者设计的三阶段学习 prompt。
> 每个阶段都是独立的 prompt，你可以逐个复制到 Claude 对话中使用。
> 建议在 sglang 项目目录下运行，这样 Claude 可以读取源码辅助解释。

---

## 阶段一：CUDA Graph 基础概念 + sglang 原始流程

### Prompt 1.1：理解 CUDA Graph 是什么

```
我正在学习 sglang 项目中的 CUDA Graph 机制，我之前完全没有接触过 CUDA Graph。

请从最基础的概念讲起：

1. **什么是 CUDA Graph？** 用生活中的比喻解释（比如录音机、蓝图等）
   - 为什么需要它？（CPU→GPU 调度开销）
   - 它解决了什么问题？
   - 它的核心工作流程：capture → instantiate → replay

2. **CUDA Graph 的基本约束**
   - 为什么只能录制固定的 GPU 操作？
   - 为什么不支持条件分支和动态形状？
   - "所有操作必须在同一个 CUDA stream 上"意味着什么？

3. **PyTorch 中的 CUDA Graph API**（结合代码示例）
   - `torch.cuda.CUDAGraph()` 是什么
   - `torch.cuda.graph()` 上下文管理器怎么用
   - 给一个最小可运行的 demo（5-10 行代码），展示 capture → replay 的完整流程

请用中文回答，用代码示例和图示说明。
将回答放到D:\work\sglang\main\sglang\doc\CUDA Graph 完全入门指南.md文件下。
```

### Prompt 1.2：sglang 中原始的 CUDA Graph 流程（不含 PCG/BCG）

```
我已经理解了 CUDA Graph 的基本概念。现在请帮我理解 sglang 中最原始的 CUDA Graph 实现，
即不使用 PCG（Piecewise）和 BCG（Breakable）的流程。

请分析 `python/sglang/srt/model_executor/cuda_graph_runner.py` 中的以下关键方法，
用中文逐步讲解：

1. **初始化阶段** (`__init__`)
   - `self.graphs = {}` 和 `self.output_buffers = {}` 的作用
   - `capture_bs` 是什么？为什么需要多个 batch size？
   - `DecodeInputBuffers.create(...)` 做了什么？为什么要预分配 buffer？

2. **录制阶段** (`capture` → `capture_one_batch_size` → `_capture_graph`)
   - 为什么从大到小录制？（`reversed(self.capture_bs)`）
   - `run_once()` 函数体做了什么？
   - 为什么 warmup 要跑两遍再 capture？
   - `_capture_graph` 中 `with torch.cuda.graph(...)` 这一行发生了什么？
   - 录制完成后 `graph` 和 `output_buffers` 里存了什么？

3. **重放阶段** (`replay` → `replay_prepare`)
   - `replay_prepare` 中数据是怎么复制到预分配 buffer 的？
   - `graph.replay()` 为什么能加速？
   - 如果实际 batch size 不在录制列表中怎么办？（padding 机制）

4. **调用链**
   - 画出从 `ModelRunner._forward_raw` 到 `CudaGraphRunner.replay` 的完整调用链

请画出完整的生命周期流程图（ASCII 或 Mermaid），标注每个关键步骤的代码位置（文件:行号）。
将回答放到D:\work\sglang\main\sglang\doc\CUDA Graph 完全入门指南.md文件里面，尾段插入。
```

### Prompt 1.3：动手实验

```
基于我对 sglang CUDA Graph 的理解，请帮我设计一个思想实验来验证我的理解：

假设我们有一个简单的模型，batch size 列表是 [1, 2, 4]。

1. 请列出完整的录制流程：
   - 第一次录制 bs=4 时，创建了哪些 buffer？录制了哪些 GPU kernel？
   - 第二次录制 bs=2 时，复用了什么？
   - 第三次录制 bs=1 时，复用了什么？

2. 请列出一次完整的推理流程：
   - 假设来了一个 bs=3 的请求
   - replay_prepare 做了什么？（padding 到 4）
   - replay 时哪些 buffer 被写入了数据？
   - 输出是怎么从 output_buffers 中截取的？

3. **限制和问题**
   - 这种方式的限制是什么？（固定 batch size、不支持动态控制流）
   - 当模型中有 if/else 分支时会发生什么？
   - 这引出了为什么需要 PCG 和 BCG

请用中文回答，用表格和图示说明。
将回答放到D:\work\sglang\main\sglang\doc\CUDA Graph 完全入门指南.md文件里面，尾段插入。
```

---

## 阶段二：Piecewise CUDA Graph (PCG)

### Prompt 2.1：为什么需要 PCG

```
我已经理解了 sglang 中原始 CUDA Graph 的流程。现在我想理解为什么需要 PCG。

1. **torch.compile 的图分割问题**
   - 当 sglang 启用 `--enable-torch-compile` 时，模型会被 torch.compile 编译
   - torch.compile 使用 FX graph 来表示模型
   - 但有些操作（如动态控制流、Python 函数调用）不能被 FX 捕获
   - FX graph 需要在这些"断点"处被分割成多个子图
   - 请解释 `torch.fx.passes.split_module.split_module()` 的作用

2. **分割后的挑战**
   - 分割成多个子图后，怎么给每个子图分别做 CUDA Graph？
   - 每个子图需要独立的 `torch.cuda.CUDAGraph` 对象吗？
   - 子图之间怎么传递数据？

3. **PCG vs 原始方式的架构差异**
   - 原始方式：整个 forward → 一个 CUDAGraph
   - PCG：forward → FX split → 多个子图 → 每个子图一个 CUDAGraph
   - 请画图说明这个架构差异

请分析以下文件来回答：
- `python/sglang/srt/compilation/backend.py`（splitGraph 方法）
- `python/sglang/srt/compilation/cuda_piecewise_backend.py`
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

用中文回答。
```

### Prompt 2.2：PCG 的核心实现

```
请深入分析 PCG 的核心实现，重点关注以下文件和代码段：

1. **`cuda_piecewise_backend.py` 中的 `CUDAPiecewiseBackend`**
   - `__call__` 方法的完整流程：接收输入 → 查找/创建 runnable → cudagraph replay
   - `ConcreteSizeEntry` 的结构：为什么需要按 size 缓存？
   - 第一次调用（compile + capture）vs 后续调用（replay）的区别

2. **`piecewise_cuda_graph_runner.py` 中的 `PiecewiseCudaGraphRunner`**
   - `capture()` 方法的流程
   - `graph_capture()` 上下文管理器的作用
   - `set_pcg_capture_stream(stream)` 为什么需要特殊的 stream？
   - `capture_num_tokens` 列表是怎么确定的？

3. **录制和重放的对比**
   - 原始方式录制一次完整的 forward
   - PCG 录制每个子图的 forward
   - 请画出一个包含 3 个子图的 PCG 录制和重放流程

4. **关键设计决策**
   - 为什么 PCG 使用 `torch.cuda.CUDAGraph` 而不是底层的 CUDA Runtime API？
   - 这对 NPU 适配有什么好处？

请用中文回答，画出详细的流程图。
```

### Prompt 2.3：NPU 适配 — PCG 的跨平台优势

```
请分析 PCG 如何适配华为 Ascend NPU，重点关注：

1. **`npu_piecewise_backend.py` 的实现**
   - 它继承自 `CUDAPiecewiseBackend`，改了哪些地方？
   - `torch.cuda.*` 被替换成了什么？
   - 改动量有多大？（提示：大约 3 行）

2. **为什么 PCG 容易适配 NPU**
   - PCG 只使用了 PyTorch 高层 API（`torch.cuda.CUDAGraph`、`torch.cuda.graph`）
   - torch_npu 提供了等效的 `torch.npu.CUDAGraph`、`torch.npu.graph`
   - 请解释 PyTorch 的设备抽象层如何使跨平台适配变得简单

3. **NPU 与 CUDA Graph 等效 API 对照表**
   | 功能 | CUDA (PyTorch) | NPU (torch_npu) |
   |------|---------------|-----------------|
   | Graph 对象 | torch.cuda.CUDAGraph() | torch.npu.CUDAGraph() |
   | 录制上下文 | torch.cuda.graph() | torch.npu.graph() |
   | ... | ... | ... |

请用中文回答。
```

---

## 阶段三：Breakable CUDA Graph (BCG)

### Prompt 3.1：为什么需要 BCG — 从 PCG 的局限说起

```
我已经理解了 PCG。现在请帮我理解为什么还需要 BCG。

1. **PCG 的局限**
   - PCG 的分割发生在编译时（FX graph split），需要 torch.compile 参与
   - 如果不启用 torch.compile，就不能使用 PCG
   - 有些操作（如动态 shape、复杂控制流）FX 无法处理
   - 请举例说明 PCG 无法处理的场景

2. **BCG 的核心思想**
   - 不在编译时分割图，而是在运行时分割 CUDA stream capture
   - 使用底层 CUDA Runtime API（`cudaStreamBeginCapture`/`cudaStreamEndCapture`）
   - 在同一个 stream 上反复 begin/end capture，形成多个图段
   - 遇到不能录制的操作时，结束当前段，执行操作，再开始新段

3. **比喻理解**
   - 原始 CUDA Graph：一镜到底的连续镜头，中间不能停
   - PCG：把剧本分成几幕，每幕独立拍摄
   - BCG：边拍边停，遇到问题就暂停录制，处理完后继续录
   - 请展开解释这个比喻

用中文回答。
```

### Prompt 3.2：BCG 的核心实现

```
请深入分析 BCG 的核心实现文件
`python/sglang/srt/model_executor/breakable_cuda_graph/breakable_cuda_graph.py`。

1. **核心数据结构**
   - `GraphBreakInfo`：存储什么？（func, output, graph_handle）
   - `_captured_graphs_var`：ContextVar 的作用，为什么用 ContextVar 而不是普通变量？
   - `_current_stream_var` 和 `_forked_streams_var` 的作用

2. **录制流程**（`BreakableCUDAGraphCapture.__enter__` → `capture_begin` → ... → `capture_end`）
   - `capture_begin()`：为什么先调用 `super().capture_begin()` 再 `_end_capture_segment`？
   - `eager_on_graph` 装饰器的工作原理：
     - 检测到在录制中 → `_end_capture_segment` → 执行函数 → 记录 `GraphBreakInfo` → `_begin_capture_segment`
   - `capture_end()`：实例化所有段，形成 `self._exec` 列表

3. **重放流程**（`BreakableCUDAGraph.replay`）
   ```
   for func, _, handle in self._exec:
       _replay_graph(handle, ...)  # 重放一个段
       func()                       # 执行 eager 函数
   _replay_graph(self.last_graph_exec, ...)  # 重放最后一个段
   ```
   - 这个交替重放的设计为什么要这样做？

4. **Stream Hook 机制**（`_hooked_wait_stream`）
   - 为什么要 monkey-patch `torch.cuda.Stream.wait_stream`？
   - Fork 和 Join 分别处理了什么场景？
   - 录制期间 stream 之间的同步有什么特殊问题？

请画出 BCG 的完整录制和重放流程图（ASCII），标注代码行号。
用中文回答。
```

### Prompt 3.3：BCG 在 sglang 中的集成

```
请分析 BCG 在 sglang 中的集成方式，关注以下代码：

1. **触发条件**（`server_args.py`）
   - `--debug-cuda-graph` 参数如何触发 BCG？
   - 它设置了 `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1` 环境变量
   - 为什么同时设置 `disable_piecewise_cuda_graph = True`？
   - BCG 和 PCG 互斥吗？为什么？

2. **集成到 CudaGraphRunner**（`cuda_graph_runner.py`）
   - `_create_device_graph()`：BCG 创建 `BreakableCUDAGraph`，普通创建 `torch.cuda.CUDAGraph`
   - `_capture_graph()`：
     - BCG 使用 `BreakableCUDAGraphCapture` 代替 `torch.cuda.graph`
     - `debug_cuda_graph` 时用 `eager_on_graph(True)` 包装 `run_once_fn`
   - 请解释 `eager_on_graph(True)(run_once_fn)` 的效果

3. **当前使用场景**
   - BCG 目前主要用于 `--debug-cuda-graph` 调试模式
   - 这个模式下，整个 forward 函数被 `eager_on_graph` 包装
   - 效果是每个操作都打断图，全部 eager 执行
   - 这相当于在 CUDA Graph 的框架下走 eager 路径，便于调试

4. **未来潜力**
   - 选择性地对某些操作使用 `@eager_on_graph(True)`
   - 实现 eager + graph 混合执行
   - 对比 PCG 的编译时分割 vs BCG 的运行时分割

请用中文回答，画出调用链流程图。
```

### Prompt 3.4：BCG 的 NPU 适配挑战

```
请分析 BCG 适配华为 Ascend NPU 的挑战：

1. **核心依赖：`cuda-python` 包**
   - BCG 使用的底层 API：
     - `cuda.bindings.runtime.cudaStreamBeginCapture`
     - `cuda.bindings.runtime.cudaStreamEndCapture`
     - `cuda.bindings.runtime.cudaGraphInstantiateWithFlags`
     - `cuda.bindings.runtime.cudaGraphLaunch`
     - `cuda.bindings.runtime.cudaGraphDestroy`
   - 这些是 CUDA Runtime C API 的 Python 绑定
   - 没有任何 NPU 等效包

2. **为什么 BCG 比 PCG 难适配**
   - PCG：只用 PyTorch 高层 API → torch_npu 提供等效实现
   - BCG：用底层 CUDA C API → 没有 NPU 等效实现
   - 需要华为提供 `npu.bindings.runtime` 等效包

3. **Stream Capture 的 NPU 支持情况**
   - 请搜索华为 Ascend NPU 是否支持类似 CUDA Stream Capture 的功能
   - torch_npu 中是否有 `npuStreamBeginCapture`/`npuStreamEndCapture` API
   - 华为是否提供了 CUDA Graph 等效的编程接口

4. **适配方案建议**
   - 方案 A：等待华为提供底层 stream capture API
   - 方案 B：在 torch_npu 层面实现等效的 `BreakableNPUGraph`
   - 方案 C：在 sglang 中为 NPU 回退到 PCG 方案

请用中文回答，给出对比表格。
```

---

## 阶段四（可选）：综合对比

### Prompt 4.1：三者对比总结

```
请帮我做一个完整的对比总结，涵盖原始 CUDA Graph、PCG、BCG 三个方案：

1. **架构对比表**
   | 维度 | 原始 CUDA Graph | PCG | BCG |
   |------|----------------|-----|-----|
   | 分割时机 | 无分割 | 编译时 (FX split) | 运行时 (stream capture) |
   | 使用的 API | torch.cuda.CUDAGraph | torch.cuda.CUDAGraph | cuda.bindings.runtime |
   | 需要 torch.compile | 否 | 是 | 否 |
   | NPU 适配难度 | 低 | 低 | 高 |
   | ... | ... | ... | ... |

2. **各自的适用场景**
   - 原始：模型无动态控制流，不需要 torch.compile
   - PCG：需要 torch.compile 优化，FX 可分割
   - BCG：调试模式 / 未来用于运行时动态分割

3. **性能对比**
   - 录制开销：三者各有什么开销？
   - 重放性能：每段的 replay 有额外开销吗？
   - 内存使用：三者的内存模型有什么区别？

4. **选择指南**
   - 什么时候用原始方式？
   - 什么时候用 PCG？
   - 什么时候需要 BCG？
   - NPU 场景下应该选择哪个？

请用中文回答，给出决策树图。
```

---

## 使用建议

1. **学习顺序**：严格按 1.1 → 1.2 → 1.3 → 2.1 → 2.2 → 2.3 → 3.1 → 3.2 → 3.3 → 3.4 → 4.1
2. **每个 prompt 独立**：每个 prompt 都可以独立使用，不需要前一个的上下文
3. **建议环境**：在 sglang 项目目录下运行，这样 Claude 可以直接读取源码
4. **互动学习**：如果某个概念不理解，可以追问"请用更简单的方式解释 XXX"
5. **验证理解**：每学完一个阶段，可以用 "请出几道题测试我对 XXX 的理解" 来自测
