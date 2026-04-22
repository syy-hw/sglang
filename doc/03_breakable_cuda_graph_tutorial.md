# Breakable CUDA Graph (BCG) 深入教程 + 三者对比

> 生成日期: 2026-04-21
> 基于 commit: `c25f00630`
> 参考文档: [breakable_cuda_graph.md](../docs/advanced_features/breakable_cuda_graph.md), [pcg_vs_bcg_analysis.md](../docs/advanced_features/pcg_vs_bcg_analysis.md)

---

## 目录

- [一、核心创新点：Graph Break 机制](#一核心创新点graph-break-机制)
  - [1. 问题：标准 CUDA Graph 不可调试且不支持不兼容算子](#1-问题标准-cuda-graph-不可调试且不支持不兼容算子)
  - [2. BCG 的解法：在 CUDA Graph 中插入"断点"](#2-bcg-的解法在-cuda-graph-中插入断点)
  - [3. 代码入口](#3-代码入口)
- [二、`@eager_on_graph` 装饰器](#二eager_on_graph-装饰器)
  - [1. 问题：如何在特定位置插入 graph break](#1-问题如何在特定位置插入-graph-break)
  - [2. BCG 的解法：装饰器自动管理 capture/end/begin](#2-bcg-的解法装饰器自动管理-captureendbegin)
  - [3. 代码入口](#3-代码入口-1)
- [三、Captured Segment 和 Eager Segment 的交替执行](#三captured-segment-和-eager-segment-的交替执行)
  - [1. Capture 阶段的交替](#1-capture-阶段的交替)
  - [2. Replay 阶段的交替](#2-replay-阶段的交替)
  - [3. Intermediate Tensor 在 Segments 间的传递](#3-intermediate-tensor-在-segments-间的传递)
- [四、两大问题的技术方案](#四两大问题的技术方案)
  - [1. 调试：`--debug-cuda-graph`](#1-调试--debug-cuda-graph)
  - [2. 不兼容算子：`@eager_on_graph` 和 `break_graph()`](#2-不兼容算子eager_on_graph-和-break_graph)
- [五、三者对比表](#五三者对比表)

---

## 一、核心创新点：Graph Break 机制

### 1. 问题：标准 CUDA Graph 不可调试且不支持不兼容算子

标准 CUDA Graph 把整个 forward pass 录成一个"黑盒"，带来两个问题：

**问题 1 — 调试困难**：当 captured graph 内出现错误（输出不对、数值不匹配、crash），无法在 graph 内部打断点或加 print，因为 graph 作为一个整体 replay。

**问题 2 — 不兼容算子**：某些操作无法被 CUDA Graph 捕获：
- 动态控制流（`if tensor_value > 0`）
- Host-device 同步（`torch.cuda.synchronize()`, `.item()`）
- JIT 编译触发点
- 跨 iteration 行为变化的 op

唯一的替代方案是**完全禁用 CUDA Graph**，但这样所有 kernel launch overhead 又回来了。

### 2. BCG 的解法：在 CUDA Graph 中插入"断点"

**类比解释**：想象一盘录制好的录像带。BCG 的做法是：把录像带剪成几段，每段之间你可以暂停、手动操作、再继续播下一段。大部分时间还在看录像（graph replay），但关键节点你可以介入。

**技术原理**：BCG 利用 CUDA stream capture 的底层 API：
- 在 capture 过程中，多次调用 `cudaStreamBeginCapture` / `cudaStreamEndCapture`
- 每对 begin/end 之间产生一个 **captured segment**（`cudaGraph_t`）
- 段之间执行 **eager 函数**（Python 代码）
- 大部分 kernel launch overhead 仍然被 graph 消除

```
标准 CUDA Graph:
  [=========== 一整段 captured graph ===========]

BCG:
  [segment 0] -- eager func -- [segment 1] -- eager func -- [segment N]
```

### 3. 代码入口

| 组件 | 文件 | 行号 |
|------|------|------|
| `BreakableCUDAGraph` 类 | `breakable_cuda_graph/breakable_cuda_graph.py` | 264-314 |
| `BreakableCUDAGraphCapture` 上下文管理器 | `breakable_cuda_graph/breakable_cuda_graph.py` | 317-345 |
| `_begin_capture_segment` | `breakable_cuda_graph/breakable_cuda_graph.py` | 162-168 |
| `_end_capture_segment` | `breakable_cuda_graph/breakable_cuda_graph.py` | 146-159 |

---

## 二、`@eager_on_graph` 装饰器

### 1. 问题：如何在特定位置插入 graph break

用户需要一种简单的方式标记"这个函数不能被 graph capture"。手动管理 `cudaStreamBegin/EndCapture` 太底层且容易出错。

### 2. BCG 的解法：装饰器自动管理 capture/end/begin

**类比解释**：`@eager_on_graph` 就像在录像带上贴一个标签："到这里暂停，手动操作，然后继续录"。

**实现分析**（`breakable_cuda_graph.py:225-261`）：

```python
def eager_on_graph(enable: bool):
    def decorator(inner: Callable):
        if not enable:
            return inner                    # enable=False: no-op

        def wrapper(*args, **kwargs):
            stream = get_current_stream()
            if not _is_capturing(stream.cuda_stream):
                return inner(*args, **kwargs)  # 非 capture 期间: 直接执行

            # ---- capture 期间的 graph break 逻辑 ----
            last_graph = _end_capture_segment(stream)   # :234 结束当前 segment
            output = inner(*args, **kwargs)              # :237 eager 执行

            def replay_fn():                              # :247-249 创建闭包
                new_out = captured_inner(*captured_args, **captured_kwargs)
                return _copy_output(captured_output, new_out)

            captured_graphs.append(GraphBreakInfo(replay_fn, output, last_graph))
            _begin_capture_segment(stream)               # :256 开始新 segment
            return output
        return wrapper
    return decorator
```

**三条执行路径**：

| 情况 | 行为 | 行号 |
|------|------|------|
| `enable=False` | 装饰器是 no-op，返回原函数 | 227-228 |
| 非 capture 期间 | 直接调用原函数 | 232-233 |
| Capture 期间 | 触发 graph break 逻辑 | 234-256 |

**闭包设计的关键**：`replay_fn` 捕获的是参数的**引用**（不是值的副本）。这些引用指向 CUDA graph 的静态 input/output buffer。replay 时 buffer 内容已更新为新数据，所以 eager 函数能看到正确的输入。

### 3. 代码入口

| 组件 | 文件 | 行号 |
|------|------|------|
| `eager_on_graph()` 装饰器 | `breakable_cuda_graph/breakable_cuda_graph.py` | 225-261 |
| `GraphBreakInfo` NamedTuple | `breakable_cuda_graph/breakable_cuda_graph.py` | 46-52 |
| `_copy_output` 三种处理策略 | `breakable_cuda_graph/breakable_cuda_graph.py` | 191-222 |

---

## 三、Captured Segment 和 Eager Segment 的交替执行

### 1. Capture 阶段的交替

```
BreakableCUDAGraphCapture.__enter__()             :333-338
  +-- _install_wait_stream_hook()                 :334  hook wait_stream 追踪 fork/join
  +-- _captured_graphs_var.set([])                :335  初始化 break 列表
  +-- super().__enter__()                         :338
        |
        +-- BreakableCUDAGraph.capture_begin()    :269-275
              +-- super().capture_begin()          PyTorch 启动 capture
              +-- _end_capture_segment()           立即结束 (PyTorch 的)
              +-- _begin_capture_segment()         BCG 开始自己的 capture

captured_fn()  (用户函数执行)
  |
  +-- [CUDA ops 录入 segment 0]
  |
  +-- 遇到 @eager_on_graph:
  |     +-- _end_capture_segment()         segment 0 结束 -> graph_0
  |     +-- eager 执行函数                 产出 output
  |     +-- GraphBreakInfo 追加到列表
  |     +-- _begin_capture_segment()       segment 1 开始
  |
  +-- [更多 CUDA ops 录入 segment 1]
  ...

BreakableCUDAGraphCapture.__exit__()             :340-345
  +-- BreakableCUDAGraph.capture_end()           :277-290
        +-- _end_capture_segment()               最后一个 segment -> graph_N
        +-- _instantiate_graph(graph_N)           实例化
        +-- for break: _instantiate_graph()       实例化每个 break 的 graph
        +-- _begin_capture_segment()              dummy capture (欺骗 PyTorch)
        +-- super().capture_end()                 PyTorch 正常结束
```

**"三明治"适配技巧**：BCG 继承 `torch.cuda.CUDAGraph`，在 `capture_begin` 中让 PyTorch 先启动 capture，然后立即用 BCG 自己的 capture 替代。在 `capture_end` 中，启动一个 dummy capture 让 PyTorch 能正常结束。这让 BCG 完全复用 PyTorch 的 `torch.cuda.graph` 上下文管理器。

### 2. Replay 阶段的交替

```
BreakableCUDAGraph.replay()                       :292-304
  |
  +-- 无 break 快速路径:
  |     _replay_graph(last_graph_exec)             :296-298
  |     (等价于标准 CUDA Graph)
  |
  +-- 有 break 交替路径:
        for (replay_fn, _, graph_exec) in self._exec:
          +-- _replay_graph(graph_exec, stream)    :300  launch segment_i
          +-- replay_fn()                          :301  eager 执行 + copy_output
        _replay_graph(last_graph_exec, stream)     :302  最后一个 segment
```

**完整交替流程图**：

```
Replay:
  |
  v
cudaGraphLaunch(segment_0_exec)  <-- 重放第一段 captured kernel
  |
  v
eager func replay:
  replay_fn()
    captured_inner(*args)         <-- args 指向已更新的静态 buffer
    _copy_output(dst, new_out)    <-- 结果原地写回静态 buffer
  |
  v
cudaGraphLaunch(segment_1_exec)  <-- 重放第二段
  |
  v
eager func replay:
  replay_fn()
    ...
  |
  v
... (交替进行) ...
  |
  v
cudaGraphLaunch(last_exec)       <-- 最后一段
  |
  v
完成
```

### 3. Intermediate Tensor 在 Segments 间的传递

**`_copy_output` 的三种处理策略**（`breakable_cuda_graph.py:191-222`）：

| 输出类型 | 处理方式 | 原因 |
|----------|----------|------|
| 纯 Tensor | `dst.copy_(src)` 原地拷贝 | 下游 graph segment 持有 dst 的引用 |
| 带 `__dict__` 的对象 | 遍历属性，tensor 属性 `copy_`，非 tensor 属性 `setattr` | 支持 dataclass/output 对象 |
| Dict | 遍历 key，tensor 值 `copy_`，非 tensor 值替换 | 支持结构化输出 |

**关键**：非 tensor 值直接替换是安全的，因为它们不参与 CUDA graph capture，只是 Python 级别的元数据。

---

## 四、两大问题的技术方案

### 1. 调试：`--debug-cuda-graph`

**入口**：`server_args.py:3722-3734`

```
--debug-cuda-graph
    |
    v
自动启用 BCG (SGLANG_USE_BREAKABLE_CUDA_GRAPH=1)
自动禁用 PCG (disable_piecewise_cuda_graph=True)
    |
    v
CudaGraphRunner._capture_graph()              cuda_graph_runner.py:825-851
    |
    +-- captured_fn = eager_on_graph(True)(run_once_fn)   :849
    |   (整个 model.forward 被包装为 eager segment!)
    |
    +-- with BreakableCUDAGraphCapture(...):
          captured_fn()  <-- 实际是 model.forward()
```

**Debug 模式下的 Capture**：

```
[segment_0: 空] -- eager: model.forward() -- [segment_last: 空]
```

两个空 segment 几乎无开销，但 `model.forward()` 完全 eager 执行，可以打断点、打印 tensor。

**仍在 CUDA Graph 的代码路径中**：数据流经相同的静态 buffer（`DecodeInputBuffers`），经过相同的 `replay_prepare` 逻辑。用户可以在 `model.forward` 内部任意位置调试，同时确保数据流路径与生产环境一致。

### 2. 不兼容算子：`@eager_on_graph` 和 `break_graph()`

**方式一**：`@eager_on_graph(enable=True)` 装饰器

```python
@eager_on_graph(enable=True)
def my_dynamic_op(x):
    return some_dynamic_operation(x)
```

**方式二**：`break_graph()` 函数（`breakable_cuda_graph.py:348-352`）

```python
# 只想在某个位置插入切割点，不执行任何 eager 计算
break_graph()
```

`break_graph()` 本质上是 `@eager_on_graph(True)` 装饰的空函数。

---

## 五、三者对比表

| 维度 | 标准 CUDA Graph | PCG | BCG |
|------|----------------|-----|-----|
| **核心问题** | Decode 阶段 kernel launch overhead | Extend 阶段动态 token 数 | 不可调试 + 不兼容算子 |
| **Graph 粒度** | 整个 forward pass 一个 graph | 按 layer 切分为多个 submod，每个独立 graph | 按 graph break 切分为多个 segment |
| **动态 shape 支持** | 不支持，需为每种 bs 分别 capture | 通过 split + padding 支持 | 不直接解决 shape 问题 |
| **适用阶段** | Decode（固定 bs，每次 1 token） | Extend/Prefill（动态 token 数） | Decode（增强标准 CG） |
| **性能收益来源** | 消除所有 kernel launch overhead | 消除分段内的 kernel launch overhead | 消除 captured segment 内的 overhead |
| **关键代码入口** | `cuda_graph_runner.py` | `piecewise_cuda_graph_runner.py` + `compilation/backend.py` | `breakable_cuda_graph/breakable_cuda_graph.py` |
| **设计取舍** | 简单但不灵活 | 复杂但覆盖 extend 场景 | 保留大部分性能，牺牲少量 break 开销 |
| **与 PyTorch 关系** | 直接使用 `torch.cuda.CUDAGraph` | 使用 `torch.compile` + 自定义 backend | 继承 `torch.cuda.CUDAGraph`，底层控制 capture |
| **可否与其它共存** | BCG 可替换它 | 独立于 decode 路径 | 替换 decode 路径的标准 CG |

**架构定位图**：

```
+-----------------------------------------------------------------+
|                        ModelRunner                               |
|                                                                  |
|   Decode 路径                   Extend 路径                      |
|   +-----------------------+    +---------------------------+    |
|   |  CudaGraphRunner      |    | PiecewiseCudaGraphRunner  |    |
|   |                       |    |                           |    |
|   |  标准 CG 或 BCG       |    |  PCG                      |    |
|   |  (环境变量选择)        |    |  (torch.compile + split)  |    |
|   +-----------------------+    +---------------------------+    |
|                                                                  |
|   选择逻辑:                                                     |
|   SGLANG_USE_BREAKABLE_CUDA_GRAPH=1  -> BCG on decode          |
|   --debug-cuda-graph                 -> BCG only, PCG disabled |
|   默认                                -> 标准 CG + PCG         |
+-----------------------------------------------------------------+
```
