# Breakable CUDA Graph (BCG) 零基础入门教程

> 生成日期: 2026-04-27
> 覆盖范围: PR #22218 重构后的最新实现
> 目标读者: 对 CUDA Graph 不了解但具备 PyTorch 基础的开发者

---

## 目录

- [第一章：CUDA Graph 是什么？](#第一章cuda-graph-是什么)
  - [1.1 先理解问题：Kernel Launch Overhead](#11-先理解问题kernel-launch-overhead)
  - [1.2 CUDA Graph 的解决方案](#12-cuda-graph-的解决方案)
  - [1.3 CUDA Graph 的限制](#13-cuda-graph-的限制)
- [第二章：BCG 的核心创新——Graph Break](#第二章bcg-的核心创新graph-break)
  - [2.1 "可断开的录像带"](#21-可断开的录像带)
  - [2.2 @eager_on_graph 装饰器](#22-eager_on_graph-装饰器)
  - [2.3 三种方案的结构对比](#23-三种方案的结构对比)
- [第三章：新版 BCG 的架构设计](#第三章新版-bcg-的架构设计)
  - [3.1 从继承到组合](#31-从继承到组合)
  - [3.2 核心类关系图](#32-核心类关系图)
  - [3.3 context.py：全局开关](#33-contextpy全局开关)
- [第四章：Capture 阶段——录制多段 Graph](#第四章capture-阶段录制多段-graph)
  - [4.1 从入口到录制完成](#41-从入口到录制完成)
  - [4.2 Segment 的 begin/end 循环](#42-segment-的-beginend-循环)
  - [4.3 共享 Mempool 与 weak_ref_tensor](#43-共享-mempool-与-weak_ref_tensor)
- [第五章：Replay 阶段——交替执行](#第五章replay-阶段交替执行)
  - [5.1 完整 Replay 流程](#51-完整-replay-流程)
  - [5.2 replay_prepare：数据搬运](#52-replay_prepare数据搬运)
  - [5.3 _copy_output：段间结果传递](#53-_copy_output段间结果传递)
- [第六章：在 Attention 层插入 Graph Break](#第六章在-attention-层插入-graph-break)
- [第七章：如何启用 BCG](#第七章如何启用-bcg)

---

## 第一章：CUDA Graph 是什么？

### 1.1 先理解问题：Kernel Launch Overhead

GPU 上的每次运算（矩阵乘法、激活函数、LayerNorm……）都需要 CPU 向 GPU **发射一个 kernel**。这个"发射"动作本身有开销——每次约 5-10 微秒。

听起来不多？但 LLM 的一次 forward pass 包含 **数百个 kernel**。在 decode 阶段（逐 token 生成），GPU 计算 1 个 token 只需几十微秒，但 CPU 发射 kernel 就花了数百微秒：

```
传统执行（无 CUDA Graph）:
  CPU -> GPU: launch kernel 1  (等待 ~5μs)
  CPU -> GPU: launch kernel 2  (等待 ~5μs)
  CPU -> GPU: launch kernel 3  (等待 ~5μs)
  ...
  CPU -> GPU: launch kernel N  (等待 ~5μs)
  总开销: N × 5μs（N 可达数百）

  问题：CPU 发射的开销 >> GPU 实际计算时间
```

这就是 **kernel launch overhead**——CPU 成为瓶颈，GPU 在等待。

### 1.2 CUDA Graph 的解决方案

CUDA Graph 的思路很直接：把整个 forward pass 录成一个"脚本"，之后每次直接重放整个脚本，跳过 CPU 逐个发射的开销。

```
CUDA Graph:
  录制阶段（只做一次）:
    记录 kernel 1 -> kernel 2 -> ... -> kernel N 的完整依赖图

  重放阶段（每次生成 token 时）:
    GPU 直接执行整张图，无需 CPU 介入

  效果：数百次 kernel launch 变成 1 次 graph launch
```

**类比**：想象你是一位厨师（CPU），要指挥助手（GPU）做一道有 50 个步骤的菜。每天三餐都要重复。

- **没有 Graph**：每步都要喊"切洋葱"、"开火"、"放盐"……嗓子疼（CPU 开销大）
- **有 Graph**：第一天把 50 步录成录像带，之后直接放录像带（1 次 launch）

### 1.3 CUDA Graph 的限制

CUDA Graph 不是万能的。它有几个硬性限制：

| 限制 | 原因 | 实际影响 |
|------|------|----------|
| **输入/输出地址必须固定** | Graph 录制的是"对这块内存做什么操作" | 需要预分配静态 buffer |
| **控制流不能变化** | Graph 是固定的执行路径，不能中途 if/else | 动态 token 数量难以处理 |
| **所有操作必须可捕获** | 某些操作（如 `.item()`、`torch.cuda.synchronize()`）涉及 CPU-GPU 同步 | 不兼容的算子必须绕过 |
| **shape 必须一致** | Graph 内的 tensor shape 在录制时就确定了 | 不同 batch size 需要分别录制 |

其中 **"所有操作必须可捕获"** 和 **"shape 必须一致"** 是最大的痛点。BCG 的出现正是为了解决这些问题。

---

## 第二章：BCG 的核心创新——Graph Break

### 2.1 "可断开的录像带"

BCG 的核心想法：**把一盘完整的录像带剪成几段，段之间可以手动操作**。

```
标准 CUDA Graph:
  [==================== 一整段录制 ====================]
  优点：所有 kernel overhead 全部消除
  缺点：中间不能做任何 CPU 操作

BCG:
  [segment 0] -- eager break -- [segment 1] -- eager break -- [segment 2]
  优点：大部分 kernel overhead 仍然消除
  优点：eager break 处可以执行任意 Python 代码
  代价：每个 break 有少量 CPU 介入开销
```

**技术本质**：在 CUDA stream capture 过程中，多次调用"开始录制"和"结束录制"，每对 begin/end 之间产生一个 captured segment（`torch.cuda.CUDAGraph` 对象）。

### 2.2 @eager_on_graph 装饰器

BCG 通过 `@eager_on_graph` 装饰器标记哪些函数需要在录制中插入断点。

**文件**: `breakable_cuda_graph/breakable_cuda_graph.py:198-237`

```python
def eager_on_graph(enable: bool):
    def decorator(inner: Callable):
        if not enable:
            return inner                       # enable=False: 无操作

        def wrapper(*args, **kwargs):
            capture = _current_capture_var.get()
            if capture is None:
                return inner(*args, **kwargs)   # 非录制期间: 直接执行

            # ---- 录制期间的 graph break 逻辑 ----
            capture._end_current_segment()      # 结束当前 segment
            output = inner(*args, **kwargs)     # eager 执行目标函数

            # 创建 replay 闭包（弱引用捕获参数）
            captured_args = tuple(_weak_ref_if_tensor(a) for a in args)
            captured_output = _weak_ref_if_tensor(output)

            def replay_fn():                    # 后续 replay 时调用
                new_out = inner(*captured_args, ...)
                return _copy_output(captured_output, new_out)

            capture.cuda_graph._break_fns.append(replay_fn)
            capture._begin_new_segment()        # 开始新的 segment
            return output
        return wrapper
    return decorator
```

**三条执行路径**：

| 场景 | 行为 | 行号 |
|------|------|------|
| `enable=False` | 装饰器是空操作，原样返回 | 200-201 |
| 非录制期间 | 直接调用原函数 | 204-206 |
| 录制期间 | 结束 segment → eager 执行 → 开始新 segment | 211-232 |

### 2.3 三种方案的结构对比

```
┌─────────────────────────────────────────────────────────────┐
│                   标准 CUDA Graph                            │
│   [=========== 一整段 graph（decode 专用）============]      │
│   适用: decode（固定 bs, 每次 1 token）                      │
│   限制: 不支持动态 shape, 不支持不兼容算子                    │
├─────────────────────────────────────────────────────────────┤
│                   PCG (Piecewise CUDA Graph)                 │
│   [submod_0] [submod_1] ... [submod_N]                      │
│   通过 torch.compile + FX graph split 实现                   │
│   适用: extend（动态 token 数）                               │
│   限制: 依赖 torch.compile, 编译链长                         │
├─────────────────────────────────────────────────────────────┤
│                   BCG (Breakable CUDA Graph)                 │
│   [seg_0] -- eager -- [seg_1] -- eager -- [seg_2]           │
│   通过 @eager_on_graph 装饰器在运行时插入 break               │
│   适用: extend（bs=1 的 prefill）                             │
│   优势: 不依赖 torch.compile, 纯 PyTorch 高层 API            │
└─────────────────────────────────────────────────────────────┘
```

---

## 第三章：新版 BCG 的架构设计

PR #22218 对 BCG 进行了架构级重构。理解这些设计决策有助于读懂后续代码。

### 3.1 从继承到组合

| 维度 | 旧版 BCG | 新版 BCG (PR #22218) |
|------|----------|---------------------|
| BreakableCUDAGraph | 继承 `torch.cuda.CUDAGraph` | 纯容器，**无继承** |
| Segment 管理 | 手动管理 CUDA C API 句柄 | `list[torch.cuda.CUDAGraph]` |
| Replay | `cudaGraphLaunch()` (C API) | `seg.replay()` (PyTorch API) |
| 依赖的 CUDA C API | 7 个 | 仅 1 个（stream 状态查询） |
| NPU 兼容 | 不兼容 | 可行 |

**新版核心思想**：每个 segment 就是一个标准的 `torch.cuda.CUDAGraph`，用 PyTorch 自带的方法管理生命周期。

### 3.2 核心类关系图

```
┌───────────────────────────────────────────────────────────┐
│                  model_runner.py                           │
│                                                           │
│   __init__()                                              │
│     |-- enable_breakable_cuda_graph?                      │
│         Yes: self.piecewise_cuda_graph_runner =           │
│              BreakableCudaGraphRunner(self)   :2754       │
│         No:  self.piecewise_cuda_graph_runner =           │
│              PiecewiseCudaGraphRunner(self)   :2756       │
│                                                           │
│   forward_extend()                                        │
│     |-- can_run = runner.can_run(batch)      :2863       │
│     |-- if can_run: runner.replay(batch)     :2870       │
└───────────────────────────────────────────────────────────┘
                          |
                          v
┌───────────────────────────────────────────────────────────┐
│           BreakableCudaGraphRunner   :71-402              │
│           breakable_cuda_graph_runner.py                   │
│                                                           │
│   +-- graphs: dict[int, BreakableCUDAGraph]  :89         │
│   +-- output_buffers: dict[int, output]      :90         │
│   +-- buffers: PrefillInputBuffers           :175         │
│   +-- replay_prepare = PiecewiseCudaGraphRunner           │
│         .replay_prepare                      :83  (绑定)   │
│                                                           │
│   方法:                                                    │
│   +-- __init__()                             :85-141      │
│   +-- _init_buffers()                        :144-186     │
│   +-- _warmup()                              :276-281     │
│   +-- _capture_all()                         :283-310     │
│   +-- _capture_one()                         :333-350     │
│   +-- can_run()                              :311-331     │
│   +-- replay()                               :352-402     │
└───────────────────────────────────────────────────────────┘
                          |
                   使用但不继承
                          |
                          v
┌───────────────────────────────────────────────────────────┐
│           BreakableCUDAGraph   :240-257                   │
│           breakable_cuda_graph.py                          │
│                                                           │
│   +-- _segments: list[torch.cuda.CUDAGraph]   :245        │
│   +-- _break_fns: list[Callable]              :246        │
│   +-- replay()                                :248-257    │
│                                                           │
│   由 BreakableCUDAGraphCapture 管理 segment 生命周期       │
└───────────────────────────────────────────────────────────┘
                          |
                          v
┌───────────────────────────────────────────────────────────┐
│      BreakableCUDAGraphCapture  :260-334                  │
│      breakable_cuda_graph.py                               │
│                                                           │
│   __enter__()                                 :290-301    │
│     +-- 安装 wait_stream hook                              │
│     +-- 设置 context vars                                  │
│     +-- _begin_new_segment()                   :316-321   │
│           graph = torch.cuda.CUDAGraph()                   │
│           graph.capture_begin(pool=shared_pool)            │
│                                                           │
│   __exit__()                                  :303-314    │
│     +-- _end_current_segment()                 :323-333   │
│           auto-join forked streams                         │
│           segment.capture_end()                            │
│     +-- 清理 context vars + hook                          │
└───────────────────────────────────────────────────────────┘
```

### 3.3 context.py：全局开关

**文件**: `breakable_cuda_graph/context.py`

```python
_in_breakable_cuda_graph = False              # 全局状态标志

def is_in_breakable_cuda_graph() -> bool:     # 供模型层查询
    return _in_breakable_cuda_graph

@contextmanager
def enable_breakable_cuda_graph():            # 上下文管理器
    global _in_breakable_cuda_graph
    _in_breakable_cuda_graph = True
    try:
        yield
    finally:
        _in_breakable_cuda_graph = False
```

**用途**：Attention 层通过 `is_in_breakable_cuda_graph()` 判断当前是否在 BCG 录制/重放中，决定是否走 `@eager_on_graph` 包装的函数。

---

## 第四章：Capture 阶段——录制多段 Graph

### 4.1 从入口到录制完成

```
model_runner.__init__()
  |
  v
BreakableCudaGraphRunner.__init__()              breakable_cuda_graph_runner.py:85
  |-- _init_buffers()                             :111  分配静态 buffer
  |-- _warmup()                                   :137  1 次 eager forward
  |-- _capture_all()                              :140  录制所有 token size
        |
        v
_capture_all()                                    :283-310
  |-- for num_tokens in reversed(sizes):          :296  从大到小（共享内存）
  |     |-- _capture_one(num_tokens, pool, stream) :307
  |           |
  |           v
  |     _capture_one()                            :333-350
  |       |-- _build_capture_forward_batch()      :335  构造 bs=1 的 ForwardBatch
  |       |-- run_once() x 2                      :341-344  预热 2 次
  |       |-- graph = BreakableCUDAGraph()        :346  创建空容器
  |       |-- with BreakableCUDAGraphCapture(...): :347  进入录制上下文
  |       |     output = run_once()               :348  执行 forward（触发 segment 切分）
  |       |-- return graph, output                :350
  |
  v
self.graphs[num_tokens] = graph                  :308  保存到字典
self.output_buffers[num_tokens] = output         :309
```

**关键：为什么从大到小录制？** 所有 size 的 graph 共享同一个 `graph_pool_handle()`（mempool）。大的 graph 先录制会分配更多内存；小的后录制时可以复用大 graph 释放的内存空间，最大化内存效率。

### 4.2 Segment 的 begin/end 循环

录制 `model.forward()` 时，Attention 层的 `@eager_on_graph` 装饰函数会触发 segment 切分：

```
BreakableCUDAGraphCapture.__enter__()                     :290-301
  +-- _begin_new_segment()                                :300
        graph_0 = torch.cuda.CUDAGraph()
        graph_0.capture_begin(pool=shared_pool)           # 开始录制 segment 0
        segments.append(graph_0)

model.forward() 执行中...
  |-- Linear, LayerNorm, etc. 的 kernel 被录入 segment_0
  |
  |-- 遇到 Attention 层（被 @eager_on_graph 装饰）:
  |     capture._end_current_segment()                    :211  结束 segment 0
  |       +-- auto-join forked streams                    :327-332
  |       +-- segments[-1].capture_end()                  :333  segment_0 录制完成
  |
  |     output = attention_forward(...)                   :215  eager 执行 attention
  |     capture.cuda_graph._break_fns.append(replay_fn)  :229  记录 break 函数
  |
  |     capture._begin_new_segment()                      :232  开始 segment 1
  |       graph_1 = torch.cuda.CUDAGraph()
  |       graph_1.capture_begin(pool=shared_pool)
  |       segments.append(graph_1)
  |
  |-- MLP, Linear 等的 kernel 被录入 segment_1
  |
  |-- 遇到下一个 Attention 层:
  |     ... (重复上述 begin/end 循环) ...

BreakableCUDAGraphCapture.__exit__()                      :303-314
  +-- _end_current_segment()                              :305  结束最后一个 segment
  +-- 清理 context vars + uninstall hook
```

**图示——一个 3 层 Transformer 的 Capture 过程**：

```
时间轴 ──────────────────────────────────────────────────────>

[seg_0 capture_begin]
  |-- Embedding kernel
  |-- Layer 0: Linear QKV
  |-- Layer 0: Linear projection
[seg_0 capture_end]          ← 第 1 个 Attention 触发 break
  |
[eager: Layer 0 Attention]   ← 运行真正的 attention 计算
  |
[seg_1 capture_begin]
  |-- Layer 1: Linear QKV
  |-- Layer 1: Linear projection
[seg_1 capture_end]          ← 第 2 个 Attention 触发 break
  |
[eager: Layer 1 Attention]
  |
[seg_2 capture_begin]
  |-- Layer 2: Linear QKV
  |-- Layer 2: Linear projection
  |-- Final Linear
[seg_2 capture_end]          ← 录制结束

结果: 3 个 CUDAGraph segments + 2 个 break 函数
```

### 4.3 共享 Mempool 与 weak_ref_tensor

所有 segment 共享同一个 CUDA memory pool。这带来两个好处：

1. **内存复用**：segment 0 释放的中间张量可以被 segment 1 复用
2. **地址稳定**：`weak_ref_tensor` 对 segment 间传递的中间结果创建弱引用视图

**文件**: `breakable_cuda_graph.py:150-163`

```python
def _weak_ref_if_tensor(x):
    if torch.is_tensor(x):
        from sglang.srt.compilation.weak_ref_tensor import weak_ref_tensors
        return weak_ref_tensors(x)       # 弱引用：不增加引用计数
    return x
```

**为什么用弱引用？** segment 间传递的 tensor 存储在共享 mempool 中。只要任何 segment 的 CUDAGraph 存活，mempool 就不会被回收。所以 Python 端不需要强引用来保持 tensor 存活——弱引用足够了，还能让 GC 及时回收不再需要的中间结果。

---

## 第五章：Replay 阶段——交替执行

### 5.1 完整 Replay 流程

```
model_runner.forward_extend()                            model_runner.py:2870
  |-- runner.can_run(batch)                              :2863
  |     check: bs==1, no input_embeds, tokens<=max       :317-331
  |
  |-- runner.replay(forward_batch)                       :2870
        |
        v
BreakableCudaGraphRunner.replay()                        breakable_cuda_graph_runner.py:352
  |-- 找到最接近的 captured token size                    :358-359
  |     bisect_left(capture_num_tokens, num_tokens)
  |
  |-- enable_breakable_cuda_graph()                      :361  设置全局标志
  |-- replay_prepare(batch)                              :362  拷贝数据到静态 buffer
  |-- 更新 static_seq_lens 等 buffer                     :367-373
  |
  |-- init_forward_metadata(batch)                       :376
  |-- set_forward_context(...)                           :377-383
  |-- graph.replay()                                     :384  交替执行!
        |
        v
BreakableCUDAGraph.replay()                              breakable_cuda_graph.py:248-257
  |
  |-- seg_0.replay()               ← 重放第 1 段（PyTorch 高层 API）
  |-- break_fns[0]()               ← eager 执行 attention
  |     +-- attention(*captured_args)    用更新后的 buffer 数据
  |     +-- _copy_output(old, new)       结果写回静态 buffer
  |
  |-- seg_1.replay()               ← 重放第 2 段
  |-- break_fns[1]()               ← eager 执行 attention
  |     ...
  |
  |-- seg_N.replay()               ← 重放最后一段
  |
  v
返回 output_buffers[token_size]                          :386
```

**ASCII 流程图——Replay 的交替执行**：

```
                    BreakableCUDAGraph.replay()
                            |
                            v
            ┌── seg_0.replay() ──┐
            │   GPU 执行录制的     │
            │   kernel 序列 0     │
            └────────────────────┘
                            |
                            v
            ┌── break_fn_0() ─────┐
            │   attention(*args)   │  ← CPU 执行 Python 代码
            │   _copy_output()     │  ← 结果写回静态 buffer
            └─────────────────────┘
                            |
                            v
            ┌── seg_1.replay() ──┐
            │   GPU 执行录制的     │
            │   kernel 序列 1     │
            └────────────────────┘
                            |
                            v
            ┌── break_fn_1() ─────┐
            │   attention(*args)   │
            │   _copy_output()     │
            └─────────────────────┘
                            |
                            v
                         ...
                            |
                            v
            ┌── seg_N.replay() ──┐
            │   最后一段录制的     │
            │   kernel 序列       │
            └────────────────────┘
                            |
                            v
                        返回结果
```

### 5.2 replay_prepare：数据搬运

`replay_prepare` 是从 `PiecewiseCudaGraphRunner` 绑定的方法（非继承）：

**文件**: `breakable_cuda_graph_runner.py:83`

```python
replay_prepare = PiecewiseCudaGraphRunner.replay_prepare   # 绑定，不继承
```

**为什么绑定而非继承？** BCG Runner 的 `__init__`、capture、replay 逻辑与 PCG 差异太大。继承会带来不必要的耦合。但 buffer 填充逻辑（`replay_prepare`）两者完全相同——把动态 batch 数据拷贝到预分配的静态 buffer。所以直接绑定这个方法。

### 5.3 _copy_output：段间结果传递

break 函数执行后，结果需要写回静态 buffer，供下一个 segment 读取。

**文件**: `breakable_cuda_graph.py:166-195`

```python
def _copy_output(dst, src):
    if torch.is_tensor(dst) and torch.is_tensor(src):
        dst.copy_(src)                    # 纯 tensor: 原地拷贝
        return dst

    if hasattr(dst, "__dict__"):          # 对象/dataclass: 遍历属性
        for key, src_val in src.__dict__.items():
            dst_val = getattr(dst, key, None)
            if torch.is_tensor(dst_val):
                dst_val.copy_(src_val)     # tensor 属性: 原地拷贝
            else:
                setattr(dst, key, src_val) # 非 tensor 属性: 直接替换
        return dst

    if isinstance(dst, dict):             # 字典: 遍历 key
        ...
    return src
```

**为什么必须原地拷贝？** 下一个 segment 的 `torch.cuda.CUDAGraph` 持有 `dst` 的地址引用。如果替换 `dst` 而不是原地修改，graph replay 时会读取到旧地址，导致错误结果。

---

## 第六章：在 Attention 层插入 Graph Break

Attention 操作涉及动态 shape 和复杂的内存访问模式，是 graph break 的典型位置。

**文件**: `layers/radix_attention.py:129-131, 219`

```python
# 定义：包装 attention 函数
bcg_unified_attention_with_output = eager_on_graph(True)(
    unified_attention_with_output
)                                                    # :219

# 使用：在 forward 中根据上下文选择
def forward(self, hidden_states, ...):
    ...
    if is_in_breakable_cuda_graph():                 # :129
        bcg_unified_attention_with_output(           # 走 graph break 路径
            q, k, v, output, ...)
    else:
        unified_attention_with_output(               # 正常路径
            q, k, v, output, ...)
```

**流程**：

```
radix_attention.forward()
  |
  +-- is_in_breakable_cuda_graph() == True?
        |
        Yes ──> bcg_unified_attention_with_output()   # @eager_on_graph 包装
        |         触发: end_segment -> eager attn -> begin_segment
        |
        No  ──> unified_attention_with_output()       # 正常执行
```

`break_graph()` 函数提供了另一种方式——只想插入切割点，不执行任何 eager 计算：

**文件**: `breakable_cuda_graph.py:336-340`

```python
@eager_on_graph(True)
def break_graph() -> None:
    """Insert a graph break. Body intentionally does nothing."""
    pass
```

---

## 第七章：如何启用 BCG

### 命令行参数

**文件**: `server_args.py:5868-5889`

```bash
# 启用 BCG（替换 PCG，用于 extend 路径）
python -m sglang.srt.server \
    --model meta-llama/Llama-3.1-8B \
    --enable-breakable-cuda-graph

# 调试模式（整个 forward 变为 eager，但仍走 graph 代码路径）
python -m sglang.srt.server \
    --model meta-llama/Llama-3.1-8B \
    --debug-cuda-graph
```

### 限制

| 限制 | 原因 | 代码位置 |
|------|------|----------|
| `batch_size == 1` | capture 时 bs=1，logits 路径硬编码 | `can_run():317` |
| 不支持 `input_embeds` | 多模态嵌入暂未适配 | `can_run():319` |
| 不支持 `return_logprob` 且 `start_len < seq_len` | 部分 logprob 路径与 graph 不兼容 | `can_run():324-330` |
| token 数 <= `max_num_tokens` | 超出捕获范围 | `can_run():331` |

### 选择逻辑总结

```
model_runner.__init__()
  |
  +-- enable_breakable_cuda_graph?
  |     Yes: piecewise_cuda_graph_runner = BreakableCudaGraphRunner
  |     No:  piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner
  |
  +-- cuda_graph_runner = CudaGraphRunner (decode 路径，始终存在)

model_runner.forward_extend()
  |
  +-- piecewise_runner.can_run(batch)?
        Yes: runner.replay(batch)    # BCG 或 PCG
        No:  eager forward()         # 兜底：普通 forward
```

**架构定位**：

```
+-----------------------------------------------------------------+
|                        ModelRunner                               |
|                                                                  |
|   Decode 路径                   Extend 路径                      |
|   +-----------------------+    +---------------------------+    |
|   |  CudaGraphRunner      |    | piecewise_cuda_graph_runner|   |
|   |  (标准 CG)            |    |                           |    |
|   |  每次 1 token          |    |  BCG 或 PCG (二选一)       |    |
|   |  固定 batch size       |    |  动态 token 数             |    |
|   +-----------------------+    +---------------------------+    |
+-----------------------------------------------------------------+
```

---

> **本文档定位**：面向零基础读者的 BCG 入门教程，覆盖 PR #22218 重构后的实现。
> - **doc/03** 覆盖旧版 BCG（继承 `torch.cuda.CUDAGraph`）
> - **doc/04** 覆盖 PR #22218 的 diff 分析
> - **本文档** 从 CUDA Graph 基础概念讲起，到新版 BCG 的完整实现
