# Piecewise CUDA Graph vs Breakable CUDA Graph 详细流程与调用链分析

> 本文档包含 PCG 和 BCG 两种 CUDA Graph 优化技术的完整流程图、调用链，以及所有关键代码引用（含文件路径和行号）。

---

## 目录

- [一、Piecewise CUDA Graph (PCG) 详细流程与调用链](#一piecewise-cuda-graph-pcg-详细流程与调用链)
  - [1. 总体流程图](#1-总体流程图)
  - [2. 详细调用链（带代码行号）](#2-详细调用链带代码行号)
- [二、Breakable CUDA Graph (BCG) 详细流程与调用链](#二breakable-cuda-graph-bcg-详细流程与调用链)
  - [1. 总体流程图](#1-总体流程图-1)
  - [2. 详细调用链（带代码行号）](#2-详细调用链带代码行号-1)
- [三、两者的 Split 机制对比图](#三两者的-split-机制对比图)
- [四、关键数据结构对比](#四关键数据结构对比)
- [五、NPU 适配分析](#五-npu-适配分析)

---

## 一、Piecewise CUDA Graph (PCG) 详细流程与调用链

### 1. 总体流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│                     启动阶段 (Server Init)                           │
│  ModelRunner.__init__()                                             │
│    └─► init_piecewise_cuda_graphs()                                 │
│          ├─ 检查禁用条件 (draft_worker / 非 language model / ...)     │
│          ├─ 收集 attention_layers / moe_layers / moe_fusions         │
│          └─► PiecewiseCudaGraphRunner.__init__()                     │
│                ├─ [Phase 1] 分配静态 buffers                        │
│                ├─ [Phase 2] torch.compile 编译                      │
│                │     ├─ install_torch_compiled()                     │
│                │     ├─ warmup_compile() × N shapes                  │
│                │     └─ SGLangBackend.__call__()                     │
│                │           ├─ split_graph() → 按 split_ops 切分子图  │
│                │           ├─ PiecewiseCompileInterpreter.run()      │
│                │           │     └─ 每个 submod → make_backend()     │
│                │           │           └─ CUDAPiecewiseBackend()     │
│                │           └─ 返回 split_gm (缝合模块)               │
│                └─ [Phase 3] capture() CUDA Graph                    │
│                      └─ capture_one_batch_size() × N shapes          │
│                            └─ run_once() × 2 (warmup + capture)     │
│                                  └─ model.forward()                  │
│                                        → trampoline()                │
│                                          → compiled_callable()       │
│                                            → split_gm.forward()     │
│                                              → 每个 submod →         │
│                                                CUDAPiecewiseBackend │
│                                                  .__call__()        │
├──────────────────────────────────────────────────────────────────────┤
│                     推理阶段 (Serving)                               │
│  ModelRunner.forward_extend()                                       │
│    ├─ can_run_graph?                                                 │
│    │   └─ PiecewiseCudaGraphRunner.can_run()                        │
│    ├─ YES → PiecewiseCudaGraphRunner.replay()                       │
│    │         ├─ replay_prepare(): 拷贝输入到静态buffer + padding     │
│    │         └─ model.forward()                                      │
│    │               → trampoline()                                    │
│    │                 → compiled_callable()                           │
│    │                   → split_gm.forward()                          │
│    │                     → 每个 submod                               │
│    │                       CUDAPiecewiseBackend.__call__()           │
│    │                         └─ entry.cudagraph.replay()             │
│    └─ NO  → model.forward() (eager path)                            │
└──────────────────────────────────────────────────────────────────────┘
```

### 2. 详细调用链（带代码行号）

#### Phase 0: 入口 — 初始化判断

```
model_runner.py:738     ModelRunner.__init__()
  └─ model_runner.py:2606   init_piecewise_cuda_graphs()
       ├─ :2610   检查 disable_piecewise_cuda_graph
       ├─ :2617   检查 is_draft_worker
       ├─ :2621   检查 model 是否有 language model
       ├─ :2628   检查 piecewise_cuda_graph_tokens 是否设置
       ├─ :2645-2699  收集 attention_layers / moe_layers / moe_fusions
       └─ :2715   self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)
```

#### Phase 1: 静态 Buffer 分配

```
piecewise_cuda_graph_runner.py:155   PiecewiseCudaGraphRunner.__init__()
  ├─ :158-165   保存 model_runner, device, graphs, tp/dp/pp_size
  ├─ :169       set_torch_compile_config()              — :136 配置 dynamo cache limit
  ├─ :178-182   CompilationConfig(tokens, compiler, debug)
  ├─ :183-186   add_split_op("sglang.moe_forward_...")  — DeepEP/Mooncake
  ├─ :191       self.capture_num_tokens = compile_config.get_capture_sizes()
  ├─ :195-200   self.capture_forward_mode = EXTEND; capture_hidden_mode = NULL/FULL
  ├─ :214-270   分配静态 buffer:
  │     input_ids (:215)    out_cache_loc (:217)     positions (:239)
  │     input_embeds (:249)  mrope_positions (:253)   (multimodal)
  │     mamba_track_* (:224-238)                      (mamba)
  │     → PrefillInputBuffers(:260)
  │     → buffers.share_buffers()(:271)
  ├─ :277-280   全局 graph memory pool
  └─ :282       with enable_piecewise_cuda_graph():   — piecewise_context_manager.py:42
```

#### Phase 2: torch.compile 编译

```
piecewise_cuda_graph_runner.py:283-319  编译阶段
  ├─ :283-285   获取 language_model.model
  ├─ :286       with patch_model(model, compiler):
  │               → _to_torch() (:102) — MultiPlatformOp 切换到 torch compile 模式
  ├─ :291       warmup_compile(num_tokens=capture_num_tokens[0])  — 首次 warmup
  ├─ :293-299   install_torch_compiled(patched_model, fullgraph=True, ...)
  │     → compile.py:111
  │       ├─ :120   unbound_fwd = module.__class__.forward
  │       ├─ :124   dyn_map = _infer_dynamic_arg_dims_from_annotations()
  │       ├─ :128-131  backend_factory = lambda gm, ex: SGLangBackend(config, pool)(gm, ex)
  │       └─ :188-200  trampoline(self, *args, **kwargs) — 替换 module.forward
  │             ├─ :189  use_compiled = is_in_piecewise_cuda_graph()
  │             ├─ :191  _ensure_compiled() — 首次调用时触发 torch.compile
  │             │   └─ :178  compiled_callable = torch.compile(bound, fullgraph=True, backend=...)
  │             │         → 触发 Dynamo trace → SGLangBackend.__call__()
  │             └─ :195  compiled_callable(*args, **kwargs) — 后续直接调用
  │
  ├─ :301       with enable_piecewise_cuda_graph_compile(): — piecewise_context_manager.py:34
  ├─ :302-312   for num_tokens in reversed(capture_num_tokens):
  │               warmup_compile(num_tokens)
  │               → :324 warmup_compile()
  │                 ├─ :326-396   构造 ForwardBatch
  │                 ├─ :400       attn_backend.init_forward_metadata()
  │                 ├─ :404       set_forward_context()
  │                 └─ :411       model.forward() → trampoline → compiled_callable
  │                               → Dynamo trace 整个模型
  └─ :314-318   synchronize + barrier
```

#### Phase 2.1: SGLangBackend — 图分裂与分片编译

```
compile.py:402   SGLangBackend.__call__(graph, example_inputs)
  ├─ :403-419   初始化缓存目录
  ├─ :424       self.graph = graph
  ├─ :425       configure_post_pass()
  ├─ :427-430   split_graph(graph, split_ops)
  │     → backend.py:220
  │       ├─ :223-236   遍历所有 node，按 split_ops 分配 subgraph_id
  │       │     split_ops 来源:
  │       │       ├─ parallel_state.py:151  @register_split_op()  → custom_all_reduce
  │       │       ├─ radix_attention.py:139 @register_split_op()  → radix_attention
  │       │       ├─ radix_linear_attention.py:105 @register_split_op() → linear_attention
  │       │       ├─ nemotron_h.py:887 @register_split_op()       → nemotron mixer
  │       │       └─ piecewise_cuda_graph_runner.py:185 add_split_op(moe_forward_...)
  │       ├─ :242-244   torch.fx.passes.split_module.split_module()
  │       └─ :246-263   生成 SplitItem 列表
  ├─ :440-444   submod_names_to_compile = [非 splitting 图]
  ├─ :446-453   PiecewiseCompileInterpreter(split_gm, ...).run(*example_inputs)
  │     → backend.py:272
  │       └─ :294-300  run() — 用 fake tensor 模拟执行
  │           └─ :302-343  call_module(target, args, kwargs)
  │             ├─ :311  if target in compile_submod_names:
  │             ├─ :318-327  compiler_manager.compile(submod, args, ...)
  │             │     → backend.py:135
  │             │       ├─ :149  compilation_counter++
  │             │       └─ :161  compiler.compile(graph, inputs, config, shape, key)
  │             │             → EagerAdapter / InductorAdaptor
  │             └─ :329-339  make_backend(submod, ...) → CUDAPiecewiseBackend/NPUPiecewiseBackend
  │                   → backend.py:40
  │                     ├─ :52-53  out-of-tree platform → platform backend
  │                     ├─ :54-55  is_npu() → NPUPiecewiseBackend
  │                     └─ :57     else → CUDAPiecewiseBackend
  └─ :473  return self.split_gm  — 缝合模块
```

#### Phase 3: CUDA Graph Capture

```
piecewise_cuda_graph_runner.py:450   capture()
  ├─ :454-456  with freeze_gc(), graph_capture():
  ├─ :457-458  stream = graph_capture_context.stream; set_pcg_capture_stream(stream)
  ├─ :465-469  for num_tokens in reversed(capture_num_tokens):  — 大→小，共享内存
  └─ :481      capture_one_batch_size(num_tokens)
        → :483
        ├─ :484-569  构造 ForwardBatch (与 warmup 类似)
        ├─ :575      attn_backend.init_forward_metadata()
        ├─ :578-604  def run_once():  — 实际执行 forward
        │     ├─ :591  set_forward_context()
        │     └─ :598  model.forward() → trampoline → compiled_callable → split_gm
        │                → 每个 submod → CUDAPiecewiseBackend.__call__()
        │                  → cuda_piecewise_backend.py:107
        │                    ├─ :108-111  first_run → 返回通用形状编译结果
        │                    ├─ :113-114  无 sym_shape → 直接运行
        │                    ├─ :116-119  shape 不在 capture 列表 → 直接运行
        │                    ├─ :126-141  need_to_compile → compiler_manager.compile()
        │                    ├─ :143-144  is_in_pcg_torch_compile() → 直接运行
        │                    ├─ :146-194  cudagraph is None:
        │                    │   ├─ :147-149  warmup (num_finished_warmup < 1)
        │                    │   ├─ :156      cudagraph = torch.cuda.CUDAGraph()
        │                    │   │            [NPU: torch.npu.NPUGraph()]
        │                    │   ├─ :169      stream = get_pcg_capture_stream()
        │                    │   ├─ :173      with torch.cuda.graph(cudagraph, pool, stream):
        │                    │   │            [NPU: with torch.npu.graph(npugraph, pool):]
        │                    │   ├─ :175      output = entry.runnable(*args)
        │                    │   ├─ :176-182  weak_ref_tensors(output)  — 最后一个图
        │                    │   ├─ :186-187  entry.output / entry.cudagraph 保存
        │                    │   └─ :189      compilation_counter++
        │                    └─ :196-206  已 capture → entry.cudagraph.replay()
        └─ :608-612  for _ in range(2):  — warmup 1次 + capture 1次
              synchronize + barrier + run_once()
```

#### Phase 4: 推理 Replay

```
model_runner.py:2818-2827  forward_extend/decode 判断
  ├─ :2818  can_run_graph = piecewise_cuda_graph_runner.can_run(batch)
  │   → piecewise_cuda_graph_runner.py:420
  │     ├─ :423  input_embeds → False
  │     ├─ :428  target_verify → False
  │     ├─ :433  capture_hidden_mode 不匹配 → False
  │     ├─ :436  replace_embeds → False
  │     ├─ :439-445  logprob 检查
  │     └─ :446-448  num_tokens <= max → True
  │
  └─ :2823-2826  can_run_graph → piecewise_cuda_graph_runner.replay(batch)
        → piecewise_cuda_graph_runner.py:777
        ├─ :782  with enable_piecewise_cuda_graph():
        ├─ :783  static_forward_batch = replay_prepare(forward_batch)
        │   → :615
        │   ├─ :622  index = bisect_left(capture_num_tokens, num_tokens)
        │   ├─ :623  static_num_tokens = capture_num_tokens[index]  — 找最近的上界
        │   ├─ :625-634  padding: zero_ 多余部分
        │   ├─ :638-662  copy_: 拷贝实际数据到静态 buffer
        │   └─ :717-770  构造 static_forward_batch
        ├─ :785  set_forward_context()
        ├─ :793  attn_backend.init_forward_metadata()
        └─ :794  model.forward() → trampoline → compiled_callable → split_gm
                  → 每个 submod → CUDAPiecewiseBackend.__call__()
                    → :205  entry.cudagraph.replay()  ← CUDA Graph 重放
                    → :206  return entry.output
```

---

## 二、Breakable CUDA Graph (BCG) 详细流程与调用链

### 1. 总体流程图

```
┌──────────────────────────────────────────────────────────────────────┐
│                     启动阶段 (Server Init)                           │
│  ModelRunner.__init__()                                             │
│    └─► init_cuda_graphs()                                           │
│          └─► CudaGraphRunner.__init__()                             │
│                ├─ 分配静态 DecodeInputBuffers                       │
│                └─► capture()                                        │
│                      └─► capture_one_batch_size(bs)                  │
│                            ├─ _create_device_graph()                │
│                            │   └─ BreakableCUDAGraph()              │
│                            ├─ run_once() × 2 (warmup)               │
│                            └─ _capture_graph(graph, pool, stream,   │
│                                               run_once)             │
│                                  ├─ BreakableCUDAGraphCapture.__init__()
│                                  ├─ BreakableCUDAGraphCapture.__enter__()
│                                  │   ├─ _install_wait_stream_hook()  │
│                                  │   ├─ 设置 ContextVar:             │
│                                  │   │   _captured_graphs = []       │
│                                  │   │   _current_stream = stream    │
│                                  │   │   _forked_streams = set()     │
│                                  │   └─ BreakableCUDAGraph.capture_begin()
│                                  │       ├─ torch.CUDAGraph.capture_begin()
│                                  │       ├─ _end_capture_segment()   │
│                                  │       └─ _begin_capture_segment() │
│                                  ├─ run_once() ← 用户函数执行        │
│                                  │   ├─ [正常 CUDA ops] → 录入 graph │
│                                  │   ├─ 遇到 @eager_on_graph 函数:   │
│                                  │   │   ├─ _end_capture_segment()   │
│                                  │   │   │   → rt.cudaStreamEndCapture()
│                                  │   │   ├─ func(*args) ← eager 执行 │
│                                  │   │   ├─ 创建 replay_fn           │
│                                  │   │   ├─ append GraphBreakInfo    │
│                                  │   │   └─ _begin_capture_segment() │
│                                  │   │       → rt.cudaStreamBeginCapture()
│                                  │   └─ [继续正常 CUDA ops] → 新 segment
│                                  └─ BreakableCUDAGraphCapture.__exit__()
│                                      ├─ torch.cuda.graph.__exit__()
│                                      │   └─ BreakableCUDAGraph.capture_end()
│                                      │       ├─ _end_capture_segment() → last_graph
│                                      │       ├─ _instantiate_graph(last_graph)
│                                      │       ├─ for each break:
│                                      │       │   _instantiate_graph(break.handle)
│                                      │       ├─ _begin_capture_segment() (dummy)
│                                      │       └─ super().capture_end()
│                                      ├─ reset ContextVars
│                                      └─ _uninstall_wait_stream_hook()
├──────────────────────────────────────────────────────────────────────┤
│                     推理阶段 (Serving)                               │
│  CudaGraphRunner.replay(forward_batch)                              │
│    ├─ replay_prepare(forward_batch)                                 │
│    │   └─ 拷贝数据到静态 buffer + bisect 找 capture_bs             │
│    └─ self.graphs[bs].replay()                                      │
│          → BreakableCUDAGraph.replay()                              │
│            ├─ for (replay_fn, _, handle) in self._exec:             │
│            │   ├─ _replay_graph(handle, stream_ptr)                 │
│            │   │   → rt.cudaGraphLaunch(exec, stream)               │
│            │   └─ replay_fn()  ← eager 函数重执行                   │
│            └─ _replay_graph(last_graph_exec, stream_ptr)            │
│                  → rt.cudaGraphLaunch(last_exec, stream)            │
└──────────────────────────────────────────────────────────────────────┘
```

### 2. 详细调用链（带代码行号）

#### Phase 0: 入口 — 创建图

```
model_runner.py  init_cuda_graphs()
  └─ CudaGraphRunner.__init__()
       → cuda_graph_runner.py
       ├─ 分配 DecodeInputBuffers.create()  (:150-260)
       └─ capture()  (:761)
            → :761  def capture()
            └─ :766  _capture_one_stream()
                 └─ :778  for bs in reversed(capture_bs):
                      └─ capture_one_batch_size(bs)
                           → :864
```

#### Phase 1: 创建 BreakableCUDAGraph 实例

```
cuda_graph_runner.py:864   capture_one_batch_size(bs)
  ├─ :868   graph = self._create_device_graph()
  │   → :857   def _create_device_graph()
  │     ├─ :858  if SGLANG_USE_BREAKABLE_CUDA_GRAPH:
  │     ├─ :859  if _is_hip: raise  — ROCm 不支持
  │     └─ :861  return BreakableCUDAGraph()
  │              → breakable_cuda_graph.py:264
  │                └─ :266  super().__new__(cls, True) — 多段 capture 模式
  │
  ├─ :870-1003  构造 ForwardBatch (decode 模式)
  ├─ :1019      attn_backend.init_forward_metadata_capture_cuda_graph()
  ├─ :1030      def run_once(): — 用户函数
  │     └─ :1055  forward(input_ids, positions, forward_batch, **kwargs)
  │
  ├─ :1065-1068  run_once() × 2  — warmup
  └─ :1074      out = self._capture_graph(graph, pool, stream, run_once)
```

#### Phase 2: Capture 过程

```
cuda_graph_runner.py:824   _capture_graph(graph, pool, stream, run_once_fn)
  ├─ :830  memory_saver_adapter (可选)
  ├─ :835  if SGLANG_USE_BREAKABLE_CUDA_GRAPH:
  ├─ :840    graph_ctx = BreakableCUDAGraphCapture
  ├─ :848  if debug_cuda_graph:
  │   └─ :849    captured_fn = eager_on_graph(True)(run_once_fn)
  │              → breakable_cuda_graph.py:225
  │                └─ wrapper(*args, **kwargs)
  │                    ├─ :232  if not _is_capturing() → 直接执行
  │                    ├─ :234  last_graph = _end_capture_segment(stream)
  │                    │   → :146 _end_capture_segment()
  │                    │     ├─ :149-155  join forked streams
  │                    │     └─ :157  rt.cudaStreamEndCapture() ← 结束当前 segment
  │                    ├─ :237  output = inner(*args) ← eager 执行
  │                    ├─ :247-249  创建 replay_fn()
  │                    ├─ :255  captured_graphs.append(GraphBreakInfo(...))
  │                    └─ :256  _begin_capture_segment(stream)
  │                        → :162
  │                          └─ :163  rt.cudaStreamBeginCapture()
  │
  │  [注意: 非 debug 模式下 captured_fn = run_once_fn，无主动 break]
  │  [break 来自模型内部 eager_on_graph 装饰的函数 / break_graph()]
  │
  └─ :853  with graph_ctx(cuda_graph=graph, pool=pool, stream=stream):
           │   → BreakableCUDAGraphCapture.__enter__()
           │     → breakable_cuda_graph.py:333
           │       ├─ :334  _install_wait_stream_hook()
           │         → :127  hook wait_stream 追踪 fork/join
           │       ├─ :335  _captured_graphs_var.set([])
           │       ├─ :336  _current_stream_var.set(stream)
           │       ├─ :337  _forked_streams_var.set(set())
           │       └─ :338  super().__enter__()
           │             → torch.cuda.graph.__enter__()
           │               → BreakableCUDAGraph.capture_begin()
           │                 → :269
           │                   ├─ :271  super().capture_begin(pool, capture_error_mode)
           │                   │   → torch.cuda.CUDAGraph.capture_begin()
           │                   │     → rt.cudaStreamBeginCapture() (PyTorch 内部)
           │                   ├─ :274  _end_capture_segment(stream)  ← 结束 PyTorch 的 capture
           │                   │   → rt.cudaStreamEndCapture()
           │                   └─ :275  _begin_capture_segment(stream) ← 开始我们的 capture
           │                       → rt.cudaStreamBeginCapture()
           │
           ├─ :854  out = captured_fn()  ← 执行用户函数
           │   └─ run_once()
           │       └─ model.forward(...)
           │           [CUDA ops 被 stream capture 记录]
           │           [遇到 @eager_on_graph / break_graph → segment break]
           │           [每个 break: end_capture → eager → begin_capture]
           │
           └─ BreakableCUDAGraphCapture.__exit__()
               → breakable_cuda_graph.py:340
                 ├─ :341  super().__exit__()
                 │   → torch.cuda.graph.__exit__()
                 │     → torch 会调用 capture_end()
                 │       → BreakableCUDAGraph.capture_end()
                 │         → :277
                 │           ├─ :279  last_graph = _end_capture_segment(stream)
                 │             → rt.cudaStreamEndCapture()  ← 最后一个 segment
                 │           ├─ :280  last_graph_exec = _instantiate_graph(last_graph)
                 │             → :171
                 │               ├─ :172  rt.cudaGraphInstantiateWithFlags()
                 │               └─ :179  rt.cudaGraphDestroy() (销毁原始 graph)
                 │           ├─ :281-286  for each break:
                 │           │   graph_exec = _instantiate_graph(handle)
                 │           │   self._exec.append(GraphBreakInfo(replay_fn, output, exec))
                 │           ├─ :289  _begin_capture_segment(stream)  ← dummy capture
                 │           └─ :290  super().capture_end()  ← 让 PyTorch 正常结束
                 │
                 ├─ :342  reset _current_stream_var
                 ├─ :343  reset _captured_graphs_var
                 ├─ :344  reset _forked_streams_var
                 └─ :345  _uninstall_wait_stream_hook()
                     → :136  恢复原始 wait_stream
```

#### Phase 3: 推理 Replay

```
cuda_graph_runner.py:1193   replay(forward_batch)
  ├─ :1202  replay_prepare(forward_batch)
  │   → :1112
  │     ├─ :1118  recapture_if_needed()  — 检查 capture_hidden_mode
  │     ├─ :1133-1136  bisect_left 找最近的 capture_bs
  │     ├─ :1138  buffers.populate_from_forward_batch()
  │     └─ :1174  attn_backend.init_forward_metadata_replay_cuda_graph()
  │
  └─ :1221  self.graphs[graph_key].replay()
            → BreakableCUDAGraph.replay()
              → breakable_cuda_graph.py:292
                ├─ :293  stream = torch.cuda.current_stream()
                ├─ :294  token = _current_stream_var.set(stream)
                ├─ :296  if not self._exec:  — 无 break
                │   └─ :297  _replay_graph(last_graph_exec, stream.cuda_stream)
                │       → :187  rt.cudaGraphLaunch(exec, stream_ptr)
                ├─ :299-302  for (func, _, handle) in self._exec:
                │   ├─ :300  _replay_graph(handle, stream.cuda_stream)
                │   │   → rt.cudaGraphLaunch()
                │   └─ :301  func()  ← 重执行 eager 函数
                │       → :248  captured_inner(*captured_args)
                │       → :249  _copy_output(captured_output, new_out)  — 原地拷贝
                └─ :304  _current_stream_var.reset(token)
```

#### BCG 关键底层 API 调用汇总

```
breakable_cuda_graph.py 中的 cuda.bindings.runtime 调用:

:163-168  rt.cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal)
:157      rt.cudaStreamEndCapture(stream)         → 返回 cudaGraph_t
:172-176  rt.cudaGraphInstantiateWithFlags(graph, cudaGraphInstantiateFlagAutoFreeOnLaunch)
:179      rt.cudaGraphDestroy(graph_ptr)           → 销毁 cudaGraph_t
:184      rt.cudaGraphExecDestroy(graph_exec_ptr)  → 销毁 cudaGraphExec_t
:188      rt.cudaGraphLaunch(exec, stream)         → 启动图执行
:75       rt.cudaStreamGetCaptureInfo(stream)      → 查询 capture 状态
```

---

## 三、两者的 Split 机制对比图

```
PCG: 编译期静态分裂 (FX Graph Level)
═══════════════════════════════════════

  torch.compile → FX Trace → split_graph()
                                    │
                                    ▼
  ┌─────────┐  split_op  ┌─────────┐  split_op  ┌─────────┐
  │ submod_0│───────────►│ submod_1│───────────►│ submod_2│ ...
  │(layers  │  AllReduce │(layers  │  Attention │(layers  │
  │ before  │   / MoE    │ between │   op       │ after   │
  │ split)  │            │ splits) │            │ split)  │
  └────┬────┘            └────┬────┘            └────┬────┘
       │                      │                      │
  torch.cuda.graph       torch.cuda.graph       torch.cuda.graph
  (独立 CUDA Graph)       (独立 CUDA Graph)       (独立 CUDA Graph)
  每个对应 N 个 size       每个对应 N 个 size       每个对应 N 个 size
  replay: submod.replay   replay: submod.replay   replay: submod.replay

  → 每个 submod 完整 captured 为一个 CUDA Graph
  → split 点固定, 运行时不变


BCG: 运行期动态分裂 (CUDA Stream Capture Level)
═════════════════════════════════════════════════

  torch.cuda.graph context
  ┌─────────────────────────────────────────────────────────┐
  │  cudaStreamBeginCapture()                               │
  │  ┌──────────────────┐                                   │
  │  │ CUDA ops (graphed)│                                  │
  │  └──────────────────┘                                   │
  │  cudaStreamEndCapture() → segment_0 (cudaGraph_t)       │
  │  ┌──────────────────┐                                   │
  │  │ eager func (Python)│  ← @eager_on_graph / break_graph│
  │  └──────────────────┘                                   │
  │  cudaStreamBeginCapture()                               │
  │  ┌──────────────────┐                                   │
  │  │ CUDA ops (graphed)│                                  │
  │  └──────────────────┘                                   │
  │  cudaStreamEndCapture() → segment_1 (cudaGraph_t)       │
  │  ...                                                    │
  │  cudaStreamBeginCapture()                               │
  │  ┌──────────────────┐                                   │
  │  │ CUDA ops (graphed)│                                  │
  │  └──────────────────┘                                   │
  │  cudaStreamEndCapture() → last_segment (cudaGraph_t)    │
  └─────────────────────────────────────────────────────────┘

  → 整个 forward 只有"一个" BreakableCUDAGraph 对象
  → 内部由多个 cudaGraphExec 组成, 运行时动态确定 break 点
  → replay: 交替 launch graph_exec + 执行 eager func
```

---

## 四、关键数据结构对比

| 数据结构 | PCG | BCG |
|---|---|---|
| **图容器** | `torch.cuda.CUDAGraph` × (num_subgraphs × num_sizes) | `BreakableCUDAGraph` × num_bs (内部多个 `cudaGraphExec`) |
| **每条目状态** | `ConcreteSizeEntry` (cuda_piecewise_backend.py:24) | `GraphBreakInfo` (breakable_cuda_graph.py:46) |
| **字段** | `runtime_shape, need_to_compile, use_cudagraph, compiled, runnable, cudagraph, output` | `func (replay_fn), output, graph_handle` |
| **内存池** | 全局共享 `global_graph_memory_pool` (piecewise_cuda_graph_runner.py:124) | 全局共享 `get_global_graph_memory_pool()` (cuda_graph_runner.py 继承) |
| **split 信息** | 编译期确定 `split_ops` 列表 (compilation_config.py:5) | 运行期 `_captured_graphs_var` 动态增长 (breakable_cuda_graph.py:55) |
| **静态 Buffer** | `PrefillInputBuffers` (piecewise_cuda_graph_runner.py:72) | `DecodeInputBuffers` (cuda_graph_runner.py:129) |

---

## 五、NPU 适配分析

### PCG — 已适配 (生产可用)

PCG 的 NPU 适配已完成，核心文件为 `npu_piecewise_backend.py`，通过继承 `CUDAPiecewiseBackend` 实现最小化改动：

- **`torch.cuda.CUDAGraph`** → **`torch.npu.NPUGraph()`** (`npu_piecewise_backend.py`)
- **`torch.cuda.graph()`** → **`torch.npu.graph()`** (`npu_piecewise_backend.py`)
- **`torch.cuda.empty_cache`** → **`torch.npu.empty_cache`** (`npu_piecewise_backend.py`)
- 后台选择逻辑在 `backend.py:52-57`：`is_npu()` 时自动选择 `NPUPiecewiseBackend`
- NPU 图运行器位于 `hardware_backend/npu/graph_runner/npu_graph_runner.py`，继承 `CudaGraphRunner`

**适配可行原因**: PCG 仅依赖 PyTorch 高层 API (`torch.cuda.CUDAGraph`, `torch.cuda.graph()`)，这些 API 在 `torch_npu` 中有对应的 `torch.npu` 实现，接口语义一致。

### BCG — 未适配 (困难)

BCG 直接依赖 NVIDIA CUDA Runtime C API 的 Python 绑定 (`cuda.bindings.runtime`)，无 NPU 等价物：

| BCG 依赖的 API | 用途 | NPU 等价物 |
|---|---|---|
| `rt.cudaStreamBeginCapture` | 开始 stream capture | 无 |
| `rt.cudaStreamEndCapture` | 结束 stream capture | 无 |
| `rt.cudaGraphInstantiateWithFlags` | 实例化图 | 无 |
| `rt.cudaGraphLaunch` | 启动图执行 | 无 |
| `rt.cudaGraphDestroy` | 销毁图 | 无 |
| `rt.cudaGraphExecDestroy` | 销毁图执行句柄 | 无 |
| `rt.cudaStreamGetCaptureInfo` | 查询 capture 状态 | 无 |

**核心障碍**: BCG 需要精细控制 CUDA stream capture 的 begin/end 来实现多段 capture，这是 NVIDIA CUDA Runtime 的专有能力。`torch_npu` 虽然提供了 `NPUGraph` 高层 API，但没有暴露底层 stream capture 控制接口。要适配 BCG 到 NPU，需要华为在 `torch_npu` 中实现等价的底层 stream capture API。



Plan: 优化 Prompt 以触发专业 Agent/Skill

 Context

 用户希望优化这条 prompt：

 ▎ "这个sglang项目的pieceweise cuda graph和breakable cuda graph特性分别是怎么实现的？两者有没有关联与区别？华为ascend npu能适配吗？"

 使其能更好地触发已配置的专业 agents 和 skills，获得更高质量的回答。

 为什么原 prompt 效果一般

 1. 太笼统 — "怎么实现的"是开放问题，没有指明需要什么深度
 2. 没有触发意图信号 — 缺少 "research"、"explore"、"analyze" 等关键词
 3. 三个子问题混在一起 — PCG 实现、BCG 实现、NPU 适配，每个适合不同 agent/skill
 4. 没有指定输出格式 — 不知道要代码分析文档、对比表格、还是架构图

 可用的 Agents/Skills 及触发方式

 ┌───────────────────────┬──────────────────────────────────────────────┬───────────────────────────────────────┐
 │      Agent/Skill      │                   触发方式                   │              适用子问题               │
 ├───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────────┤
 │ Explore agent         │ 自动（探索性代码搜索时）                     │ PCG/BCG 代码实现细节                  │
 ├───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────────┤
 │ deep-research         │ /deep-research 或 "深度研究/research"        │ CUDA Graph 技术原理、NPU 适配可行性   │
 ├───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────────┤
 │ codebase-onboarding   │ /codebase-onboarding 或 "帮我理解这个代码库" │ sglang 整体架构概览                   │
 ├───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────────┤
 │ documentation-lookup  │ /documentation-lookup 或问 "怎么用/如何配置" │ PyTorch CUDA Graph API 文档           │
 ├───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────────┤
 │ exa-search            │ /exa-search 或 "搜索/查找最新"               │ 华为 Ascend NPU 最新的 torch_npu 支持 │
 ├───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────────┤
 │ architect agent       │ 自动（架构决策时）                           │ PCG vs BCG 架构对比分析               │
 ├───────────────────────┼──────────────────────────────────────────────┼───────────────────────────────────────┤
 │ python-reviewer agent │ 自动（Python 代码审查时）                    │ 审查 PCG/BCG 代码质量                 │
 └───────────────────────┴──────────────────────────────────────────────┴───────────────────────────────────────┘

 优化后的 Prompt（三种场景）

 场景 A：单条精炼 Prompt（推荐）

 /research sglang 项目中 Piecewise CUDA Graph (PCG) 和 Breakable CUDA Graph (BCG)
 的实现架构对比分析。请从以下维度深入分析：

 1. 代码实现：分别探索 PCG 和 BCG 的核心类、数据结构、调用链
    （关注 piecewise_cuda_graph_runner.py、breakable_cuda_graph.py）
 2. 架构对比：分段策略、编译方式、内存管理、动态形状处理
 3. 华为 Ascend NPU 适配性：检查 npu_piecewise_backend.py 的实现，
    分析 BCG 的 CUDA stream capture API 在 torch_npu 上的可用性

 请搜索 Ascend NPU 最新的 CUDA Graph 等效 API 支持情况。
 输出对比表格和架构图（用 ASCII/Mermaid）。

 触发的 Skills/Agents:
 - /research → deep-research skill（主动触发）
 - 代码探索关键词 → Explore agent（自动）
 - "搜索 Ascend NPU" → exa-search skill（自动）
 - "架构对比" → architect agent（自动）

 场景 B：拆分为多条 Prompt（最精确控制）

 Prompt 1（代码探索）：
 /explore 深入分析 sglang 中 Piecewise CUDA Graph 的完整实现：
 - 核心类 PiecewiseCudaGraphRunner 和 CUDAPiecewiseBackend
 - 图分割点（SPLIT_OPS）和编译流程
 - capture 和 replay 的完整调用链
 - KV cache 管理和 padding 策略

 Prompt 2（代码探索）：
 /explore 深入分析 sglang 中 Breakable CUDA Graph 的完整实现：
 - BreakableCUDAGraph 和 BreakableCUDAGraphCapture 类
 - @eager_on_graph 装饰器和 break_graph() 的工作机制
 - stream fork/join 跟踪机制
 - 与标准 CUDA Graph 的集成方式

 Prompt 3（深度研究）：
 /deep-research 华为 Ascend NPU 对 CUDA Graph 的支持现状：
 1. torch_npu 中 NPUGraph 的 API 完整度（对比 torch.cuda.CUDAGraph）
 2. Stream capture API（cudaStreamBeginCapture 等）是否已实现
 3. SGLang 已有的 NPU 适配（npu_piecewise_backend.py）策略
 4. Ascend CANN 软件栈中是否有 CUDA Graph 等效机制


 Prompt 4（架构对比）：
 基于前面的分析结果，对比 PCG 和 BCG 的架构差异，输出：
 1. 对比表格（分段策略、编译方式、内存管理、动态形状、适用阶段）
 2. 两者是否互补，能否共存
 3. NPU 适配的关键瓶颈和可能的解决方案

 场景 C：最简触发（日常快速使用）

 帮我深入分析 sglang 中 PCG 和 BCG 的实现差异，
 以及华为 Ascend NPU 能否适配。请搜索相关代码和最新资料。

 关键词 "深入分析" + "搜索" 会自动触发 Explore agent 和 exa-search skill。

 关键触发词速查表

 ┌───────────────────────┬──────────────────────────────────────────┐
 │       想触发的        │             在 Prompt 中加入             │
 ├───────────────────────┼──────────────────────────────────────────┤
 │ deep-research         │ /deep-research 或 "深度研究/research"    │
 ├───────────────────────┼──────────────────────────────────────────┤
 │ codebase-onboarding   │ /codebase-onboarding 或 "帮我理解代码库" │
 ├───────────────────────┼──────────────────────────────────────────┤
 │ exa-search            │ /exa-search 或 "搜索最新/查找资料"       │
 ├───────────────────────┼──────────────────────────────────────────┤
 │ Explore agent         │ "探索代码"/"搜索实现"/"/explore"         │
 ├───────────────────────┼──────────────────────────────────────────┤
 │ architect agent       │ "架构对比"/"设计分析"                    │
 ├───────────────────────┼──────────────────────────────────────────┤
 │ python-reviewer agent │ "审查代码"/"review 代码质量"             │
 ├───────────────────────┼──────────────────────────────────────────┤
 │ docs-lookup           │ "怎么用"/"API 文档"/"library 用法"       │
 └───────────────────────┴──────────────────────────────────────────┘

 验证方法

 用优化后的 prompt 开一个新 session，观察：
 1. Skill 工具是否被调用（主动触发的 skills）
 2. Agent 工具是否被调用，且 subagent_type 正确
 3. 回答质量是否比原 prompt 更全面、更有条理