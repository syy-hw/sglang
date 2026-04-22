# sglang CUDA Graph 学习教程

> 基于 commit: `c25f00630`
> 生成日期: 2026-04-21

---

## 教程目录

### 阶段一：[CUDA Graph 基础与 sglang 架构全景](./01_cuda_graph_basics_and_architecture.md)

从零开始理解 CUDA Graph 的 capture-replay 机制，以及 sglang 如何在 decode/extend 双路径中使用它。

- Kernel launch overhead 与 CUDA Graph 的核心思想
- 静态 shape / 静态内存地址约束的类比解释
- CudaGraphRunner 的初始化链路和状态机
- decode 与 extend 路径的差异

### 阶段二：[Piecewise CUDA Graph (PCG) 深入](./02_piecewise_cuda_graph_tutorial.md)

理解 PCG 如何通过模型切分解决 extend 阶段的动态 token 数问题。

- split_points 机制：为什么按层切分、如何注册 split op
- torch.compile + SGLangBackend 的编译流程
- 递增粒度的 capture size 生成与 bisect padding 策略
- 从输入 token 到输出的完整 replay 数据流

### 阶段三：[Breakable CUDA Graph (BCG) 深入 + 三者对比](./03_breakable_cuda_graph_tutorial.md)

理解 BCG 的 graph break 机制，以及标准 CG、PCG、BCG 三者的定位对比。

- `@eager_on_graph` 装饰器的三条执行路径
- captured segment 和 eager segment 的交替执行
- `--debug-cuda-graph` 的"三明治"适配技巧
- 标准 CG / PCG / BCG 全景对比表

---

## 参考文档

以下为项目已有的参考文档，本教程在其基础上补充教学视角：

- [piecewise_cuda_graph.md](../docs/advanced_features/piecewise_cuda_graph.md) — PCG 官方文档
- [breakable_cuda_graph.md](../docs/advanced_features/breakable_cuda_graph.md) — BCG 官方文档
- [pcg_vs_bcg_analysis.md](../docs/advanced_features/pcg_vs_bcg_analysis.md) — PCG vs BCG 详细流程与调用链分析
- [cuda_graph_for_multi_modal_encoder.md](../docs/advanced_features/cuda_graph_for_multi_modal_encoder.md) — ViT CUDA Graph 文档
