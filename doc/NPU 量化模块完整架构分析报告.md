# NPU 量化模块完整架构分析报告

## 1. 类继承图

```
                         ┌─────────────────────────────┐
                         │      QuantizationConfig      │ (base_config.py)
                         │  量化配置入口，注册量化方法名    │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │     ModelSlimConfig          │ (modelslim/modelslim.py:83)
                         │  NPU 专用量化配置              │
                         │  - get_linear_scheme()       │
                         │  - get_moe_scheme()          │
                         └──────────────┬──────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                                       │
     ┌──────────────┴──────────────┐       ┌───────────────┴──────────────┐
     │   ModelSlimLinearMethod     │       │   ModelSlimFusedMoEMethod    │
     │  (modelslim.py:261)         │       │  (modelslim.py:314)          │
     │  extends _NPULinearMethod   │       │  extends FusedMoEMethodBase  │
     │                             │       │                              │
     │  create_weights() → scheme  │       │  create_weights() → scheme   │
     │  process_weights() → scheme │       │  process_weights() → scheme  │
     │  apply() → scheme           │       │  apply() → scheme            │
     └─────────────────────────────┘       └──────────────────────────────┘
                    │                                       │
                    │ 委托到 layer.scheme                    │ 委托到 layer.scheme
                    ▼                                       ▼
     ┌──────────────────────────────┐    ┌────────────────────────────────┐
     │   <<abstract>>               │    │   <<abstract>>                 │
     │   BaseLinearScheme           │    │   BaseMoEScheme                │
     │   (base_scheme.py:16)        │    │   (base_scheme.py:55)          │
     ├──────────────────────────────┤    ├────────────────────────────────┤
     │ + create_weights()           │    │ + create_weights()             │
     │ + process_weights_after_     │    │ + process_weights_after_       │
     │   loading()                  │    │   loading()                    │
     │ + apply_weights()            │    │ + apply_weights()              │
     └──────────────┬───────────────┘    │ + create_moe_runner()          │
                    │                     └───────────────┬────────────────┘
                    │                                     │
         ┌──────────┴──────────┐              ┌───────────┼────────────┐
         │                     │              │           │            │
         ▼                     ▼              ▼           ▼            ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────┐ ┌──────────┐ ┌──────────┐
│ModelSlimW8A8Int8│  │ModelSlimW4A4Int4│  │W4A4 MoE  │ │W4A8 MoE  │ │W8A8 MoE  │
│(w8a8_int8.py:20)│  │(w4a4_int4.py:16)│  │(MoE)     │ │(MoE)     │ │(MoE)     │
│                 │  │                 │  │          │ │          │ │          │
│ kernel: ────────┼──┼─────────────────┼──┼──────────┼─┼──────────┼─┼──────┐   │
└─────────────────┘  └─────────────────┘  └──────────┘ └──────────┘ └──────┼───┘
         │                    │                  │           │            │   │
         ▼                    ▼                  ▼           ▼            ▼   ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    NPU Kernels (hardware_backend/npu/quantization/)          │
│                                                                              │
│  linear_method_npu.py:                                                       │
│  ┌──────────────────────────┐  ┌────────────────────────────────────┐        │
│  │ NPUW8A8Int8LinearMethod  │  │ NPUW8A8Int8DynamicLinearMethod     │        │
│  │ (静态 W8A8)              │  │ (动态 W8A8)                        │        │
│  └──────────────────────────┘  └────────────────────────────────────┘        │
│  ┌──────────────────────────┐                                                │
│  │ NPU_W4A4DynamicLinear    │                                                │
│  │ Method (动态 W4A4)       │                                                │
│  └──────────────────────────┘                                                │
│                                                                              │
│  fused_moe_method_npu.py:                                                    │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐              │
│  │NPUW4A4Int4Dynamic│ │NPUW8A8Int8Dynamic│ │NPUW4A8Int8Dynamic│              │
│  │MoEMethod         │ │MoEMethod         │ │MoEMethod         │              │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 2. Scheme ↔ Kernel 对应关系表

| Scheme 类 | 文件 | Kernel 类 | 文件 | 量化类型 |
|-----------|------|-----------|------|---------|
| `ModelSlimW8A8Int8` (is_dynamic=False) | `w8a8_int8.py` | `NPUW8A8Int8LinearMethod` | `linear_method_npu.py:21` | W8A8 静态 INT8 |
| `ModelSlimW8A8Int8` (is_dynamic=True) | `w8a8_int8.py` | `NPUW8A8Int8DynamicLinearMethod` | `linear_method_npu.py:79` | W8A8 动态 INT8 |
| `ModelSlimW4A4Int4` | `w4a4_int4.py` | `NPU_W4A4DynamicLinearMethod` | `linear_method_npu.py:114` | W4A4 动态 INT4 |
| `ModelSlimW4A4Int4MoE` | `w4a4_int4_moe.py` | `NPUW4A4Int4DynamicMoEMethod` | `fused_moe_method_npu.py:396` | W4A4 动态 MoE |
| `ModelSlimW4A8Int8MoE` | `w4a8_int8_moe.py` | `NPUW4A8Int8DynamicMoEMethod` | `fused_moe_method_npu.py:586` | W4A8 动态 MoE |
| `ModelSlimW8A8Int8MoE` | `w8a8_int8_moe.py` | `NPUW8A8Int8DynamicMoEMethod` | `fused_moe_method_npu.py:464` | W8A8 动态 MoE |

### Scheme 与 Kernel 的协作模式（以 `ModelSlimW8A8Int8` 为例）

```
Scheme (modelslim_w8a8_int8.py)              Kernel (linear_method_npu.py)
─────────────────────────────────              ─────────────────────────────
create_weights():                              (不负责权重创建)
  - 注册 weight (int8)
  - 注册 weight_scale (float)
  - 注册 weight_offset (float)
  - 如果非动态: 注册 input_scale,
    input_offset, quant_bias, deq_scale

process_weights_after_loading():               process_weights_after_loading():
  → self.kernel.process_weights_after_loading    - transpose weight
    (layer)                                       - npu_format_cast
                                                  - flatten scale/offset
                                                  - 创建 aclnn_input_scale 等

apply_weights():                               apply():
  → self.kernel.apply(layer, x, bias)            - npu_dynamic_quant(x) 或
                                                  - npu_quantize(x, ...)
                                                  - npu_quant_matmul(x, weight, ...)
```

### 核心分工

- **Scheme** = 权重注册 + 元数据描述（在 `layers/quantization/` 层）
- **Kernel** = 权重后处理 + 推理计算（在 `hardware_backend/npu/` 层）

## 3. 重构前后架构对比表

### 重构前（commit 894c0dc57, 2025-12-04 → 424a38007^）

```
hardware_backend/npu/quantization/
├── linear_method_npu.py    ← 215 行，包含 create_weights + apply
│   ├── NPUW8A8Int8LinearMethod      (create_weights + apply 都在这里)
│   └── NPUW8A8Int8DynamicLinearMethod (create_weights + apply 都在这里)
├── fused_moe_method_npu.py ← 916 行，巨大单文件
│   ├── NPUW4A8Int4DynamicMoEMethod
│   ├── NPUW4A16Int4DynamicMoEMethod
│   ├── NPUW8A8Int8DynamicMoEMethod
│   └── 各种 npu_fused_experts_* 辅助函数
└── modelslim.py            ← 250 行，Config + 量化方法选择全在一处
    ├── ModelSlimConfig (直接 return NPU kernel 类)
    └── get_quant_method() 直接 if/else 判断返回不同 kernel
```

**问题：**
1. `create_weights()` 散布在 kernel 文件中 → kernel 既管权重注册又管推理计算
2. `get_quant_method()` 用 if/else 硬编码判断量化类型
3. Config 直接返回 kernel 实例，没有中间抽象层
4. 新增量化类型需同时修改 Config + kernel 文件

### 重构后第一阶段（commit 424a38007, 2026-01-14）

```
layers/quantization/modelslim/         ← 新目录，从 hardware_backend/ 分离
├── modelslim.py                       ← Config + Linear/MoE Method 委托层
│   ├── ModelSlimConfig
│   │   ├── get_linear_scheme()  ← 新增：返回 Scheme 对象
│   │   └── get_moe_scheme()     ← 暂未实现（MoE 还在 modelslim_moe.py）
│   ├── ModelSlimLinearMethod     ← create_weights/apply 委托给 scheme
│   └── (ModelSlimMoEMethod 在 modelslim_moe.py)
├── modelslim_moe.py                   ← 中间态，MoE 还没用 Scheme 模式
│   ├── ModelSlimMoEMethod
│   ├── ModelSlimW4A8Int8MoE      ← 直接包含 create_weights
│   └── ModelSlimW8A8Int8MoE
└── schemes/
    ├── modelslim_scheme.py            ← 抽象基类
    │   ├── ModelSlimLinearScheme
    │   └── ModelSlimMoEScheme
    ├── modelslim_w8a8_int8.py         ← Linear Scheme
    └── modelslim_w4a4_int4.py         ← Linear Scheme

hardware_backend/npu/quantization/
├── linear_method_npu.py    ← 从 215 行减到 ~137 行，只保留 process + apply
└── fused_moe_method_npu.py ← 保持大文件
```

### 重构后第二阶段（commit aeca7d348, 2026-02-17）

```
layers/quantization/modelslim/
├── modelslim.py                       ← 统一 Config，合并 MoE 逻辑
│   ├── ModelSlimConfig
│   │   ├── get_linear_scheme()  ← 返回 Scheme
│   │   └── get_moe_scheme()     ← 返回 Scheme (新增！)
│   ├── ModelSlimLinearMethod
│   └── ModelSlimFusedMoEMethod  ← 替代了 modelslim_moe.py 中的类
├── (modelslim_moe.py 已删除)
└── schemes/
    ├── modelslim_scheme.py            ← ModelSlimMoEScheme 新增 create_moe_runner
    ├── modelslim_w8a8_int8.py
    ├── modelslim_w4a4_int4.py
    ├── modelslim_w4a4_int4_moe.py     ← 新增 MoE Scheme
    ├── modelslim_w4a8_int8_moe.py     ← 从 w4a8 重构而来
    └── modelslim_w8a8_int8_moe.py     ← 从 modelslim_moe.py 提取
```

## 4. 关键 Git Commits 时间线

| 日期 | Hash | 描述 | 阶段 |
|------|------|------|------|
| 2025-12-04 | `894c0dc57` | [NPU][1/N] NPU basic functions refactor | 初始版本，无 Scheme |
| 2025-12-18 | `d36299ad7` | perf update with kvcache nz & w4a8 quant | 新增 W4A8 |
| 2026-01-14 | **`424a38007`** | **NPU quantization refactoring & more quantization formats** | **重构第一阶段：引入 Scheme** |
| 2026-01-27 | `5297b02c8` | Wan2.2-T2V Diffusion modelslim quantization | 新增 Diffusion 支持 |
| 2026-02-17 | **`aeca7d348`** | **[3/N] Quantization Refactor: ModelSlim MoE schemes** | **重构第二阶段：MoE Scheme** |
| 2026-03-04 | `ed42af99a` | w4a4 MoE layer support | 新增 W4A4 MoE |
| 2026-04-03 | `1b4933d45` | adapt w2 quant layer for Minimax2.5 | 模型适配 |

## 5. 重构动机总结

### 动机 1：关注点分离（Separation of Concerns）

**重构前：** `linear_method_npu.py` 中的 kernel 类同时包含 `create_weights()` 和 `apply()`，权重注册逻辑（Scheme 层关注点）与推理计算逻辑（Kernel 层关注点）混在一起。

**重构后：** Scheme 负责"权重长什么样"（类型、形状、scale、offset 等元数据），Kernel 负责"怎么计算"（transpose、format cast、CANN 算子调用）。`linear_method_npu.py` 从 215 行缩减到 144 行。

### 动机 2：消除 if/else 硬编码分发

**重构前：** `ModelSlimConfig.get_quant_method()` 用多个 if/else 判断 `is_dynamic`、`is_moe_w4_dynamic` 等布尔字段，直接返回不同的 kernel 实例。添加新量化类型需要在这个方法里加更多 if/else。

**重构后：** `get_linear_scheme()` 和 `get_moe_scheme()` 用配置驱动的 scheme 列表做匹配：

```python
linear_quant_schemes = [
    ("W4A4_DYNAMIC", ModelSlimW4A4Int4),
    ("W8A8", ModelSlimW8A8Int8),
    ("W8A8_DYNAMIC", ModelSlimW8A8Int8),
]
```

新增量化类型只需添加一个 Scheme 类 + 在列表中注册一行。

### 动机 3：与 vLLM Compressed Tensors 架构对齐

代码注释明确标注了来源：`# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors`。sglang 的 Scheme 模式复用了 vLLM 的 compressed_tensors 架构，使得两边的代码结构更接近，降低维护成本。`base_scheme.py` 中的 `BaseLinearScheme` 和 `BaseMoEScheme` 是通用抽象，NPU 的 `ModelSlimLinearScheme` 和 NVIDIA 侧的 `CompressedTensorsScheme` 都继承自它们。

### 动机 4：支持新增量化类型的可扩展性

新增一个量化类型的流程，从"改 3 个文件"变为"加 1 个 Scheme 文件 + 注册 1 行"：

| 步骤 | 重构前 | 重构后 |
|------|--------|--------|
| 1 | 在 `linear_method_npu.py` 中添加 kernel 类（含 create_weights + apply） | 在 `linear_method_npu.py` 中添加 kernel 类（仅 process + apply） |
| 2 | 在 `modelslim.py` 的 `get_quant_method()` 中加 if/else | 在 `schemes/` 中新建 Scheme 文件（仅 create_weights + 委托） |
| 3 | 在 `modelslim.py` 中修改 `__init__` 中的布尔判断 | 在 `modelslim.py` 的 scheme 列表中加一行 |
| 总计 | 改 2 个文件，逻辑交织 | 改 2 个文件，逻辑隔离 |

### 动机 5：文件位置的语义正确性

重构前 `modelslim.py` 在 `hardware_backend/npu/quantization/` 下，但它的 `ModelSlimConfig` 是框架层的量化配置（属于 `layers/quantization/` 职责），不是硬件后端代码。重构后移动到 `layers/quantization/modelslim/`，Scheme 与其他量化配置（compressed_tensors、awq 等）平级。

---

**Scheme 的核心设计模式一句话总结：** Scheme 是"权重注册表"——它定义了每种量化类型的权重布局（参数名、数据类型、形状），并在运行时将实际计算委托给 NPU kernel。两者通过 `layer.scheme = scheme_instance` 和 `self.kernel = kernel_instance` 双向组合。
