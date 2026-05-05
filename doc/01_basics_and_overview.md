# 阶段 1：量化基础与 sglang 量化全景

## 目录

- [1. 量化是什么？为什么需要量化？](#1-量化是什么为什么需要量化)
- [2. 量化基本概念](#2-量化基本概念)
- [3. 离线量化 vs 在线量化](#3-离线量化-vs-在线量化)
- [4. 主流量化方法原理速览](#4-主流量化方法原理速览)
- [5. sglang 支持的量化方法总览](#5-sglang-支持的量化方法总览)
- [6. Ascend NPU 平台支持状态](#6-ascend-npu-平台支持状态)
- [学习检查点](#学习检查点)

---

## 1. 量化是什么？为什么需要量化？

### 一句话解释

**量化 = 用更少的 bit 来表示模型的权重和/或激活值，从而减少内存占用和加速推理。**

### 为什么需要量化？

大语言模型（LLM）的参数量巨大。以 Llama-3 70B 为例：

| 精度 | 每参数占用 | 总显存需求 |
|------|-----------|-----------|
| FP32 (32 bit) | 4 bytes | ~280 GB |
| FP16/BF16 (16 bit) | 2 bytes | ~140 GB |
| FP8 (8 bit) | 1 byte | ~70 GB |
| INT8 (8 bit) | 1 byte | ~70 GB |
| INT4 (4 bit) | 0.5 bytes | ~35 GB |

量化让同样的模型在更少的显存上运行，或者在同等显存上跑更大的模型。代价是**精度损失**，但好的量化方法可以把损失降到几乎不可察觉。

### 量化涉及的三个维度

```
WxAy  记法：
  W = Weight（权重）
  A = Activation（激活值）
  x/y = bit 数

例如：
  W8A8 = 权重 8bit + 激活 8bit
  W4A16 = 权重 4bit + 激活 16bit（只量化权重，激活保持原精度）
  W8A8_FP8 = 权重和激活都用 FP8 格式
```

---

## 2. 量化基本概念

### 2.1 浮点数表示

计算机中的浮点数由三部分组成：**符号位 (sign) + 指数位 (exponent) + 尾数位 (mantissa)**

```
FP32: 1 sign + 8 exponent + 23 mantissa = 32 bit
FP16: 1 sign + 5 exponent + 10 mantissa = 16 bit
BF16: 1 sign + 8 exponent + 7 mantissa  = 16 bit  (指数范围同 FP32)
FP8 E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bit
FP8 E5M2: 1 sign + 5 exponent + 2 mantissa = 8 bit
INT8: 8 bit 整数（需要 scale 来映射到浮点范围）
```

### 2.2 FP8 的两种格式

FP8 是当前最热门的在线量化格式，有两种变体：

| 格式 | Sign | Exponent | Mantissa | 范围 | 精度 | 用途 |
|------|------|----------|----------|------|------|------|
| **E4M3** | 1 bit | 4 bit | 3 bit | ±448 | 较高 | **权重 + 前向激活** |
| **E5M2** | 1 bit | 5 bit | 2 bit | ±57344 | 较低 | **梯度**（训练用） |

在推理场景中，我们主要使用 **E4M3**。

```
量化公式（均匀量化）：
  quantized_value = round(float_value / scale)

  其中 scale = max_abs_value / max_representable_value

反量化公式：
  float_value ≈ quantized_value * scale
```

### 2.3 量化的粒度

量化可以按不同粒度进行：

| 粒度 | 含义 | 精度 | 开销 |
|------|------|------|------|
| **per-tensor** | 整个张量一个 scale | 低 | 最小 |
| **per-channel/per-row** | 每行/列一个 scale | 中 | 适中 |
| **per-block (block-wise)** | 每 N 个元素一个 scale | 高 | 较大 |
| **per-group** | 每 group 一个 scale | 高 | 较大 |

sglang 中的 `blockwise_int8` 就是 block 级粒度量化（默认 block_shape=[128, 128]）。

---

## 3. 离线量化 vs 在线量化

这是理解 sglang 量化模块的**关键区分**。

### 离线量化（Offline / Post-Training Quantization, PTQ）

```
原始模型（FP16/BF16）
    ↓
使用校准数据集（calibration dataset）运行推理
    ↓
统计各层权重的分布（max、min、percentile 等）
    ↓
计算最优量化参数（scale、zero-point）
    ↓
保存量化后的权重和参数到磁盘
    ↓
推理时直接加载量化权重（无需 --quantization 参数）
```

**特点：**
- 量化发生在推理之前，是一个独立的预处理步骤
- 需要校准数据，但不需要训练
- 量化后的模型保存为特定格式（如 GPTQ 的 `.pt` 文件、AWQ 的 safetensors）
- 精度通常优于在线量化
- 推理时**自动检测**：sglang 从模型配置文件中读取量化信息

**对应方法：** GPTQ、AWQ、GGUF、BitsAndBytes、ModelOpt 等

### 在线量化（Online / Dynamic Quantization）

```
原始模型（FP16/BF16 checkpoint）
    ↓
模型加载时：process_weights_after_loading() 将权重一次性量化为低精度
  weight_fp16 → weight_fp8 + weight_scale （只做一次，缓存）
    ↓
每次 forward（apply）：
  1. 动态量化激活值：activation → activation_fp8 + scale（每次重算）
  2. 低精度矩阵乘：weight_fp8 @ activation_fp8
  3. 乘以 scale 得到 float 输出
```

**特点：**
- 不需要预处理，加载原始 checkpoint 后自动量化
- 权重只量化一次（加载时），激活值每次 forward 动态计算 scale
- 使用更简单，但精度可能略低于离线量化
- 需要 `--quantization` 参数显式指定

**对应方法：** fp8、w8a8_fp8、w8a8_int8、blockwise_int8 等

### 对比表

| 维度 | 离线量化 | 在线量化 |
|------|---------|---------|
| 时机 | 推理前（离线工具） | 模型加载时（process_weights_after_loading） |
| 校准数据 | 需要 | 不需要 |
| 精度 | 较高 | 较低（但 fp8 已足够好） |
| 易用性 | 需要额外步骤 | 直接加参数即可 |
| 磁盘占用 | 小（低精度 checkpoint） | 大（原始精度 checkpoint） |
| sglang 参数 | 自动检测 | `--quantization fp8` |
| 推荐度 | **推荐** | 备选 |

> **sglang 官方建议：** 推荐使用离线量化，效果更好。参考 `docs/advanced_features/quantization.md:L12`

---

## 4. 主流量化方法原理速览

### 4.1 FP8 量化

```
原理：直接将 FP16/BF16 权重/激活转换为 FP8（E4M3）格式
粒度：通常 per-tensor 或 per-channel
优点：硬件原生支持（H100+），计算速度快
缺点：需要硬件支持 FP8（A100 不支持，H100 支持）
```

### 4.2 INT8 量化（W8A8）

```
原理：将浮点权重/激活量化为 INT8 整数
公式：W_int8 = round(W_fp / scale), scale = max(|W|) / 127
粒度：per-tensor 或 per-block（128x128）
优点：硬件支持广泛
缺点：均匀量化，对异常值敏感
```

### 4.3 GPTQ

```
原理：基于 Hessian 矩阵的后训练量化
      逐列量化权重，利用二阶信息补偿量化误差
粒度：per-group（通常 group_size=128）
精度：W4A16 或 W3A16
优点：精度高，支持极低比特（3bit、4bit）
缺点：需要校准数据，量化过程较慢
```

### 4.4 AWQ (Activation-aware Weight Quantization)

```
原理：观察激活值分布，识别"重要"权重通道
      对重要通道进行缩放后再量化，保护关键信息
粒度：per-group
精度：通常 W4A16
优点：比 GPTQ 更快，精度相当
缺点：需要校准数据
```

### 4.5 MXFP4

```
原理：基于 Microscaling (MX) 标准的 FP4 格式
      使用共享指数（shared exponent）减少存储
精度：W4A8 或 W4A4
优点：极致压缩（4bit），有硬件加速支持
缺点：需要特定硬件（CDNA3/CDNA4、Blackwell）
```

### 4.6 方法对比总表

| 方法 | 类型 | 精度 | 校准数据 | 硬件要求 | 推荐场景 |
|------|------|------|---------|---------|---------|
| **FP8** | 在线 | W8A8 | 不需要 | H100+/MI300X+ | 首选在线量化 |
| **W8A8_INT8** | 在线 | W8A8 | 不需要 | 通用 | 不支持 FP8 的硬件 |
| **blockwise_int8** | 在线 | W8A8 | 不需要 | 通用 | 精度敏感场景 |
| **GPTQ** | 离线 | W4A16 | 需要 | 通用 | 极致压缩 |
| **AWQ** | 离线 | W4A16 | 需要 | 通用 | 极致压缩（比 GPTQ 快） |
| **GGUF** | 离线 | 多种 | 不需要 | 通用 | CPU 推理 / llama.cpp 兼容 |
| **ModelOpt** | 离线 | W4/W8 | 需要 | 特定 | NVIDIA 官方工具链 |
| **MXFP4** | 在线/离线 | W4A4 | 可选 | 特定 | 极致压缩 + 硬件加速 |

---

## 5. sglang 支持的量化方法总览

以下数据来自 `docs/advanced_features/quantization.md`（sglang 官方文档）和源码 `python/sglang/srt/layers/quantization/__init__.py:L62-L88`。

### 5.1 代码中的注册表

sglang 在 `__init__.py` 中维护了一个量化方法注册表 `QUANTIZATION_METHODS`：

```python
# python/sglang/srt/layers/quantization/__init__.py:L62-L88
BASE_QUANTIZATION_METHODS = {
    "fp8": Fp8Config,
    "mxfp8": Fp8Config,
    "blockwise_int8": BlockInt8Config,
    "modelopt": ModelOptFp8Config,
    "modelopt_fp8": ModelOptFp8Config,
    "modelopt_fp4": ModelOptFp4Config,
    "modelopt_mixed": ModelOptMixedPrecisionConfig,
    "w8a8_int8": W8A8Int8Config,
    "w8a8_fp8": W8A8Fp8Config,
    "awq": AWQConfig,
    "awq_marlin": AWQMarlinConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    "gptq": GPTQConfig,
    "gptq_marlin": GPTQMarlinConfig,
    "moe_wna16": MoeWNA16Config,
    "compressed-tensors": CompressedTensorsConfig,
    "qoq": QoQConfig,
    "w4afp8": W4AFp8Config,
    "petit_nvfp4": PetitNvFp4Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    "quark": QuarkConfig,
    "auto-round": AutoRoundConfig,
    "modelslim": ModelSlimConfig,
    "quark_int4fp8_moe": QuarkInt4Fp8Config,
}
```

每个字符串键（如 `"fp8"`）就是 `--quantization` 参数接受的值，对应的值（如 `Fp8Config`）是该方法的配置类。

### 5.2 平台兼容性表（来自官方文档）

| 方法 | NVIDIA GPUs | AMD GPUs (MI300X/MI325X/MI350X) | Ascend NPUs (A2/A3) |
|------|:-----------:|:-------------------------------:|:-------------------:|
| `fp8` | Yes | Yes | WIP |
| `mxfp4` | Yes | Yes | WIP |
| `blockwise_int8` | Yes | Yes | No |
| `w8a8_int8` | Yes | Yes | No |
| `w8a8_fp8` | Yes | Yes | No |
| `awq` | Yes | Yes | Yes (A2/A3) |
| `gptq` | Yes | Yes | Yes (A2/A3) |
| `gguf` | Yes | - | Yes (A2/A3) |
| `bitsandbytes` | Yes | - | No |
| `modelopt_fp8` | Yes | - | No |
| `modelopt_fp4` | Yes | - | No |
| `compressed-tensors` | Yes | Yes | Yes (A2/A3) |

---

## 6. Ascend NPU 平台支持状态

以下信息来自 `docs/platforms/ascend/ascend_npu_quantization.md`。

### 6.1 NPU 上的量化方法支持

Ascend NPU 的量化路径与 GPU 不同，走的是独立的 `hardware_backend/npu/quantization/` 目录。

#### ModelSlim / Compressed-tensors 量化（W4A4, W8A8）

| 量化方案 | 层类型 | A2 | A3 | A5 |
|---------|--------|:--:|:--:|:--:|
| W4A4 dynamic | Linear | Yes | Yes | TBD |
| W8A8 static | Linear | Yes | Yes | TBD |
| W8A8 dynamic | Linear | Yes | Yes | TBD |
| MXFP8 | Linear | No | No | WIP |
| W4A4 dynamic | MoE | Yes | Yes | TBD |
| W4A8 dynamic | MoE | Yes | Yes | TBD |
| W8A8 dynamic | MoE | Yes | Yes | TBD |
| MXFP8 | MoE | No | No | WIP |

#### AWQ 量化（W4A16, W8A16）

| 量化方案 | 层类型 | A2 | A3 | A5 |
|---------|--------|:--:|:--:|:--:|
| W4A16 | Linear | Yes | Yes | TBD |
| W8A16 | Linear | Yes | Yes | TBD |
| W4A16 | MoE | Yes | Yes | TBD |

#### GPTQ 量化（W4A16, W8A16）

| 量化方案 | 层类型 | A2 | A3 | A5 |
|---------|--------|:--:|:--:|:--:|
| W4A16 | Linear | Yes | Yes | TBD |
| W8A16 | Linear | Yes | Yes | TBD |
| W4A16 MoE | MoE | Yes | Yes | TBD |
| W8A16 MoE | MoE | Yes | Yes | TBD |

### 6.2 NPU 量化的关键观察

1. **MXFP8 在 NPU 上是 WIP** — 这是 Ascend A5 系列的重点工作
2. **FP8 在通用量化表中标记为 WIP** — 但 NPU 有独立的 W8A8 dynamic 支持
3. **NPU 的量化实现不共用 GPU 路径** — 有独立的 `npu/quantization/` 目录
4. **Mix-bits 已支持** — 不同层可以独立定义量化方案
5. **GGUF 在 NPU 上特殊处理** — 权重在加载时预反量化为 FP16/BF16

### 6.3 作为 NPU 量化 Owner 需要关注的重点

```
高优先级（WIP / TBD）：
├── MXFP8 量化适配（A5 平台）
├── FP8 在线量化的 NPU 适配
└── A5 平台的全面量化验证

中优先级（已有支持，需维护）：
├── W8A8 dynamic（Linear + MoE）
├── W4A4 dynamic（Linear + MoE）
└── AWQ / GPTQ / Auto-round 的 NPU 路径维护

低优先级（了解即可）：
├── GGUF（已有独立实现）
└── Compressed-tensors（已有支持）
```

---

## 学习检查点

完成本阶段后，你应该能回答以下问题：

1. **离线量化和在线量化有什么区别？为什么 sglang 推荐离线量化？**
2. **FP8 E4M3 和 E5M2 的区别是什么？推理场景中主要用哪个？为什么？**
3. **sglang 的量化方法注册表在哪个文件？如何添加一个新的量化方法？**
   > 提示：`python/sglang/srt/layers/quantization/__init__.py:L62`
4. **Ascend NPU 上当前哪些量化方案是 WIP 状态？**
5. **W8A8 记法中 W 和 A 分别代表什么？8 代表什么？**

---

> 下一阶段：[02_architecture.md](./02_architecture.md) — 深入 sglang 量化模块的代码架构
