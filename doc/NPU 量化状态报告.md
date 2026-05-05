找到了。问题 #14424 是 **NPU 量化 2026 Q1 路线图**。现在我已全面了解情况。这是完整的报告。

---

# NPU 量化状态报告（基于 Issue #14424）

## 总览

- **标题**: [NPU] [Roadmap] NPU quantization 2026 Q1 Roadmap
- **作者**: OrangeRedeng (Артем Савкин)
- **状态**: OPEN（持续追踪中）
- **创建时间**: 2025-12-04
- **关联 RFC**: #21584 (MXFP8/MXFP4 on Ascend A5, by YChange01)

---

## 已完成

### 硬件后端 — 新层类型/特性

| 功能                     | PR     | 作者         | 合并时间   | 关键类/文件                                                                |
| ------------------------ | ------ | ------------ | ---------- | -------------------------------------------------------------------------- |
| W4A4 线性层支持          | #14504 | OrangeRedeng | 2025-12    | `npu/quantization/linear_method_npu.py` → `NPU_W4A4DynamicLinearMethod`    |
| W4A8 激活裁剪            | #14736 | OrangeRedeng | 2025-12    | `npu/quantization/linear_method_npu.py` → `NPUW4A8Int8DynamicLinearMethod` |
| W4A4 MoE 层              | #18924 | OrangeRedeng | 2026-03-11 | `npu/quantization/fused_moe_method_npu.py` → `NPUW4A4Int4DynamicMoEMethod` |
| Diffusion 模型量化       | #17936 | OrangeRedeng | 2026-02    | Wan2.2 ModelSlim 量化                                                      |
| W8A8 MoE 解码优化        | #19913 | heziiop      | 2026-03-17 | `dequant_swiglu_quant`, `moe_init_routing_v2`, `npu_moe_token_unpermute`   |
| Qwen3.5 量化 bugfix      | #21692 | OrangeRedeng | 2026-04-08 | 修复 Qwen3.5 在 NPU 上的量化错误                                           |
| Qwen3-Next W8A8 精度修复 | #21698 | ranjiewen    | 2026-04-27 | 修复精度问题                                                               |

### 重构 — 推理代码与量化框架分离

| 完成项                                | 状态      | 说明                                      |
| ------------------------------------- | --------- | ----------------------------------------- |
| W4A8 MoE 重构                         | ✅        | 分离 kernel 和 framework 代码             |
| W8A8 线性层重构                       | ✅        | 同上                                      |
| W8A8 MoE 重构                         | ✅        | 同上                                      |
| ModelSlim 线性层方案 (w8a8/w4a8/w4a4) | ✅        | `modelslim/schemes/` 目录下的 scheme 模式 |
| ModelSlim 自动配置检测                | ✅        | 无需 `--quantization modelslim` 参数      |
| Compressed-tensors MoE 方案           | ✅ #17503 | TamirBaydasov                             |
| ModelSlim MoE 方案                    | ✅ #17993 | TamirBaydasov                             |
| 单元测试                              | ✅        | w4a4 modelslim, w8a8 compressed tensors   |

### 第三方量化框架支持

| 框架                                | PR             | 作者            | 合并时间   | 支持范围                        |
| ----------------------------------- | -------------- | --------------- | ---------- | ------------------------------- |
| AWQ                                 | #10158         | —               | 2025-10    | 线性层 + MoE                    |
| Compressed-tensors (LLM Compressor) | #14504, #12759 | TamirBaydasov   | 2025-12    | 线性层 + MoE                    |
| GPTQ (Dense)                        | #15203         | 22dimensions    | 2026-01    | 线性层                          |
| GPTQ (MoE)                          | #16364         | Wenlin7150      | 2026-03-04 | MoE 层                          |
| Auto-round                          | #16699         | Wenlin7150      | 2026-01    | 线性层                          |
| GGUF                                | #17883         | TheKonka        | 2026-04-25 | Dense + MoE                     |
| ModelSlim                           | 多个 PR        | OrangeRedeng 等 | 持续       | 线性层 + MoE (主力离线量化框架) |

---

## 正在进行 (WIP)

### 硬件后端

| 功能                              | PR/Issue | 作者          | 状态                | 关键细节                                                                                                  |
| --------------------------------- | -------- | ------------- | ------------------- | --------------------------------------------------------------------------------------------------------- |
| **MXFP8 (W8A8) Qwen3 Dense**      | #22352   | TallMessiWu   | 🚧 OPEN             | 依赖 #20922；Ascend A5+ 专用；`NPUMXFP8LinearMethod`；在线 `--quantization mxfp8` + 离线 ModelSlim 双模式 |
| **MXFP4 (W4A8) Qwen3 Dense**      | #23650   | TallMessiWu   | 🚧 OPEN             | 依赖 #22352                                                                                               |
| **MXFP4 (W4A4) Qwen3 Dense**      | #23795   | TallMessiWu   | 🚧 OPEN             | 依赖 #22352 + #23650；`NPUSingleLevelMXFP4LinearMethod`；`float4_e2m1fn_x2` 格式；3D scale reshape        |
| **MXFP8 Diffusion (Wan2.2)**      | #20922   | TallMessiWu   | 🚧 OPEN             | Diffusion 模型的 MXFP8 基础设施                                                                           |
| **MXFP4 Diffusion (Wan2.2)**      | #22338   | TallMessiWu   | 🚧 OPEN             | W4A4 量化 Diffusion 模型                                                                                  |
| **W4A4 NZ 格式**                  | #20860   | OrangeRedeng  | 🚧 OPEN             | 支持非对齐(NZ)维度                                                                                        |
| **通信量化**                      | #20520   | egvenediktov  | 🚧 OPEN（部分合入） | Qwen3 TP 通信压缩                                                                                         |
| **KV Cache 量化**                 | —        | TamirBaydasov | 🚧 进行中           | 无公开 PR                                                                                                 |
| **Qwen3.5-MoE / Qwen3-Next 量化** | #22674   | Dmovic        | 🚧 OPEN             | 修复 GDN 层融合映射 `in_proj_qkvz`/`in_proj_ba`                                                           |
| **Soft FP8 (Atlas A2/A3)**        | #16644   | LinyuanLi0046 | 🚧 OPEN             | 为 A2/A3 平台增加 FP8 支持                                                                                |
| **强制动态量化 ModelSlim**        | #20135   | LinyuanLi0046 | 🚧 OPEN             | `force_dynamic_quant` for modelslimW8A8Int8                                                               |

### 重构

| 功能                      | PR     | 状态    |
| ------------------------- | ------ | ------- |
| WnAn MoE 重构（混合量化） | #17361 | 🚧 OPEN |

---

## 计划中 (Future / TBD)

### 硬件后端 (来自 #14424 和 #21584)

| 功能                                             | 来源           | 优先级推断                 |
| ------------------------------------------------ | -------------- | -------------------------- |
| **MXFP8/MXFP4 Qwen3 MoE** (W8A8, W4A8, W4A4)     | #21584 Phase 2 | 高 — RFC 明确列出          |
| **MXFP8/MXFP4 Qwen3.5 Dense** (W8A8, W4A8, W4A4) | #21584 Phase 3 | 中 — Phase 3               |
| **MXFP8/MXFP4 Qwen3.5 MoE** (W8A8, W4A8, W4A4)   | #21584 Phase 3 | 中                         |
| W4A8 线性层                                      | #14424 Future  | 低                         |
| 注意力量化                                       | #14424 Future  | 低                         |
| PDMIX                                            | #14424 Future  | 低 — 参考 vllm-ascend#4469 |
| Groupsize 支持 (w8a8, w4a4)                      | #14424 Future  | 中                         |
| 在线旋转 (FlatQuant)                             | #14424 Future  | 低                         |
| 向量量化 (QuIP#, AQLM)                           | #14424 Future  | 低                         |
| QoQ 框架支持                                     | #14424 Future  | 低                         |
| Quark 框架支持                                   | #14424 Future  | 低                         |
| 在线量化 (非 MXFP)                               | #14424 Future  | 中                         |

### MXFP8/MXFP4 分阶段实施路线图 (来自 #21584)

```
Phase 1: Diffusion (Wan2.2)           ← WIP (#20922, #22338)
  ├── W8A8 MXFP8
  └── W4A4 MXFP4

Phase 2: LLM (Qwen3)                  ← WIP (#22352, #23650, #23795)
  ├── Qwen3 Dense
  │   ├── W8A8 MXFP8  ← #22352
  │   ├── W4A8 MXFP4  ← #23650
  │   └── W4A4 MXFP4  ← #23795
  └── Qwen3 MoE                        ← 计划中
      ├── W8A8 MXFP8
      ├── W4A8 MXFP4
      └── W4A4 MXFP4

Phase 3: VLM (Qwen3.5)                ← 计划中
  ├── Qwen3.5 Dense (W8A8/W4A8/W4A4)
  └── Qwen3.5 MoE  (W8A8/W4A8/W4A4)
```

---

## 关联线索

### 关键贡献者

| GitHub ID         | 姓名         | 角色                      | 主要贡献                                                                     |
| ----------------- | ------------ | ------------------------- | ---------------------------------------------------------------------------- |
| **OrangeRedeng**  | Артем Савкин | 核心架构 + W4A4/MoE       | #14424 路线图作者，W4A4 线性/MoE 实现，ModelSlim 重构，Diffusion 量化        |
| **TallMessiWu**   | Junlin Wu    | MXFP8/MXFP4 开发          | 3 个 MXFP PR (#22352, #23650, #23795)，Diffusion MXFP (#20922, #22338)       |
| **TamirBaydasov** | —            | 重构 + Compressed-tensors | Compressed-tensors MoE (#17503), ModelSlim MoE (#17993), KV Cache 量化 (WIP) |
| **YChange01**     | YeChang Guo  | MXFP RFC + GPTQ MoE       | #21584 RFC 作者, GPTQ MoE 支持 (#16364)                                      |
| **TheKonka**      | 1874.        | GGUF 量化                 | NPU GGUF 支持 (#17883) + 文档 (#23845)                                       |
| **heziiop**       | —            | MoE 解码优化              | W8A8 MoE decode kernel (#19913)                                              |
| **Dmovic**        | Lei Ding     | Qwen3.5-MoE 适配          | GDN 层融合映射 (#22674)                                                      |
| **egvenediktov**  | —            | 通信量化                  | TP 通信压缩 (#20520)                                                         |
| **zhuyijie88**    | Yijie Zhu    | Qwen3-Next 特性           | W8A8 + MTP + disaggregation (#14391)                                         |
| **LinyuanLi0046** | LinyuanLi    | A2/A3 FP8                 | Soft FP8 (#16644), force dynamic quant (#20135)                              |

### 关联 PR/Issue 网络

```
#14424 (NPU Quantization Roadmap) ← 顶层追踪 Issue
├── #21584 (MXFP8/MXFP4 RFC, Ascend A5)
│   ├── #20922 (Diffusion MXFP8, OPEN)
│   ├── #22338 (Diffusion MXFP4, OPEN)
│   ├── #22352 (LLM Qwen3 Dense MXFP8, OPEN) ← 依赖 #20922
│   ├── #23650 (LLM Qwen3 Dense W4A8, OPEN)  ← 依赖 #22352
│   └── #23795 (LLM Qwen3 Dense W4A4, OPEN)  ← 依赖 #22352 + #23650
├── #18924 (W4A4 MoE, MERGED)
├── #20860 (W4A4 NZ, OPEN)
├── #17361 (WnAn MoE 重构, OPEN)
├── #20520 (通信量化, OPEN/部分合入)
├── #16644 (Soft FP8 A2/A3, OPEN)
├── #17883 (GGUF, MERGED)
├── #16364 (GPTQ MoE, MERGED)
└── #22674 (Qwen3.5-MoE GDN, OPEN)
```

### Reviewer / Maintainer 模式

核心 reviewer 组（在多个 MXFP PR 中重复出现）：
- **ping1jing2** — assignee，负责 review 和 merge
- **merrymercy** (Lianmin Zheng) — sglang 核心维护者
- **BBuf** — 量化方向 reviewer
- **ch-wan** (Cheng Wan) — 量化/MoE reviewer
- **HaiShaw**, **AniZpZ**, **Ying1123** — NPU 方向 reviewer

### 当前代码结构

```
NPU 量化核心文件：
python/sglang/srt/hardware_backend/npu/quantization/
├── linear_method_npu.py      ← NPU 线性层量化方法
└── fused_moe_method_npu.py   ← NPU MoE 层量化方法

ModelSlim 离线量化方案：
python/sglang/srt/layers/quantization/modelslim/
├── modelslim.py              ← ModelSlim 配置入口
└── schemes/
    ├── modelslim_scheme.py           ← 基类
    ├── modelslim_w8a8_int8.py        ← W8A8 INT8
    ├── modelslim_w4a8_int8_moe.py    ← W4A8 INT8 MoE
    ├── modelslim_w4a4_int4.py        ← W4A4 INT4
    ├── modelslim_w4a4_int4_moe.py    ← W4A4 INT4 MoE
    └── modelslim_w8a8_int8_moe.py    ← W8A8 INT8 MoE
```

> **注意**：MXFP PR 合入后会增加 `schemes/modelslim_mxfp8.py`、`schemes/modelslim_mxfp4.py`、`layers/quantization/npu_mxfp4_w4a4.py` 等新文件。

---

## 对你（NPU 量化模块所有者）的行动建议

1. **优先关注 MXFP 链条**：#22352 → #23650 → #23795 是有序依赖，这是当前最活跃的开发线。阅读这些 PR 的代码可以了解最新的 NPU 量化模式。

2. **Qwen3 MoE 的 MXFP 支持**是 #21584 明确列出但尚无 PR 的空白 — 这可能是一个好的切入点。

3. **ModelSlim 重构**（scheme 模式）是 #14424 的核心架构变更，理解 `schemes/` 目录的设计是做新量化方法的前提。

4. **关键人物关注**：OrangeRedeng（架构）、TallMessiWu（MXFP）、ping1jing2（review/merge）是与你的工作最相关的三个人。