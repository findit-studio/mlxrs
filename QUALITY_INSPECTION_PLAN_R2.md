# mlxrs 100-Round Quality Inspection — Round 2

**日期:** 2026-05-28
**代码版本:** 9139c94 (latest main)
**代码规模:** 168,912 行源码 + 52,001 行测试
**前一轮发现:** 已全部修复（#257, #258, #259, #260）

---

## 本轮目标

基于前一轮发现已修复的前提，进行更深入的 100 轮审查：
1. 验证所有修复是否正确实施
2. 检查大规模重构是否引入新问题
3. 对新功能（det/slogdet/median/scatter 等）进行完整审查
4. 深入之前覆盖不足的区域

---

## 专家团设计（每模块 4 团并行）

- **团 1: Faithfulness** — 对比上游源码，验证移植忠实度
- **团 2: Rust Quality** — 审查 rust-golden-skills 合规、API 设计、错误处理
- **团 3: Performance** — 分析性能、分配、SIMD 利用率
- **团 4: Adversarial** — 主动找 bug、反模式、回归

---

## Round 分配（25 模块 × 4 专家 = 100 rounds）

### Phase 1: 核心基础 (R1-R16)

**R1-R4: ffi/ + array/ 核心**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R1 | Faithfulness | ffi/ (94行, VectorArrayGuard/drain_vector/opt_array 共享) + array/mod.rs + array/construction.rs (1,199行) |
| R2 | Rust Quality | 同上 |
| R3 | Performance | 同上 |
| R4 | Adversarial | 同上 |

**R5-R8: ops/ 基础 + 新 ops**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R5 | Faithfulness | ops/arithmetic.rs + ops/comparison.rs + ops/reduction.rs + ops/shape.rs (含新增 moveaxis/roll/tile) |
| R6 | Rust Quality | ops/indexing.rs (含新增 scatter/slice_update) + ops/misc.rs + ops/logical.rs + ops/fft.rs |
| R7 | Performance | ops/linalg_basic.rs + ops/linalg_full.rs (含新增 det/slogdet) + ops/random.rs |
| R8 | Adversarial | ops/quantized.rs + ops/fast/metal_kernel.rs (验证 MetalKernelApplyConfig 封装修复) |

### Phase 2: LM 核心 (R9-R28)

**R9-R12: lm/cache/**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R9 | Faithfulness | cache/standard.rs + cache/rotating.rs + cache/chunked.rs (验证修复) |
| R10 | Rust Quality | cache/quantized.rs + cache/batch.rs + cache/batch_rotating.rs (验证新增测试) |
| R11 | Performance | cache/lru_cache.rs + cache/arrays.rs + cache/cache_list.rs + cache/prompt.rs |
| R12 | Adversarial | cache/persist.rs + cache/mask.rs + cache/mod.rs + cache/util.rs (验证 #260 测试覆盖) |

**R13-R16: lm/load + generate + session**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R13 | Faithfulness | lm/load.rs (3,297行, 验证错误处理迁移) + lm/gguf.rs (1,125行) |
| R14 | Rust Quality | lm/generate.rs (871行, 极度精简) + lm/session.rs (1,262行) + lm/cache_prompt.rs (673行) |
| R15 | Performance | lm/sample.rs + lm/speculative.rs + lm/perplexity.rs (332行) |
| R16 | Adversarial | lm/factory.rs (708行) + lm/model.rs + lm/tool_parsers.rs (64行) |

**R17-R20: lm/lora + quant + fuse**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R17 | Faithfulness | lm/lora.rs (4,502行, 大幅精简后) + lm/convert.rs |
| R18 | Rust Quality | lm/quant.rs (2,892行) + lm/fuse.rs |
| R19 | Performance | lora + quant 的热路径性能 |
| R20 | Adversarial | lora + quant 的边界情况和回归 |

**R21-R24: lm/tuner + nn**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R21 | Faithfulness | tuner/trainer.rs + tuner/datasets.rs + tuner/losses.rs (验证 #245 合规) |
| R22 | Rust Quality | tuner/optimizers/ (11 个优化器, 验证 #245 封装) |
| R23 | Performance | nn/switch.rs + nn/norm.rs + nn/rope.rs + nn/rope_scaling.rs + nn/attention.rs (5,483行) |
| R24 | Adversarial | nn/ 模块的数值精度和边界情况 |

### Phase 3: VLM (R25-R36)

**R25-R28: vlm/ 核心**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R25 | Faithfulness | vlm/load.rs (2,081行) + vlm/model.rs + vlm/mod.rs |
| R26 | Rust Quality | vlm/generate.rs (1,015行) + vlm/prompt.rs (2,056行) + vlm/inputs.rs (635行) |
| R27 | Performance | vlm/image.rs (2,542行, 含 SIMD 路径) + vlm/resize.rs (1,085行) |
| R28 | Adversarial | vlm/video.rs (947行) + vlm/feature_cache.rs (1,073行) |

### Phase 4: Audio (R29-R44)

**R29-R32: audio/dsp + features + io**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R29 | Faithfulness | audio/dsp.rs (2,192行, 大幅精简后) + audio/features.rs (1,570行) |
| R30 | Rust Quality | audio/io.rs (885行, 大幅精简后) + audio/load.rs (356行) |
| R31 | Performance | DSP + features 的 SIMD 路径性能 |
| R32 | Adversarial | audio/io 的 unsafe 块和边界情况 |

**R33-R36: audio/stt + tts + sts + playback + vad + lid + codec**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R33 | Faithfulness | audio/stt/ (2,497行) + audio/tts/ (1,786行) |
| R34 | Rust Quality | audio/sts/ (241行) + audio/playback/ (1,714行) + audio/vad/ (367行) + audio/lid/ (382行) + audio/codec/ (224行) |
| R35 | Performance | STT + TTS 热路径 |
| R36 | Adversarial | STS pipeline + playback 并发安全 |

### Phase 5: Tokenizer + Embeddings (R37-R48)

**R37-R40: tokenizer/**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R37 | Faithfulness | tokenizer/hf_tokenizer.rs + tokenizer/sentencepiece.rs (验证 PieceType 修复) |
| R38 | Rust Quality | tokenizer/stream.rs + tokenizer/generated.rs + tokenizer/encode_options.rs |
| R39 | Performance | tokenizer/tools.rs (4,786行, 工具调用解析) |
| R40 | Adversarial | tokenizer/chat.rs (验证 rust-golden-skills 合规) |

**R41-R44: embeddings/**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R41 | Faithfulness | embeddings/factory.rs (1,384行) + embeddings/config.rs |
| R42 | Rust Quality | embeddings/encode.rs + embeddings/pooling.rs (582行, 验证 #260 测试) |
| R43 | Performance | embeddings/colvision.rs (602行) + embeddings/similarity.rs |
| R44 | Adversarial | embeddings/model.rs + embeddings/fast.rs + embeddings/normalize.rs |

### Phase 6: SIMD + Transforms + Memory (R45-R56)

**R45-R48: simd/**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R45 | Faithfulness | simd/audio/ (4,128行, 8 文件) |
| R46 | Rust Quality | simd/vlm/ (2,177行, 4 文件) + simd/dispatch + simd/arch + simd/scalar |
| R47 | Performance | SIMD 性能基准验证 |
| R48 | Adversarial | SIMD MaybeUninit 安全和 NEON intrinsics |

**R49-R52: transforms/ + memory/ + error/**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R49 | Faithfulness | transforms/ (1,543行, 验证 closure 精简) |
| R50 | Rust Quality | memory/ (1,099行) + error.rs (验证 Backend/ShapeMismatch 迁移完成) |
| R51 | Performance | transforms + memory 性能 |
| R52 | Adversarial | error.rs 的新 typed payload 是否完整覆盖所有变体 |

### Phase 7: 底层类型 + IO (R53-R60)

**R53-R56: dtype + device + stream + shape**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R53 | Faithfulness | dtype.rs (验证 FromStr 实现) + device.rs + stream.rs |
| R54 | Rust Quality | shape.rs + ops_traits.rs + diagnostics.rs |
| R55 | Performance | dtype/device/stream 热路径 |
| R56 | Adversarial | Display::fmt NULL 检查验证 (#258 H1 修复) |

**R57-R60: io.rs**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R57 | Faithfulness | io.rs safetensors 部分 |
| R58 | Rust Quality | io.rs GGUF 部分 |
| R59 | Performance | io.rs 内存映射和分配 |
| R60 | Adversarial | io.rs TOCTOU 安全和错误处理 |

### Phase 8: 跨模块审查 (R61-R68)

**R61-R64: 一致性审查**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R61 | Faithfulness | 全局上游同步检查（mlx-lm/mlx-vlm/mlx-audio 最新变更） |
| R62 | Rust Quality | 全局 rust-golden-skills 合规验证 |
| R63 | Performance | 全局热路径性能分析 |
| R64 | Adversarial | 全局回归检查（重构是否引入新 bug） |

**R65-R68: 安全 + 依赖 + 构建**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R65 | Faithfulness | mlxrs-sys/ 子模块版本同步 |
| R66 | Rust Quality | Cargo.toml feature gate 设计 |
| R67 | Performance | 依赖审计 (osv-scanner) |
| R68 | Adversarial | xtask codegen 正确性 |

### Phase 9: 测试质量审查 (R69-R76)

**R69-R72: 测试覆盖验证**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R69 | Faithfulness | 验证 #260 修复：batch_rotating, tokenizer wrapper, TTS generate 测试 |
| R70 | Rust Quality | 验证 #260 修复：array/construction, array/conversion, autograd 测试 |
| R71 | Performance | 测试执行效率（有无不必要的慢测试） |
| R72 | Adversarial | 测试质量（有无虚假通过的测试、有无遗漏的 edge case） |

**R73-R76: 集成测试审查**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R73 | Faithfulness | lm/ 集成测试完整性 |
| R74 | Rust Quality | vlm/ + audio/ 集成测试完整性 |
| R75 | Performance | tokenizer/ + embeddings/ 集成测试 |
| R76 | Adversarial | ui-tests (compile-fail) 正确性 |

### Phase 10: 深度对抗审查 (R77-R88)

**R77-R80: 最高风险模块深度审查**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R77 | Adversarial | lm/cache/ 全部 13 文件（最复杂的模块，10,890行） |
| R78 | Adversarial | ops/quantized.rs + ops/fast/metal_kernel.rs（FFI 边界） |
| R79 | Adversarial | io.rs（62K行文件，文件系统交互） |
| R80 | Adversarial | error.rs（3,220行，全项目基础） |

**R81-R84: 数值精度深度审查**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R81 | Adversarial | RoPE 数值精度（所有变体：base/llama3/su/yarn） |
| R82 | Adversarial | 量化/反量化数值一致性 |
| R83 | Adversarial | DSP 数值精度（mel/Kaldi/STFT/ISTFT） |
| R84 | Adversarial | LoRA/DoRA 数学正确性 |

**R85-R88: 并发 + 安全深度审查**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R85 | Adversarial | Stream/Device 线程安全 |
| R86 | Adversarial | Audio playback 并发（cpal callback + try_lock） |
| R87 | Adversarial | WiredLimitGuard 引用计数正确性 |
| R88 | Adversarial | 所有 catch_unwind 路径验证 |

### Phase 11: 边界 + 回归 (R89-R100)

**R89-R92: 边界情况**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R89 | Adversarial | 空数组/零元素数组处理（全项目） |
| R90 | Adversarial | 整数溢出检查（全项目的 checked_mul/add/saturating） |
| R91 | Adversarial | 大文件/大数组的内存安全 |
| R92 | Adversarial | 错误恢复路径（每个 ? 操作符的错误是否都被正确处理） |

**R93-R96: API 设计审查**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R93 | Rust Quality | 公共 API 一致性（所有 pub fn/struct/enum） |
| R94 | Rust Quality | 文档完整性（所有 pub 项是否有 doc comment） |
| R95 | Rust Quality | 命名一致性（snake_case/PascalCase/SCREAMING_SNAKE_CASE） |
| R96 | Rust Quality | feature gate 设计和文档 |

**R97-R100: 最终汇总**
| Round | 专家 | 审查范围 |
|-------|------|---------|
| R97 | All | 合并所有发现 |
| R98 | All | 按严重度排序 |
| R99 | All | 生成优化建议清单 |
| R100 | All | 最终报告 + GitHub Issues |

---

## 执行方法

每个 Round：
1. 拉取最新代码 (`git pull origin main`)
2. 4 个专家团并行执行（delegate_task batch=3 + 1 单独）
3. 记录发现到内部状态
4. 进入下一个 Round

## 预计输出

- 100 份独立审查报告
- 1 份最终汇总报告
- GitHub Issues（如有新发现）
