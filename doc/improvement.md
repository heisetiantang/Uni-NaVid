# Uni-NaVid 模块级改进方向

> 从大模块视角出发，聚焦可替换、可升级的核心组件。
> 按 **难度 × 预期效果** 综合排序，性价比高的改进优先。
> 生成日期：2026-05-26

---

## 排序说明

```
难度：  低（数小时内完成）| 中（1-3天）| 高（1-2周）| 极高（需仿真环境/大量算力）
效果：  低（边际提升）   | 中（可见提升）| 高（显著提升）| 极高（质的飞跃）
```

综合优先级 = 效果 ÷ 难度，效果相同时难度低者优先。

---

## 目录

- [第一梯队：低难度 × 高效果（立刻可做）](#第一梯队低难度--高效果立刻可做)
- [第二梯队：低难度 × 中效果（顺手可做）](#第二梯队低难度--中效果顺手可做)
- [第三梯队：中难度 × 高效果（近期规划）](#第三梯队中难度--高效果近期规划)
- [第四梯队：中难度 × 中效果（按需实施）](#第四梯队中难度--中效果按需实施)
- [第五梯队：高难度 × 高效果（长期规划）](#第五梯队高难度--高效果长期规划)
- [第六梯队：极高难度 × 极高效果（研究方向）](#第六梯队极高难度--极高效果研究方向)
- [全局汇总表](#全局汇总表)
- [参考论文](#参考论文)

---

## 第一梯队：低难度 × 高效果（立刻可做）

> 改动小，收益大，应当优先实施。

---

### 1. 语言模型替换：Vicuna-7B → LLaMA-3.1-8B

| 项目 | 当前 | 替换后 |
|------|------|--------|
| 基座 | LLaMA-2（2023年中） | LLaMA-3.1（2024年） |
| 上下文 | 4K token | 128K token |
| MT-Bench | 6.57 | 8.22 |
| 架构兼容性 | — | 高度兼容，改动极小 |

**为什么难度低**：LLaMA-3.1 与 LLaMA-2 架构高度兼容，`llava_llama_vid.py` 几乎不需要修改，只需更换权重路径并升级 `transformers ≥ 4.40`。

**预期效果**：指令跟随、复杂推理、长上下文理解全面提升；128K 上下文可支持更长导航历史，无需额外压缩。

```
替换步骤：
  1. 下载 meta-llama/Meta-Llama-3.1-8B-Instruct
  2. 修改 scripts/uninavid_stage_1.sh 中 PREV_MODEL 路径
  3. pip install "transformers>=4.40"
  4. 用新基座重新运行 Stage 1 训练
```

---

### 2. 训练范式升级：SFT → SFT + DPO

| 项目 | 当前 | 升级后 |
|------|------|--------|
| 范式 | SFT（CE Loss） | SFT 预热 + DPO 对齐 |
| 监督信号 | 仅正确动作 | 成功轨迹 vs 失败轨迹对比 |
| 工具依赖 | HuggingFace Trainer | HuggingFace TRL（原生支持） |

**为什么难度低**：TRL 库原生提供 `DPOTrainer`，数据格式为 `{"prompt", "chosen", "rejected"}` 三元组，无需修改模型结构。

**预期效果**：模型从"知道正确答案"升级为"知道为什么这个更好"，导航成功率（SR）可提升 3-8%（参考 NaVRL 等工作）。

```
数据构造：
  同一起点出发，记录：
    chosen:  成功到达目标的动作序列
    rejected: 走错路/超时失败的动作序列
  可从已有 VLN-CE 仿真器日志中直接提取
```

---

## 第二梯队：低难度 × 中效果（顺手可做）

> 改动量小，各有一定收益，可与其他工作并行推进。

---

### 3. 推理量化：bf16 → AWQ 4-bit

| 项目 | 当前（bf16） | 量化后（4-bit） |
|------|------------|----------------|
| 显存需求 | ~16 GB | ~8 GB |
| 精度损失 | — | < 1% |
| 可运行 GPU | A100/A40 | RTX 3090 / 4080 |
| 推理速度 | 基准 | 快约 1.5× |

**为什么难度低**：仅需在 `builder.py` 的 `load_pretrained_model()` 中增加量化加载分支，2-3 行代码。

**预期效果**：部署门槛大幅降低，可在消费级 GPU 上运行；适合快速验证和演示。

```python
# 接入方式（在 builder.py 中）
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_pretrained(model_path, load_in_4bit=True)
```

---

### 4. 动作空间扩展：4 离散 → 细粒度离散

| 当前动作 | 扩展后动作 |
|----------|-----------|
| forward | forward_fast / forward_slow |
| left | left_15 / left_30 / left_45 / left_90 |
| right | right_15 / right_30 / right_45 / right_90 |
| stop | stop |

**为什么难度低**：LLM 不需要改动，仅需修改训练标注格式和 `offline_eval_uninavid.py` 中的动作解析与轨迹更新逻辑。

**预期效果**：转弯精度提升，尤其在需要小角度调整的狭窄走廊、门口场景中，路径更顺滑。

---

### 5. 轻量语言模型替换：Vicuna-7B → Phi-3.5-mini（3.8B）

| 项目 | Vicuna-7B | Phi-3.5-mini |
|------|-----------|--------------|
| 参数量 | 7B | 3.8B |
| 推理速度 | 基准 | 约快 2× |
| MT-Bench | 6.57 | 8.38 |
| 上下文 | 4K | 128K |

**为什么难度低**：Phi-3 系列基于 Llama 架构变体，继承关系改动小。

**预期效果**：在追求推理速度或资源受限场景下（边缘设备、实机部署），推理频率可从 5 Hz 提升至接近 10 Hz。

---

### 6. 训练策略升级：随机采样 → 课程学习

**当前问题**：`LazySupervisedDataset` 随机采样，简单和困难样本混合训练，初期收敛慢。

**改进方案**：按路径难度（步数、转弯次数、指令复杂度）从易到难排列训练样本。

**为什么难度低**：改动仅在 `LazySupervisedDataset` 的采样策略，无需修改模型结构。

**预期效果**：训练收敛速度加快约 20-30%，最终精度与随机采样持平或略优。

---

## 第三梯队：中难度 × 高效果（近期规划）

> 需要 1-3 天工程实现，但收益显著，值得投入。

---

### 7. 语言模型替换：Vicuna-7B → Qwen2.5-7B（中文场景）

| 项目 | Vicuna-7B | Qwen2.5-7B |
|------|-----------|------------|
| 中文理解 | 弱 | 极强 |
| MT-Bench | 6.57 | 8.40 |
| AlignBench（中文） | 落后 | 领先 |
| 架构 | LlamaForCausalLM | Qwen2ForCausalLM |

**为什么难度中**：需将 `llava_llama_vid.py` 的继承从 `LlamaForCausalLM` 改为 `Qwen2ForCausalLM`，并适配对话模板格式，约 1 天工作量。

**预期效果**：若导航指令含中文（如 ScanQA、中文 R2R），理解准确率大幅提升；同时受益于 Qwen2.5 更强的推理能力。

```
替换步骤：
  1. 下载 Qwen/Qwen2.5-7B-Instruct
  2. 修改 llava_llama_vid.py 继承关系
  3. 在 conversation.py 中增加 Qwen 对话模板
  4. 检查 tokenizer 特殊 token 兼容性
```

---

### 8. 视觉编码器替换：EVA ViT-g → SigLIP-SO400M

| 项目 | EVA ViT-g | SigLIP-SO400M |
|------|-----------|---------------|
| 参数量 | ~1B | ~400M |
| 输出维度 | 1408 | 1152 |
| 权重大小 | ~1.8 GB | ~0.8 GB |
| 图文对齐 | InfoNCE | Sigmoid Loss（更强） |
| 发布时间 | 2022 | 2023（ICCV） |

**为什么难度中**：需新增 `siglip_encoder.py`，修改 `multimodal_encoder/builder.py` 路由，调整 `mm_projector` 输入维度（1408→1152），并重新训练 projector（视觉编码器 frozen，约 1-2 小时）。

**预期效果**：图文对齐质量提升，视觉问答和指令理解更准确；模型更轻，推理更快。

```
视觉编码器维度变化：
  EVA ViT-g (1408) → SigLIP (1152)
  ↓
  mm_projector: 1408→4096  改为  1152→4096
  ↓
  config.mm_hidden_size = 1152
```

---

### 9. 动作输出扩展：文本解析 → 连续动作回归

**改进方案**：在 LLM 的 `lm_head` 之后增加轻量 MLP 回归头，直接输出连续值 `(v, ω)`（线速度、角速度），损失函数改为 Huber Loss。

**为什么难度中**：需修改 `llava_llama_vid.py` 的 forward 输出结构，以及 `train.py` 的 loss 计算逻辑，约 1-2 天。

**预期效果**：动作更平滑连续，避免离散动作的抖动；适合实机部署和 ROS 接口对接，导航路径更自然。

---

## 第四梯队：中难度 × 中效果（按需实施）

> 有一定工程量，但收益相对局限，可按研究需要选择实施。

---

### 10. Token 压缩替换：固定阈值 → ToMe（无需训练）

**当前问题**：`process_tensor` 使用固定余弦相似度阈值 0.985，对静态场景过度合并、对动态场景合并不足。

**改进方案**：引入 Token Merging（ToMe），在 EVA ViT 的 Transformer Block 内部用双向软匹配合并 token，**无需重新训练**。

**为什么难度中**：需修改 `eva_vit.py` 的 Transformer Block，在注意力计算前后插入 ToMe merge/unmerge 步骤，约 1 天。

**预期效果**：吞吐量提升约 2×，精度下降 < 0.5%；同时消除固定阈值带来的场景不适应问题。

**参考**：[Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461)，ICLR 2023。

---

### 11. 推理框架替换：HF generate → vLLM

**改进方案**：将 `offline_eval_uninavid.py` 中的 `model.generate()` 替换为 vLLM 的 `LLM.generate()` 接口，利用 PagedAttention 提升批量评测吞吐。

**为什么难度中**：vLLM 对自定义多模态模型的支持需要额外适配（注册视觉 encoder 处理逻辑），约 1-2 天。

**预期效果**：批量评测时吞吐提升 3-5×，适合大规模 benchmark 跑分场景；单步推理延迟改善有限。

---

### 12. Token 压缩替换：固定阈值 → 可学习时序卷积

**改进方案**：参考 VideoLLaMA2，用可学习的时序卷积（Temporal Conv）替代固定余弦相似度合并，端到端训练帧重要性。

**为什么难度中**：需新增时序卷积模块并集成进 `vlm_attention`，以及对应的训练 loss，约 2-3 天。

**预期效果**：模型自动学习哪些帧重要，动态场景（快速移动、目标切换）理解更准确。

**参考**：[VideoLLaMA 2](https://arxiv.org/abs/2406.07476)，arXiv 2024。

---

### 13. 视觉编码器补充：DINOv2-ViT-L（空间定位优先）

**改进方案**：若导航的瓶颈是空间定位（而非图文对齐），可引入 DINOv2-ViT-L 的空间特征，与 SigLIP 做双编码器融合。

**为什么难度中**：需新增编码器并设计双编码器融合策略（concat / cross-attn），约 2 天。

**预期效果**：深度感知、物体位置判断、场景结构理解提升，对 ObjectNav 任务尤为有效。

**参考**：[DINOv2](https://arxiv.org/abs/2304.07193)，TMLR 2024。

---

## 第五梯队：高难度 × 高效果（长期规划）

> 需要较长工程周期，但对模型能力有结构性提升，适合作为长期研究方向。

---

### 14. 多模态投影头升级：MLP → Resampler

**改进方案**：引入 Flamingo 式 Resampler——固定数量的可学习 latent（如 64 个），通过交叉注意力"读取"任意数量的视觉特征，输出固定 token 数。

**为什么难度高**：Resampler 模块本身需要设计与实现，且需要与现有 `token_generation` / `process_tensor` 机制整合或替代，约 1-2 周。

**预期效果**：彻底解决长视频 token 爆炸问题，无论导航多少步，输入 LLM 的视觉 token 数量始终固定；大幅降低显存压力。

**参考**：[Flamingo](https://arxiv.org/abs/2204.14198)，NeurIPS 2022；[Otter](https://arxiv.org/abs/2305.03726)。

---

### 15. 视觉编码器升级：EVA ViT-g → InternViT-6B

| 项目 | EVA ViT-g | InternViT-6B |
|------|-----------|--------------|
| 参数量 | ~1B | ~6B |
| OCR 与细粒度理解 | 一般 | 极强 |
| 空间关系推理 | 一般 | 强 |
| 显存占用 | 中 | 高 |

**为什么难度高**：InternViT-6B 输出维度 3200，projector 结构需重新设计；显存需求大，训练配置需要大幅调整，约 1-2 周。

**预期效果**：视觉理解能力质的飞跃，尤其对复杂室内场景（物体遮挡、小物体识别、文字标识）效果显著。

**参考**：[InternVL](https://arxiv.org/abs/2312.14238)，CVPR 2024。

---

### 16. Token 压缩替换：相似度合并 → Mamba 状态空间模型

**改进方案**：用 Mamba 的线性时间状态空间模型替代当前的滑动窗口 + 相似度合并机制，天然支持超长序列（数百步导航历史）。

**为什么难度高**：需要将 Mamba 集成为视觉历史的序列压缩器，与 LLM 的接口对齐，架构改动较大，约 2 周。

**预期效果**：注意力复杂度从 O(T²) 降为 O(T)，可处理极长导航轨迹（> 200 步）而不损失早期历史信息。

**参考**：[Mamba](https://arxiv.org/abs/2312.00752)，ICLR 2024。

---

### 17. 推理部署：TensorRT-LLM 引擎化

**改进方案**：将训练好的模型导出为 TensorRT-LLM 引擎，在 NVIDIA GPU 上实现极致推理性能。

**为什么难度高**：自定义多模态模型的 TRT-LLM 适配工作量大，需要手动编写 plugin，约 2 周。

**预期效果**：推理速度提升 5-10×，在 A100 上可实现 20+ Hz 在线推理（当前约 5 Hz）。

---

## 第六梯队：极高难度 × 极高效果（研究方向）

> 需要仿真环境、大量算力支持，适合作为后续发表方向。

---

### 18. 训练范式：SFT → PPO / GRPO 强化学习

**改进方案**：以导航成功率（SR）和路径效率（SPL）为奖励，用 PPO 或 GRPO 在 Habitat-Sim 仿真环境中在线强化学习。

**为什么难度极高**：
- 需要搭建 Habitat-Sim 在线仿真环境
- 需要设计奖励函数（稀疏 SR + 稠密距离奖励）
- 强化学习训练不稳定，需要大量 GPU 资源（通常 8-16 卡）

**预期效果**：导航成功率可突破 SFT 的天花板，在 VLN-CE R2R 等 benchmark 上有望进入 SOTA 行列；这是当前 LLM-based 导航最前沿的研究方向。

**参考**：EmbodiedGPT、NavRL 等工作。

---

## 全局汇总表

| 排序 | 模块 | 当前 → 替换方案 | 难度 | 预期效果 | 性价比 |
|------|------|----------------|------|----------|--------|
| 1 | 语言模型 | Vicuna-7B → **LLaMA-3.1-8B** | 低 | 高 | ★★★★★ |
| 2 | 训练范式 | SFT → **SFT + DPO** | 低 | 高 | ★★★★★ |
| 3 | 推理部署 | bf16 → **AWQ 4-bit 量化** | 低 | 中 | ★★★★ |
| 4 | 动作空间 | 4 离散 → **细粒度离散动作** | 低 | 中 | ★★★★ |
| 5 | 语言模型 | Vicuna-7B → **Phi-3.5-mini** | 低 | 中 | ★★★★ |
| 6 | 训练策略 | 随机采样 → **课程学习** | 低 | 中 | ★★★★ |
| 7 | 语言模型 | Vicuna-7B → **Qwen2.5-7B** | 中 | 高 | ★★★★ |
| 8 | 视觉编码器 | EVA ViT-g → **SigLIP-SO400M** | 中 | 高 | ★★★★ |
| 9 | 动作输出 | 文本解析 → **连续动作回归** | 中 | 高 | ★★★★ |
| 10 | Token 压缩 | 固定阈值 → **ToMe**（免训练） | 中 | 中 | ★★★ |
| 11 | 推理部署 | HF generate → **vLLM** | 中 | 中 | ★★★ |
| 12 | Token 压缩 | 固定阈值 → **可学习时序卷积** | 中 | 中 | ★★★ |
| 13 | 视觉编码器 | EVA ViT-g → **DINOv2-ViT-L** | 中 | 中 | ★★★ |
| 14 | 多模态投影 | MLP 2层 → **Resampler** | 高 | 高 | ★★★ |
| 15 | 视觉编码器 | EVA ViT-g → **InternViT-6B** | 高 | 极高 | ★★★ |
| 16 | Token 压缩 | 相似度合并 → **Mamba** | 高 | 高 | ★★ |
| 17 | 推理部署 | HF generate → **TensorRT-LLM** | 高 | 中 | ★★ |
| 18 | 训练范式 | SFT → **PPO / GRPO** | 极高 | 极高 | ★★ |

---

## 参考论文

| 方案 | 论文 | 会议/期刊 |
|------|------|-----------|
| LLaMA-3.1 | [Meta Llama 3](https://arxiv.org/abs/2407.21783) | arXiv 2024 |
| Qwen2.5 | [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) | arXiv 2024 |
| Phi-3.5 | [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) | arXiv 2024 |
| SigLIP | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | ICCV 2023 |
| InternViT | [InternVL: Scaling up Vision Foundation Models](https://arxiv.org/abs/2312.14238) | CVPR 2024 |
| DINOv2 | [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193) | TMLR 2024 |
| Token Merging | [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461) | ICLR 2023 |
| VideoLLaMA2 | [VideoLLaMA 2: Advancing Temporal and Spatial Reasoning](https://arxiv.org/abs/2406.07476) | arXiv 2024 |
| Flamingo / Resampler | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | NeurIPS 2022 |
| DPO | [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | NeurIPS 2023 |
| AWQ | [AWQ: Activation-aware Weight Quantization for LLM Compression](https://arxiv.org/abs/2306.00978) | MLSys 2024 |
| vLLM | [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | SOSP 2023 |
| Mamba | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | ICLR 2024 |

---

*文档生成于 2026-05-26。*
