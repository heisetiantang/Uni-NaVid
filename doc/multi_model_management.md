# 多模型管理与快速切换方案

> 核心思路：每个改进**只训练一次**，checkpoint 永久保留在 `model_zoo/` 下，  
> 评测时通过参数切换模型，无需重训。  
> 当前执行路线详见 `action_plan.md`。

---

## 目录

- [最重要：代码版本与 checkpoint 的绑定关系](#最重要代码版本与-checkpoint-的绑定关系)
- [各阶段代码改动与兼容性分析](#各阶段代码改动与兼容性分析)
- [当前计划的模型列表](#当前计划的模型列表)
- [目录结构约定](#目录结构约定)
- [训练后保存 checkpoint](#训练后保存-checkpoint)
- [单模型评测（参数化调用）](#单模型评测参数化调用)
- [批量评测](#批量评测)
- [结果对比](#结果对比)
- [完整工作流示例](#完整工作流示例)
- [磁盘空间参考](#磁盘空间参考)

---

## 最重要：代码版本与 checkpoint 的绑定关系

**不同阶段的改进会修改模型架构或配置，导致旧代码无法加载新 checkpoint，反之亦然。**  
必须用对应的代码版本加载对应的 checkpoint，否则报错或静默加载错误权重。

| checkpoint | 对应代码分支 | 不兼容的原因 | 能否用 baseline 代码加载？ |
|---|---|---|---|
| `uninavid_baseline` | `main` | — | ✅ 是 |
| `uninavid_s1_qwen25_curriculum` | `feat/qwen25` | LLM 从 Vicuna-7B（LlamaForCausalLM）换为 Qwen2.5-7B（Qwen2ForCausalLM），tokenizer、词表（152064）、模型类均不同 | ❌ 否 |
| `uninavid_s2_dpo` | `feat/qwen25` | 与 S1 架构完全一致，只是权重不同 | ❌ 否（同 S1） |
| `uninavid_s3_siglip` | `feat/siglip` | 视觉编码器变更，`mm_hidden_size` 从 1408 变为 1152，projector 结构不同 | ❌ 否 |

**管理方式：用 Git 分支隔离代码版本**

```
main                    ← baseline 代码，加载 uninavid_baseline
  └─ feat/qwen25        ← S1/S2 代码，加载 s1 / s2 checkpoint
       └─ feat/siglip   ← S3 代码，在 feat/qwen25 基础上修改，加载 s3 checkpoint
```

评测时切换分支即切换对应代码，再指定 `MODEL_PATH` 即可加载对应 checkpoint：

```bash
# 评测 baseline
git checkout main
bash eval.sh --model uninavid_baseline --benchmark r2r

# 评测第一/二阶段
git checkout feat/qwen25
bash eval.sh --model uninavid_s1_qwen25_curriculum --benchmark r2r
bash eval.sh --model uninavid_s2_dpo --benchmark r2r

# 评测第三阶段
git checkout feat/siglip
bash eval.sh --model uninavid_s3_siglip --benchmark r2r
```

> ToMe（第零阶段）通过 flag 控制，`--use-tome` 在任意分支上均可选择性启用，不影响 checkpoint 兼容性。

---

## 各阶段代码改动与兼容性分析

### 第零阶段：改进 10（ToMe）

**修改文件**：`uninavid/model/multimodal_encoder/eva_vit.py`

**改动内容**：在 ViT Transformer Block 的注意力计算前后插入 ToMe merge / unmerge 步骤。

**兼容性**：
- ✅ 不改变 checkpoint 格式，权重完全兼容
- ✅ 通过 `use_tome: bool` 参数控制是否启用，可随时开关
- ✅ 所有分支均可使用，不影响其他阶段

**注意**：ToMe 插入后模型的中间输出 shape 会变化，但最终输出 shape 不变（unmerge 还原），checkpoint 加载无影响。

---

### 第一阶段：改进 7 + 6（Qwen2.5-7B + 课程学习）

#### 改进 7 代码改动

**修改文件**：
- `uninavid/model/language_model/llava_llama_vid.py`：将继承关系从 `LlamaForCausalLM` 改为 `Qwen2ForCausalLM`，适配 Qwen2 的 forward 签名
- `uninavid/model/conversation.py`：新增 Qwen2.5 对话模板（ChatML 格式）
- `uninavid/model/builder.py`：检查 tokenizer 特殊 token（`<|im_start|>` / `<|im_end|>`）兼容性
- `scripts/uninavid_stage_1.sh`：`PREV_MODEL` 路径改为 Qwen2.5-7B-Instruct 权重路径
- `requirements`：`transformers>=4.37`（Qwen2 支持）

**checkpoint 配置变化**：
```
config.json 中：
  "_name_or_path": "Qwen/Qwen2.5-7B-Instruct"       ← 变化
  "architectures": ["Qwen2ForCausalLM"]              ← 变化（原 LlamaForCausalLM）
  "vocab_size": 152064                               ← 变化（原 Vicuna 32000）
  "bos_token_id": 151643                             ← 变化
  "eos_token_id": 151645                             ← 变化

tokenizer_config.json：
  使用 Qwen2 tokenizer（tiktoken BPE）               ← 完全不同，不可混用
  chat_template: ChatML 格式（<|im_start|>...<|im_end|>）
```

**不兼容原因**：模型类（`Qwen2ForCausalLM` vs `LlamaForCausalLM`）、词表大小（32000 → 152064）、tokenizer 格式均不同，用 baseline 代码加载会导致类型不匹配或 embedding 维度不匹配。

**分支操作**：
```bash
git checkout -b feat/qwen25
# 修改上述文件后提交
git add uninavid/model/language_model/llava_llama_vid.py
git add uninavid/model/conversation.py
git add uninavid/model/builder.py
git add scripts/uninavid_stage_1.sh
git commit -m "feat: replace Vicuna-7B base with Qwen2.5-7B"
```

#### 改进 6 代码改动

**修改文件**：`uninavid/train/train.py` 中的 `LazySupervisedDataset`

**改动内容**：在数据集初始化时，按路径难度字段（步数、转弯次数）对样本排序，训练前期用简单样本，后期逐步引入复杂样本。

**兼容性**：
- ✅ 纯训练逻辑改动，不影响模型结构，checkpoint 格式与改进 7 完全相同
- ✅ 训练完成后 checkpoint 可与 S1 的代码/评测脚本直接互换

```bash
# 在 feat/qwen25 分支上追加提交
git add uninavid/train/train.py
git commit -m "feat: add curriculum learning to LazySupervisedDataset"
```

---

### 第二阶段：改进 2（DPO）

**修改文件**：新增 `uninavid/train/train_dpo.py`（DPO 训练入口）

**改动内容**：
- 引入 `trl.DPOTrainer`，接受 `{"prompt", "chosen", "rejected"}` 三元组数据
- 在 S1 checkpoint 基础上继续训练，不修改模型结构

**兼容性**：
- ✅ 模型架构与 S1 完全一致，`config.json` 无变化
- ✅ S2 checkpoint 与 S1 代码完全兼容，可用同一套评测代码加载
- ✅ 在 `feat/qwen25` 分支上追加文件即可，无需新分支

**checkpoint 关系**：
```
uninavid_s1_qwen25_curriculum  ──DPO fine-tune──→  uninavid_s2_dpo
       （起点）                                          （结果）
       同一架构，只有权重数值不同
```

```bash
# 仍在 feat/qwen25 分支
git add uninavid/train/train_dpo.py
git commit -m "feat: add DPO training script"
```

---

### 第三阶段：改进 8（SigLIP 编码器）

**修改文件**：
- 新增 `uninavid/model/multimodal_encoder/siglip_encoder.py`
- `uninavid/model/multimodal_encoder/builder.py`：添加 SigLIP 路由分支
- `uninavid/model/uninavid_arch.py`：`mm_hidden_size` 读取逻辑兼容 1152

**checkpoint 配置变化**：
```
config.json 中：
  "mm_vision_tower": "google/siglip-so400m-patch14-384"  ← 变化
  "mm_hidden_size": 1152                                 ← 变化（原 1408）

projector 权重维度：
  原：Linear(1408, 4096)
  新：Linear(1152, 4096)  ← 形状不同，不可共用
```

**不兼容原因**：projector 的 Linear 层 shape 不同，用旧代码加载会直接报 shape mismatch 错误。

**分支操作**：
```bash
git checkout feat/qwen25
git checkout -b feat/siglip
# 修改上述文件后提交
git add uninavid/model/multimodal_encoder/siglip_encoder.py
git add uninavid/model/multimodal_encoder/builder.py
git add uninavid/model/uninavid_arch.py
git commit -m "feat: replace EVA ViT-g with SigLIP-SO400M encoder"
```

**训练方式**（只需 Stage 1）：

```bash
# Stage 1：视觉编码器 frozen，只训练 projector（约 1-2 小时）
bash scripts/uninavid_stage_1_siglip.sh
# 起点：使用 S2（DPO）的 LLM 权重 + 随机初始化的新 projector
# 视觉编码器：SigLIP-SO400M（frozen）

# Stage 2：无需重跑，直接用 Stage 1 产出的 checkpoint 做评测
```

---

## 当前计划的模型列表

| 阶段 | checkpoint 名称 | 对应改进 | 代码分支 | 训练方式 |
|---|---|---|---|---|
| baseline | `uninavid_baseline` | 原始 Vicuna-7B | `main` | 已有 |
| 第零阶段 | 无新 checkpoint | 改进 10（ToMe） | 任意分支 + `--use-tome` | 无需训练 |
| 第一阶段 | `uninavid_s1_qwen25_curriculum` | 改进 7 + 6（合并） | `feat/qwen25` | Stage 1 + Stage 2 |
| 第二阶段 | `uninavid_s2_dpo` | 改进 2（DPO） | `feat/qwen25` | DPO fine-tune（小时级） |
| 第三阶段 | `uninavid_s3_siglip` | 改进 8（SigLIP） | `feat/siglip` | 仅 Stage 1（1-2 小时） |
| 第三阶段（可选） | `uninavid_s3_siglip_dino` | 改进 8 + 13（DINOv2） | `feat/siglip` | Stage 1 重训 |
| 长期规划 | `uninavid_lt_grpo` | 改进 18（PPO/GRPO） | `feat/grpo` | 在线 RL，独立流程 |

---

## 目录结构约定

```
~/Uni-NaVid/                               ← 训练仓库
  output/
    uninavid_s1_qwen25_curriculum/         ← 训练产出（原始位置）
    uninavid_s2_dpo/
    uninavid_s3_siglip/

~/NaVid-VLN-CE/                            ← 评测仓库
  model_zoo/                               ← 所有 checkpoint 统一入口（软链接）
    uninavid_baseline/                     ← 原始模型（已有）
    uninavid_s1_qwen25_curriculum/         ← → 软链接到 Uni-NaVid/output/
    uninavid_s2_dpo/                       ← → 软链接
    uninavid_s3_siglip/                    ← → 软链接
    uninavid_s3_siglip_dino/               ← → 软链接

  tmp/                                     ← 评测结果
    results_baseline_r2r/
    results_baseline_tome_r2r/             ← ToMe 对比
    results_s1_qwen25_curriculum_r2r/
    results_s2_dpo_r2r/
    results_s3_siglip_r2r/
```

---

## 训练后保存 checkpoint

### 软链接到 model_zoo（推荐）

```bash
cd ~/NaVid-VLN-CE

# 第一阶段完成后
ln -s ~/Uni-NaVid/output/uninavid_s1_qwen25_curriculum \
      model_zoo/uninavid_s1_qwen25_curriculum

# 第二阶段完成后
ln -s ~/Uni-NaVid/output/uninavid_s2_dpo \
      model_zoo/uninavid_s2_dpo

# 第三阶段完成后
ln -s ~/Uni-NaVid/output/uninavid_s3_siglip \
      model_zoo/uninavid_s3_siglip
```

### 第零阶段（ToMe）不产生新 checkpoint

```bash
# --use-tome 是运行时 flag，不改变权重，结果存到独立目录便于对比
bash eval.sh --model uninavid_baseline --benchmark r2r              # 不启用
bash eval.sh --model uninavid_baseline --benchmark r2r --use-tome  # 启用 ToMe
```

---

## 单模型评测（参数化调用）

**必须先切换到对应代码分支，再执行评测：**

```bash
# ── baseline ─────────────────────────────────────
git -C ~/Uni-NaVid checkout main
bash eval.sh --model uninavid_baseline --benchmark r2r

# ── 第零阶段：ToMe（任意分支均可）────────────────
bash eval.sh --model uninavid_baseline --benchmark r2r --use-tome

# ── 第一/二阶段（feat/qwen25 分支）──────────────
git -C ~/Uni-NaVid checkout feat/qwen25
bash eval.sh --model uninavid_s1_qwen25_curriculum --benchmark r2r
bash eval.sh --model uninavid_s2_dpo               --benchmark r2r

# ── 第三阶段（feat/siglip 分支）──────────────────
git -C ~/Uni-NaVid checkout feat/siglip
bash eval.sh --model uninavid_s3_siglip --benchmark r2r
```

> `eval.sh` 待后续创建，内部自动推导 `MODEL_PATH=model_zoo/$MODEL`、`SAVE_PATH=tmp/results_${MODEL}_${BENCHMARK}`。

---

## 批量评测

由于不同 checkpoint 依赖不同代码分支，批量评测需按分支分组执行：

```bash
# 第一步：main 分支的评测
git -C ~/Uni-NaVid checkout main
bash eval.sh --model uninavid_baseline --benchmark r2r
bash eval.sh --model uninavid_baseline --benchmark r2r --use-tome

# 第二步：feat/qwen25 分支的评测
git -C ~/Uni-NaVid checkout feat/qwen25
bash eval.sh --model uninavid_s1_qwen25_curriculum --benchmark r2r
bash eval.sh --model uninavid_s2_dpo               --benchmark r2r

# 第三步：feat/siglip 分支的评测
git -C ~/Uni-NaVid checkout feat/siglip
bash eval.sh --model uninavid_s3_siglip --benchmark r2r

# 最后：汇总对比
python compare_results.py --tmp-dir tmp --sort sr
```

> `batch_eval.sh` 可封装上述逻辑，待后续创建。

---

## 结果对比

所有阶段评测完成后，结果均在 `tmp/` 下，随时可对比：

```bash
# 按阶段顺序对比
python compare_results.py \
  --paths tmp/results_baseline_r2r \
          tmp/results_baseline_tome_r2r \
          tmp/results_s1_qwen25_curriculum_r2r \
          tmp/results_s2_dpo_r2r \
          tmp/results_s3_siglip_r2r \
  --sort sr
```

> `compare_results.py` 待后续创建，详见 `eval_pipeline.md` 第四步。

---

## 完整工作流示例

```bash
# ════════════════════════════════════════════════════════
# 第零阶段：ToMe 验证
# ════════════════════════════════════════════════════════
cd ~/Uni-NaVid
git checkout main           # ToMe 改动提交到 main

# 修改 eva_vit.py 插入 ToMe，提交
git add uninavid/model/multimodal_encoder/eva_vit.py
git commit -m "feat: add ToMe token merging to EVA ViT"

cd ~/NaVid-VLN-CE
bash eval.sh --model uninavid_baseline --benchmark r2r
bash eval.sh --model uninavid_baseline --benchmark r2r --use-tome
python compare_results.py \
  --paths tmp/results_baseline_r2r tmp/results_baseline_tome_r2r


# ════════════════════════════════════════════════════════
# 第一阶段：Qwen2.5-7B + 课程学习
# ════════════════════════════════════════════════════════
cd ~/Uni-NaVid
git checkout -b feat/qwen25
# 修改代码（见各阶段代码改动章节），提交后训练
bash scripts/uninavid_stage_1.sh
bash scripts/uninavid_stage_2.sh

cd ~/NaVid-VLN-CE
ln -s ~/Uni-NaVid/output/uninavid_s1_qwen25_curriculum \
      model_zoo/uninavid_s1_qwen25_curriculum

git -C ~/Uni-NaVid checkout feat/qwen25
bash eval.sh --model uninavid_s1_qwen25_curriculum --benchmark r2r
python compare_results.py --tmp-dir tmp --sort sr


# ════════════════════════════════════════════════════════
# 第二阶段：DPO（在 feat/qwen25 分支继续）
# ════════════════════════════════════════════════════════
cd ~/Uni-NaVid
# feat/qwen25 分支上新增 train_dpo.py，准备三元组数据后训练
bash scripts/uninavid_dpo.sh

cd ~/NaVid-VLN-CE
ln -s ~/Uni-NaVid/output/uninavid_s2_dpo model_zoo/uninavid_s2_dpo

git -C ~/Uni-NaVid checkout feat/qwen25
bash eval.sh --model uninavid_s2_dpo --benchmark r2r
python compare_results.py --tmp-dir tmp --sort sr


# ════════════════════════════════════════════════════════
# 第三阶段：SigLIP（新建 feat/siglip 分支）
# ════════════════════════════════════════════════════════
cd ~/Uni-NaVid
git checkout feat/qwen25
git checkout -b feat/siglip
# 修改代码（siglip_encoder.py / builder.py / uninavid_arch.py），提交后训练
bash scripts/uninavid_stage_1_siglip.sh   # 仅 Stage 1，约 1-2 小时

cd ~/NaVid-VLN-CE
ln -s ~/Uni-NaVid/output/uninavid_s3_siglip model_zoo/uninavid_s3_siglip

git -C ~/Uni-NaVid checkout feat/siglip
bash eval.sh --model uninavid_s3_siglip --benchmark r2r
python compare_results.py --tmp-dir tmp --sort sr
```

---

## 磁盘空间参考

| checkpoint | 大小 | 代码分支 | 备注 |
|---|---|---|---|
| `uninavid_baseline`（Vicuna-7B，bf16） | ~14 GB | `main` | 已有 |
| `uninavid_s1_qwen25_curriculum`（Qwen2.5-7B） | ~14 GB | `feat/qwen25` | 第一阶段 |
| `uninavid_s2_dpo` | ~14 GB | `feat/qwen25` | 第二阶段，与 S1 架构相同 |
| `uninavid_s3_siglip` | ~12 GB | `feat/siglip` | 第三阶段，视觉编码器更小 |
| 软链接 | 0 GB | — | 推荐，节省磁盘 |

> 当前计划实际存储约 **54 GB**（全部软链接，原始文件在 `~/Uni-NaVid/output/` 下）。  
> 若磁盘紧张，S1 checkpoint 在 S2 验收后可考虑删除（S2 是 S1 的超集）。
