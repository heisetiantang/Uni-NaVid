# Uni-NaVid 代码库文档

> **论文**：*A Video-based Vision-Language-Action Model for Unifying Embodied Navigation Tasks*
> **发表**：RSS 2025 | **机构**：北京大学 EPIC 实验室

---

## 目录

1. [项目简介](#1-项目简介)
2. [整体架构](#2-整体架构)
3. [目录结构](#3-目录结构)
4. [核心模块](#4-核心模块)
  - 4.1 [模型定义](#41-模型定义-uninavidmodel)
  - 4.2 [训练流程](#42-训练流程-uninavidtrain)
  - 4.3 [工具函数](#43-工具函数)
  - 4.4 [离线评测](#44-离线评测)
5. [关键技术：视频 Token 压缩](#5-关键技术视频-token-压缩)
6. [特殊 Token 体系](#6-特殊-token-体系)
7. [数据格式](#7-数据格式)
8. [训练流程详解](#8-训练流程详解)
9. [推理流程详解](#9-推理流程详解)
10. [依赖环境](#10-依赖环境)
11. [快速定位指南](#11-快速定位指南)

---

## 1. 项目简介

**Uni-NaVid** 是一个基于视频的视觉-语言-动作（VLA）模型，旨在统一多类具身导航任务。

### 核心思想

以**历史视频帧序列 + 当前观测帧**作为视觉输入，通过大语言模型（Vicuna-7B）直接输出离散导航动作，在统一框架内处理以下任务：


| 任务类型           | 说明             |
| -------------- | -------------- |
| VLN（视觉语言导航）    | 按自然语言指令从 A 到 B |
| 目标跟踪（Tracking） | 跟随特定目标移动       |
| ObjectNav      | 导航至指定类别物体      |
| EQA（具身问答）      | 边导航边回答问题       |


### 输出动作空间

```
forward  |  left  |  right  |  stop
```

### 技术谱系

- **视觉编码**：继承自 [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID)
- **导航框架**：继承自 [NaVid](https://github.com/jzhzhang/NaVid-VLN-CE)
- **语言模型**：[Vicuna-7B-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)

---

## 2. 整体架构

```
┌──────────────────────────────────────────────────┐
│                    输入                           │
│   视频帧序列 [历史]  +  当前帧  +  语言指令       │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│              视觉编码（EVA ViT-g）                │
│         eva_vit_g.pth  ·  224×224 输入           │
│         输出：每帧 1408 维 patch 特征             │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│           视频 Token 压缩（核心创新）             │
│  · Grid 池化：历史帧 → 4 token/帧（2×2 pool）    │
│  · 当前帧：8×8 grid → 64 token                  │
│  · 相似帧合并：余弦相似度 > 阈值 → 1 token       │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│           多模态投影（mm_projector）              │
│         MLP 2层 + GELU  ·  1408 → 4096           │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│             Vicuna-7B（Llama 骨干）               │
│   inputs_embeds = [视觉 embedding + 文本 token]  │
└────────────────┬─────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│                    输出                           │
│        forward / left / right / stop             │
└──────────────────────────────────────────────────┘
```

---

## 3. 目录结构

```
Uni-NaVid/
├── README.md                        # 安装、训练、评测说明
├── LICENSE                          # Apache 2.0
├── pyproject.toml                   # 依赖与 pip 可编辑安装
├── offline_eval_uninavid.py         # 离线视频评测入口
├── assets/
│   └── uninavid.png                 # 方法示意图
├── doc/
│   └── uni-navid.pdf                # 论文原文
├── scripts/
│   ├── uninavid_stage_1.sh          # Stage 1 训练（从 Vicuna 微调）
│   ├── uninavid_stage_2.sh          # Stage 2 训练（继续微调）
│   ├── zero2.json                   # DeepSpeed ZeRO-2 配置
│   └── zero2_offload.json           # ZeRO-2 + CPU offload 配置
└── uninavid/                        # 主 Python 包（17 个 .py 文件）
    ├── __init__.py                  # 导出 LlavaLlamaAttForCausalLM
    ├── constants.py                 # 特殊 token 与常量定义
    ├── conversation.py              # 对话模板（imgsp_v1）
    ├── mm_utils.py                  # 图像处理与推理工具函数
    ├── model/
    │   ├── __init__.py
    │   ├── builder.py               # 模型加载入口
    │   ├── uninavid_arch.py         # ★ 核心架构（~700 行）
    │   ├── language_model/
    │   │   └── llava_llama_vid.py   # LLM 封装（继承 Llama + UniNaVID）
    │   ├── multimodal_encoder/
    │   │   ├── builder.py           # 视觉编码器选择
    │   │   ├── eva_vit.py           # EVA ViT-g 实现
    │   │   └── clip_encoder.py      # 标准 CLIP ViT（备用）
    │   └── multimodal_projector/
    │       └── builder.py           # 投影头构建（linear/mlp/identity）
    ├── train/
    │   ├── train.py                 # ★ 训练主程序（~1400 行）
    │   ├── train_mem.py             # 训练入口（patch FlashAttention）
    │   ├── llava_trainer.py         # 自定义 Trainer（长度分组采样）
    │   └── llama_flash_attn_monkey_patch.py  # FlashAttention 替换
    └── processor/
        ├── clip-patch14-224/        # 224 分辨率 CLIP 预处理配置
        └── clip-patch14-336/        # 336 分辨率 CLIP 预处理配置

# 运行时目录（被 .gitignore，需手动准备）
├── model_zoo/
│   ├── eva_vit_g.pth                # EVA ViT-g 权重
│   ├── vicuna-7b-v1.5/              # 基础语言模型
│   └── uninavid-7b-full-224-video-fps-1-grid-2/  # 发布权重
├── data/Nav-Finetune/
│   ├── nav_videos/                  # 导航视频文件
│   └── open_uninavid_sampled_500.json  # 训练标注（子集示例）
└── test_cases/
    ├── vln_1/                       # VLN 测试用例
    └── tracking_1/                  # 跟踪测试用例
```

---

## 4. 核心模块

### 4.1 模型定义 (`uninavid/model/`)

#### `uninavid_arch.py` — 架构中枢（最重要文件）

定义两个核心 Mixin 类：

`**UniNaVIDMetaModel**`


| 方法                                             | 功能                                 |
| ---------------------------------------------- | ---------------------------------- |
| `initialize_vision_modules()`                  | 构建 vision_tower（EVA）和 mm_projector |
| `initialize_online_inference_nav_feat_cache()` | 初始化推理时的帧特征缓存                       |


`**UniNaVIDMetaForCausalLM**`（核心算法）


| 方法                                                                                                                     | 输入参数                                                                                                                                                       | 输出                                                                                                                                                | 功能描述                                                                                                                                                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `token_generation(self, vis_embed, image_counts=None, navigation=False)`                                               | vis_embed: torch.Tensor，视觉特征，shape=[N, L, C]；image_counts: int 或 None，对应帧数；navigation: bool，是否为导航任务                                                        | vis_embed: (torch.Tensor) 池化压缩后的视觉特征；vis_embed_nav: (torch.Tensor or None) 导航最后一帧特征                                                               | 参考`uninavid/model/uninavid_arch.py`，对视觉特征做 grid 池化（8x8/2x2），压缩为 token，若为导航，历史帧和当前帧分别单独池化处理，输出压缩后特征并传递给 projector                                                                                                                         |
| `process_tensor(self, final_token, nav_size, length_threshold=64, similarity_threshold=0.985)`                         | final_token: torch.Tensor，shape=[1, N, 64, D]，所有导航帧的 token；nav_size: int，导航总帧数；length_threshold: int，最大缓存帧数，默认64；similarity_threshold: float，相似度阈值，默认0.985 | final_token: (torch.Tensor) 合并后的特征，shape=[1, X, 64, D]；lengths_list: list，每段token数量                                                               | 参考`uninavid/model/uninavid_arch.py`，训练时将最近length_threshold帧保留，剩余早期帧合并为1 token，利用余弦相似度聚合，返回合并结果及每段token数量                                                                                                                                 |
| `online_process_tensor(self, nav_size, length_threshold=64, similarity_threshold=0.985)`                               | nav_size: int，导航总帧数；length_threshold: int，最大缓存帧数，默认64；similarity_threshold: float，相似度阈值，默认0.985                                                            | final_token: (torch.Tensor) 合并后特征；lengths_list: list，各段token数量                                                                                    | 参考`uninavid/model/uninavid_arch.py`，推理时根据当前帧与缓存（feat_cache、long_feat_cache）做增量聚合，历史帧按相似度合并，动态维护缓存和分割，输出合并后特征及token计数                                                                                                                     |
| `vlm_attention(self, prompt, images, image_counts=None, navigation=False)`                                             | prompt: list[str]，输入文本；images: torch.Tensor，视觉输入，[B, ...]；image_counts: list[int]或None，每样本对应帧数；navigation: bool，是否导航                                       | img_feat_lst: list[torch.Tensor]，每样本token；video_or_not: list[bool]，是否视频任务；nav_or_not: list，若导航则为特征，否则None；final_token_length_lst: list，各样本token长度 | 参考`uninavid/model/uninavid_arch.py`，按prompt内容判断是navigation/video/image，分别处理，如为video或navigation则合并帧特征，否则单帧特征，最终返回token及标志位                                                                                                                |
| `prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images, prompts=None)` | input_ids: torch.Tensor；attention_mask: torch.Tensor；past_key_values: 生成用缓存；labels: torch.Tensor；images: torch.Tensor，视觉输入；prompts: list[str]，文本           | 同步调整后的 inputs, labels                                                                                                                             | 参考`uninavid/model/uninavid_arch.py`，将图片特征按input_ids中的占位符插入，构造多模态输入和训练标签                                                                                                                                                                  |
| `initialize_vision_tokenizer(self, tokenizer)`                                                                         | tokenizer: 分词器对象                                                                                                                                           | 无                                                                                                                                                 | 参考`uninavid/model/uninavid_arch.py`，将 6 个特殊 token 详细注册到分词器：`<im_start>`（图像段开始）、`<im_end>`（图像段结束）、`<video_start>`（视频段开始）、`<video_end>`（视频段结束）、`<nav>`（导航任务标识）、`<image>`（普通图片占位符）。注册后自动调整（resize）模型 embedding 层大小，确保能正确处理这些特殊 token 的输入输出。 |


在这里说明一下“余弦相似度”是什么东西：

余弦相似度（Cosine Similarity）是用来衡量两个向量方向有多相似的一种方法。计算时会取两个向量夹角的余弦值，取值范围为[-1, 1]。值越接近1表示越相似，越接近-1表示越不相似。在本项目里，余弦相似度通常用来衡量不同帧特征之间的相似程度，比如决定哪些帧应该聚合。

**导航任务判定**：检查 prompt 首句是否含常量：

```python
NAVIGATION_IDENTIFIER = 'a video of historical observations and an image of the current observation'
```

---

#### `language_model/llava_llama_vid.py` — LLM 封装

```python
class LlavaAttLlamaModel(UniNaVIDMetaModel, LlamaModel): ...
class LlavaLlamaAttForCausalLM(LlamaForCausalLM, UniNaVIDMetaForCausalLM): ...
```

- 注册为 HuggingFace `AutoConfig` / `AutoModelForCausalLM` 的 `"llava"` 类型
- `forward()`：多模态 embedding → Llama backbone → lm_head → CE loss
- `prepare_inputs_for_generation()`：生成时透传 `images` 参数

---

#### `builder.py` — 模型加载

```python
load_pretrained_model(model_path, model_base, model_name, ...)
```

- 模型名含 `vid` 时加载 `LlavaLlamaAttForCausalLM`
- 支持 `model_base` + 仅 `mm_projector.bin` 的分体加载
- 加载后自动调用 `vision_tower.load_model()` 并绑定 `image_processor`

---

#### `multimodal_encoder/` — 视觉编码器


| 文件                | 说明                                      |
| ----------------- | --------------------------------------- |
| `eva_vit.py`      | EVA ViT-g（`EVAVisionTowerLavis`），训练默认使用 |
| `clip_encoder.py` | 标准 CLIP ViT，路径含 `openai` / `laion` 时使用  |
| `builder.py`      | 按路径字符串自动选择编码器类型                         |


---

#### `multimodal_projector/builder.py` — 投影头

支持三种模式：


| 模式             | 结构                              |
| -------------- | ------------------------------- |
| `linear`       | 单层线性                            |
| `mlp{N}x_gelu` | N 层 MLP + GELU（默认 `mlp2x_gelu`） |
| `identity`     | 恒等映射（调试用）                       |


---

### 4.2 训练流程 (`uninavid/train/`)

#### `train.py` — 训练主程序（最大文件，~1400 行）

`**LazySupervisedDataset**` 数据集类：

- 懒加载（仅在 `__getitem__` 时读视频）
- 按 `video_fps=1` 用 `decord` 抽帧
- 导航样本增强（含 `NAV_ID` 的样本）：
  - 随机丢帧（最多 10%）
  - 随机重复帧
  - `random_color_jitter`
  - 保证最后一帧不丢

`**preprocess_imgsp_v1**` Token 化：

```
普通视频：<video_special> <image_sep> <image> × T </video_special>
导航任务：上述 + <image_special> <image> </image_special> + [Navigation]
```

人类轮次 label 全部 mask 为 `IGNORE_INDEX`，仅监督 GPT 回复部分。

`**DataCollatorForSupervisedDataset**` 输出字段：

```python
{
    "input_ids": ...,       # token 序列（含特殊 token 占位）
    "labels": ...,          # 仅 GPT 部分有效，其余为 IGNORE_INDEX
    "attention_mask": ...,
    "images": ...,          # 视频/图像 tensor [T, 3, H, W]
    "prompts": ...          # 原始 prompt，用于 vlm_attention 判断任务类型
}
```

---

#### `train_mem.py` — 训练入口

```python
replace_llama_attn_with_flash_attn()  # 先 patch
import train                           # 再加载训练逻辑
train.train()
```

---

#### `llava_trainer.py` — 自定义 Trainer

- `**LengthGroupedSampler**`：按序列模态长度分组，减少 padding 浪费
- 可选分层学习率（vision tower / projector / LLM 分开设置）

---

#### `llama_flash_attn_monkey_patch.py`

将 Llama 的标准 Attention 替换为 FlashAttention 2，降低长序列显存占用。

---

### 4.3 工具函数


| 文件                | 关键功能                                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| `constants.py`    | `IMAGE_TOKEN_INDEX=-200`；6 个特殊 token 字符串；`NAVIGATION_IDENTIFIER`                                     |
| `conversation.py` | `conv_imgsp_v1`（训练默认）；支持多种对话格式（Vicuna/LLaVA/plain）                                                   |
| `mm_utils.py`     | `process_images()`、`tokenizer_image_token()`、`KeywordsStoppingCriteria`、`get_model_name_from_path()` |


---

### 4.4 离线评测

`**offline_eval_uninavid.py**` — `UniNaVid_Agent` 类


| 方法                       | 功能                           |
| ------------------------ | ---------------------------- |
| `__init__()`             | 加载预训练模型，设置 `run_type="eval"` |
| `reset()`                | 清空帧缓存，重置轨迹                   |
| `act(rgb)`               | 接收当前 RGB 帧，输出动作列表            |
| `draw_traj_arrows_fpv()` | 绘制第一人称视角轨迹箭头                 |


**评测流程**：

```python
agent = UniNaVid_Agent(model_path="model_zoo/uninavid-7b-full-224-video-fps-1-grid-2")
agent.reset()
for frame in video_frames:
    actions = agent.act(frame)  # → ["forward", "left", ...]
# 保存为 result.gif
```

**测试用例结构**：

```
test_cases/vln_1/
├── images/
│   ├── 000.jpg
│   ├── 001.jpg
│   └── ...
└── instruction.json   # {"instruction": "Go to the kitchen"}
```

---

## 5. 关键技术：视频 Token 压缩

这是 Uni-NaVid 相比 LLaVA 类方法的核心差异，解决**长视频历史导致 token 爆炸**的问题。

### 5.1 Grid 池化

```
每帧原始 patch：14×14 = 196 token（224分辨率）
                         ↓ avg_pool2d(kernel=7, stride=7)
历史帧压缩后：  2×2 = 4 token/帧（grid:2）
当前帧保留：    8×8 = 64 token（更高分辨率感知）
```

### 5.2 相似帧合并（训练时：`process_tensor`）

```
时间轴：... [旧帧] [旧帧] [旧帧] ... [近64帧]
                 ↓ 余弦相似度 > 阈值
相似旧帧合并为单个 token → 极大压缩序列长度
近64帧保持4token/帧不变
```

### 5.3 在线增量合并（推理时：`online_process_tensor`）

```
每步新帧到来：
  feat_cache（短期）← 新帧特征
  long_feat_cache（长期） ← 相似帧合并后的压缩特征
无需重新处理所有历史帧，实现 ~5 Hz 在线推理
```

---

## 6. 特殊 Token 体系


| Token              | 含义                                 | 使用场景   |
| ------------------ | ---------------------------------- | ------ |
| `<image>`          | 视觉占位，由 `IMAGE_TOKEN_INDEX=-200` 标识 | 所有视觉输入 |
| `<video_special>`  | 历史视频段开始                            | 视频/导航  |
| `</video_special>` | 历史视频段结束                            | 视频/导航  |
| `<image_sep>`      | 视频帧间分隔符                            | 视频/导航  |
| `<image_special>`  | 当前帧段开始                             | 导航专用   |
| `</image_special>` | 当前帧段结束                             | 导航专用   |
| `[Navigation]`     | 导航任务触发标记                           | 导航专用   |


### 导航 Prompt 结构示意

```
<video_special>
  <image>  <image_sep>
  <image>  <image_sep>
  ...（历史帧，每帧4token）
</video_special>
<image_special>
  <image>（当前帧，64token）
</image_special>
[Navigation]
Go to the kitchen. What is the next action?
```

---

## 7. 数据格式

### 训练 JSON 格式

```json
{
  "id": "NAV_r2r_000123",
  "video": "nav_videos/r2r_episode_123.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "a video of historical observations and an image of the current observation. <image>\nGo straight to the bedroom and turn left at the door. What is the next action?"
    },
    {
      "from": "gpt",
      "value": "forward forward left stop"
    }
  ]
}
```

> **注意**：`id` 含 `NAV_ID` 字样的样本会触发导航专用数据增强。

### 图像样本格式

```json
{
  "id": "img_000456",
  "image": "images/scene_456.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe what you see."},
    {"from": "gpt", "value": "A living room with..."}
  ]
}
```

---

## 8. 训练流程详解

### 两阶段训练


| 项目   | Stage 1               | Stage 2                                   |
| ---- | --------------------- | ----------------------------------------- |
| 基础模型 | `vicuna-7b-v1.5`      | `uninavid-7b-full-224-video-fps-1-grid-2` |
| 启动脚本 | `uninavid_stage_1.sh` | `uninavid_stage_2.sh`                     |
| 目的   | 从语言模型到导航模型            | 在特定任务上继续微调                                |


### 关键超参数

```bash
--version imgsp_v1          # 对话模板
--video_fps 1               # 1 FPS 视频采样
--compress_type "grid:2"    # 4 token/帧压缩
--image_aspect_ratio pad    # 图像 padding
--model_max_length 2048     # 最大序列长度
--bf16 True                 # BF16 训练
--gradient_checkpointing True
--per_device_train_batch_size 8
--gradient_accumulation_steps 2  # 等效 batch 16/卡
--learning_rate 2e-5
--mm_vision_select_layer -2     # 倒数第 2 层特征
```

### 单步 Forward 数据流

```
JSON 样本
  → decord 读视频（1 FPS 抽帧）
  → image_processor → tensor [T, 3, 224, 224]
  → preprocess_imgsp_v1 → input_ids（含特殊 token）
  → DataCollator → batch

LlavaLlamaAttForCausalLM.forward()
  → prepare_inputs_labels_for_multimodal()
       → encode_images(): EVA ViT → grid 压缩 → process_tensor → mm_projector
       → 将视觉 embedding 替换 IMAGE_TOKEN_INDEX 位置
  → LlamaModel(inputs_embeds=mixed_embeds)
  → lm_head → CrossEntropyLoss（仅非 IGNORE_INDEX 位置）
  → 反向传播
```

---

## 9. 推理流程详解

```
agent.reset()
  → model.config.run_type = "eval"
  → initialize_online_inference_nav_feat_cache()
  → feat_cache = [], long_feat_cache = []

agent.act(rgb_frame)  每步调用
  ↓
  rgb_list.append(rgb_frame)
  ↓
  构造 prompt（导航格式，含 <image> 占位）
  ↓
  tokenizer_image_token() → input_ids
  ↓
  model.generate(
      input_ids=input_ids,
      images=[video_tensor],    # [T, 3, 224, 224]
      max_new_tokens=32,
      ...
  )
  ↓
  prepare_inputs_labels_for_multimodal()
    → token_generation(): EVA → grid pool
    → online_process_tensor(): 增量写入 feat_cache / long_feat_cache
    → mm_projector() → 视觉 embedding
  ↓
  Llama 自回归生成
  ↓
  decode → "forward forward left"
  ↓
  split() → ["forward", "forward", "left"]
  ↓
  更新轨迹 [x, y, yaw]
```

**推理性能**：单卡 A100，在线 token 合并约 **5 Hz**。

---

## 10. 依赖环境

### 核心依赖（`pyproject.toml`）


| 类别           | 包                                                          | 版本     |
| ------------ | ---------------------------------------------------------- | ------ |
| 深度学习         | `torch`                                                    | 2.0.1  |
| 深度学习         | `torchvision`                                              | 0.15.2 |
| Transformers | `transformers`                                             | 4.31.0 |
| 分布式训练        | `deepspeed`                                                | 0.9.5  |
| 参数高效微调       | `peft`                                                     | 0.4.0  |
| 量化           | `bitsandbytes`                                             | 0.41.0 |
| 加速           | `accelerate`                                               | 0.21.0 |
| 视觉           | `timm`                                                     | 0.6.13 |
| 视频读取         | `decord`                                                   | latest |
| 其他           | `einops`, `opencv-python`, `sentencepiece`, `scikit-learn` |        |


### 额外安装

```bash
# FlashAttention（train_mem.py 强依赖）
pip install flash-attn==2.5.9.post1

# 项目本身（可编辑安装）
pip install -e .
```

### 外部资源


| 资源                                        | 用途           | 获取方式           |
| ----------------------------------------- | ------------ | -------------- |
| `eva_vit_g.pth`                           | EVA ViT-g 权重 | README 链接      |
| `vicuna-7b-v1.5`                          | 基础语言模型       | HuggingFace    |
| `uninavid-7b-full-224-video-fps-1-grid-2` | 发布权重         | HuggingFace    |
| 导航视频数据                                    | 微调数据         | HuggingFace 子集 |


### VLN-CE / EVT 基准评测

本仓库**不包含**仿真器代码，需配合外部仓库：

- **VLN-CE R2R/RxR**：[NaVid-VLN-CE](https://github.com/jzhzhang/NaVid-VLN-CE)
- **EVT-Bench**：[TrackVLA](https://github.com/wsakobe/TrackVLA)

---

## 11. 快速定位指南


| 目标                    | 文件路径                                                                           |
| --------------------- | ------------------------------------------------------------------------------ |
| **改 token 压缩算法**      | `uninavid/model/uninavid_arch.py` → `process_tensor` / `online_process_tensor` |
| **改导航 prompt 判定逻辑**   | `uninavid/model/uninavid_arch.py` → `vlm_attention()`                          |
| **改多模态 embedding 注入** | `uninavid/model/uninavid_arch.py` → `prepare_inputs_labels_for_multimodal()`   |
| **改 LLM 接口**          | `uninavid/model/language_model/llava_llama_vid.py`                             |
| **改视觉 backbone**      | `uninavid/model/multimodal_encoder/eva_vit.py`                                 |
| **改投影头结构**            | `uninavid/model/multimodal_projector/builder.py`                               |
| **改数据加载 / 增强**        | `uninavid/train/train.py` → `LazySupervisedDataset`                            |
| **改 token 化逻辑**       | `uninavid/train/train.py` → `preprocess_imgsp_v1`                              |
| **改训练超参**             | `scripts/uninavid_stage_1.sh` / `scripts/uninavid_stage_2.sh`                  |
| **改 DeepSpeed 配置**    | `scripts/zero2.json` / `scripts/zero2_offload.json`                            |
| **改推理 demo**          | `offline_eval_uninavid.py` → `UniNaVid_Agent`                                  |
| **改特殊 token**         | `uninavid/constants.py`                                                        |
| **改对话模板**             | `uninavid/conversation.py`                                                     |
| **加载模型**              | `uninavid/model/builder.py` → `load_pretrained_model()`                        |


---

