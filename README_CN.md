<div align="center">

# <img src="assets/onevl_logo_new.png" alt="OneVL Logo" height="48" style="vertical-align:middle"/> OneVL：融合视觉语言解释的单步隐式推理规划框架

[![技术报告](https://img.shields.io/badge/Tech%20Report-arXiv-red?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.18486/)
[![项目主页](https://img.shields.io/badge/Project%20Page-blue?style=flat-square&logo=googlechrome)](https://xiaomi-embodied-intelligence.github.io/OneVL/)
[![模型权重](https://img.shields.io/badge/Model%20Weights-HuggingFace-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/collections/xiaomi-research/onevl-models/)
[![开源协议](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE)

</div>

---

[English](README.md)、[简体中文](README_CN.md)

## 概述

**OneVL** 是面向自动驾驶的视觉-语言-动作（VLA）框架，轨迹预测精度达到业界顶尖水平，推理时延与仅输出答案的自回归模型持平。该框架创新性引入双模态辅助解码器，将语言推理逻辑与未来场景动态信息压缩至精简隐式令牌中，突破了传统隐式思维链算法的固有缺陷。

### 三类思维链实现范式

<div align="center">
<img src="assets/comparison.png" alt="三类思维链范式对比" width="90%"/>
</div>

> **(a) 显式思维链**：先完整输出推理过程再给出结果，可解释性强但推理速度慢。
> 
> **(b) 隐式思维链**：把推理信息压缩为不可解读的隐向量，速度快但丧失可解释性。
> 
> **(c) 本文OneVL**：采用视觉隐令牌`v`与语言隐令牌`l`双结构；训练阶段通过双辅助解码器，分别还原未来画面与推理文本。推理时舍弃解码器，将隐令牌预填充至上下文，兼顾隐式方案的运行速度与显式方案的图文双重可解释能力。

### 整体架构

<div align="center">
<img src="assets/framework.png" alt="OneVL架构图" width="90%"/>
</div>

> 训练过程中，视觉隐位隐藏状态输入**视觉辅助解码器**，预测0.5秒、1秒后的场景画面令牌；语言隐位隐藏状态输入**语言辅助解码器**，还原完整推理文本。模型推理阶段移除两类解码器，隐令牌一次性预填充处理，轨迹依旧采用自回归生成，整体时延与纯答案输出模型一致。

OneVL 基于**通义千问3-VL-4B-指令模型**扩展搭建，核心新增模块如下：

- **隐令牌交互接口**：在回答文本前增设4个视觉隐令牌、2个语言隐令牌，复用原有词表，无需新增特殊标识符。
- **视觉辅助解码器**：依托视觉隐态预测后续帧图像令牌，采用Emu3.5 IBQ 13万码本结构，充当场景世界模型监督约束。
- **语言辅助解码器**：结合视觉特征，从语言隐态中还原可读的逻辑推理文本。
- **预填充推理机制**：推理剔除辅助解码分支，隐令牌并行一次性处理，仅轨迹序列自回归生成，大幅降低耗时。

### 核心创新点

- **双模态辅助解码**：语言分支还原人类可读推理逻辑，视觉分支预判场景未来画面，以物理场景动态约束隐令牌表征。
- **预填充推理策略**：隐令牌单次并行载入上下文，在NAVSIM数据集推理速度较显式思维链提升1.5倍，ROADWork数据集提升2.3倍，时延对标纯答案模型。
- **压缩表征提升泛化性**：目前唯一在四项基准测试中性能全面超越显式自回归思维链的隐式推理算法。

---

## 开源资源汇总

| 项目内容 | 开源状态 |
|-----------|--------|
| 📄 技术论文 | ✅ [查看论文](https://arxiv.org/abs/2604.18486) |
| ⚖️ 模型权重 | ✅ [下载权重](https://huggingface.co/collections/xiaomi-research/onevl-models) |
| 🔍 推理代码 | ✅ [代码仓库](https://github.com/xiaomi-research/onevl)|
| 🏋️ 训练代码 | ✅ [训练源码](https://github.com/GeorgeLuImmortal/OneVL_training/tree/main) |

---

## 实验结果

### 精度-效率权衡曲线（NAVSIM、ROADWork）

<div align="center">
<img src="assets/teaser_bar.png" alt="多数据集精度效率对比" width="90%"/>
</div>

> OneVL 处于图表绿色最优区间，时延最低且预测指标最优。传统隐式思维链算法在自动驾驶任务中表现均不及基础答案模型，OneVL 彻底解决该短板。

### NAVSIM 数据集全量对比

| 算法 | 模型参数量 | PDM评分 ↑ | 推理耗时(秒) ↓ | 可解释性 |
|--------|:----------:|:-----------:|:-------------:|:----------------:|
| AdaThinkDrive | 80亿 | 86.20 | — | 文本推理 |
| LaST-VLA | 80亿 | 87.30 | — | — |
| 纯答案自回归(AR Answer) | 40亿 | 87.47 | <u>4.49</u> | — |
| 显式推理+答案(AR CoT+Answer) | 40亿 | <u>88.29</u> | 6.58 | 文本推理 |
| COCONUT | 40亿 | 84.84 | 5.93 | — |
| CODI | 40亿 | 83.92 | 8.62 | — |
| SIM-CoT | 40亿 | 84.21 | 10.86 | 文本推理 |
| **OneVL** | **40亿** | **88.84** | **4.46** | **视觉+文本双解释** |

### ROADWork 数据集全量对比

| 算法 | 平均位移误差(像素) ↓ | 终点位移误差(像素) ↓ | 推理耗时(秒) ↓ | 可解释性 |
|--------|:----------:|:----------:|:-------------:|:----------------:|
| YNet | 22.68 | 80.78 | — | — |
| 纯答案自回归(AR Answer) | 15.98 | 40.29 | <u>4.74</u> | — |
| 显式推理+答案(AR CoT+Answer) | <u>13.18</u> | <u>29.98</u> | 10.74 | 文本推理 |
| COCONUT | 15.44 | 38.60 | 6.06 | — |
| CODI | 16.45 | 44.28 | 6.73 | — |
| SIM-CoT | 16.49 | 44.32 | 6.19 | 文本推理 |
| **OneVL** | **12.49** | **28.80** | **4.71** | **视觉+文本双解释** |

### Impromptu 数据集全量对比

| 算法            | 平均位移误差(米) ↓ | 终点位移误差(米) ↓ | 推理耗时(秒) ↓ | 可解释性 |
|---------------|:---------:|:---------:|:-------------:|:----------------:|
| Impromptu VLA | 1.60 | 4.28 | 6.10 | — |
| 纯答案自回归(AR Answer)      | 1.46 | 4.03 | <u>4.24</u> | — |
| 显式推理+答案(AR CoT+Answer)       | <u>1.42</u> | <u>3.96</u> | 6.84 | 文本推理 |
| COCONUT       | 1.49 | 4.07 | 5.27 | — |
| CODI          | 1.86 | 5.18 | 5.24 | — |
| SIM-CoT       | 2.43 | 6.10 | 5.09 | 文本推理 |
| **OneVL**     | **1.34** | **3.70** | **4.02** | **视觉+文本双解释** |

### APR1 数据集全量对比

| 算法                     | 平均位移误差(米) ↓ | 终点位移误差(米) ↓ | 推理耗时(秒) ↓ | 可解释性 |
|------------------------|:---------:|:---------:|:-------------:|:----------------:|
| Cosmos-Reason          | <u>2.86</u> | **7.42** | — | 文本推理 |
| 纯答案自回归(AR Answer)               | 3.27 | 9.59 | 3.06 | — |
| 显式推理+答案(AR CoT+Answer) | 2.99 | 8.54 | 3.51 | 文本推理 |
| COCONUT                | 3.29 | 9.48 | 3.76 | — |
| CODI                   | 3.22 | 9.25 | 3.85 | — |
| SIM-CoT                | 3.40 | 9.85 | 3.78 | 文本推理 |
| **OneVL**              | **2.62** | <u>7.53</u> | **3.26** | **视觉+文本双解释** |

### 文本推理质量评估（NAVSIM）

| 算法                | 元动作准确率 ↑ | 语义相似度分数 ↑ | 大模型评测得分 ↑ | 综合均分 ↑ | 推理耗时(秒) ↓ |
|-------------------|:-----------------:|:-----------:|:-----------:|:------:|:------:|
| 显式推理+答案(AR CoT+Answer)         | 73.20 | 79.75 | 81.86 | **78.27** | <u>6.58</u> |
| SIM-CoT           | 67.20 | 76.25 | 78.73 | 74.06 | 10.86 |
| **OneVL（语言辅助分支）** | 71.00 | 78.26 | 79.13 | <u>76.13</u> | **4.46** |

OneVL 语言辅助分支可还原97%的显式推理文本质量，推理速度与纯答案模型持平。

### 消融实验（NAVSIM PDM评分）

| 模型变体 | 语言辅助解码器 | 视觉辅助解码器 | 分阶段训练 | PDM评分 ↑ |
|---------------|:---------------:|:--------------:|:------------:|:-----------:|
| 移除视觉解码器 | ✓ | — | ✓ | 87.97 |
| 移除语言解码器 | — | ✓ | ✓ | 88.53 |
| 无分阶段训练 | ✓ | ✓ | — | 67.13 |
| **完整OneVL模型** | **✓** | **✓** | **✓** | **88.84** |

两类辅助解码模块均对性能有正向增益，分阶段训练为模型核心必要条件，缺失后性能大幅暴跌。

---

## 效果示例

### NAVSIM 场景案例

<div align="center">
<img src="assets/navsim_example1.png" alt="NAVSIM效果示例" width="95%"/>
</div>

> 效果图将真实轨迹（绿色）与预测轨迹（红色）叠加在原车相机画面中，同时展示视觉解码器还原的0.5秒、1秒后预测画面，以及语言分支生成的推理过程文本。

### ROADWork 施工区域通行案例

<div align="center">
<img src="assets/roadwork_example1.png" alt="ROADWork效果示例" width="95%"/>
</div>

---

## 环境部署

**运行要求**：Python 3.10及以上版本，CUDA架构显卡；启用辅助解码功能建议显存不低于16GB。

```bash
# 1. 使用 venv 创建并激活虚拟环境
uv venv venv/onevl --python 3.12
source venv/onevl/bin/activate

# 或使用 conda 
conda create -n onevl python=3.12 -y
conda activate onevl

# 2. 安装依赖库(推荐使用加速源镜像)

# 清华源(推荐)
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 阿里云
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 豆瓣
pip install -r requirements.txt -i http://pypi.douban.com/simple/
```

核心依赖清单（requirements.txt）：
```
torch==2.10.0
torchvision==0.25.0
transformers==4.57.0
safetensors==0.7.0
Pillow>=10.0.0
omegaconf>=2.3.0
einops>=0.7.0
numpy>=1.24.0
```

> **注意**：需安装 4.57.0 及以上版本 transformers，才可兼容 通义千问3(Qwen3VLForConditionalGeneration) 视觉模型调用接口。

---

## 模型推理

### 单卡快速运行

```bash
source venv/onevl/bin/activate

# 仅轨迹预测（最快预填充推理模式）
python infer_onevl.py \
    --model_path /path/to/OneVL-checkpoint \
    --test_set_path test_data/navsim_test.json \
    --image_base_path ""
    --output_path output/navsim/results.json \
    --device cuda:0 \
    --num_latent 2 --num_latent_vis 4 \
    --max_new_tokens 1024 --answer_prefix "[" --prefix_k 0

# 附带文本推理解释
python infer_onevl.py \
    --model_path /path/to/OneVL-checkpoint \
    --test_set_path test_data/navsim_test.json \
    --image_base_path ""
    --output_path output/navsim/results_explain.json \
    --device cuda:0 \
    --num_latent 2 --num_latent_vis 4 \
    --max_new_tokens 1024 --answer_prefix "[" --prefix_k 0 \
    --decoder_explain --aux_visual_condition \
    --c_thought 2 --max_explain_tokens 1024

# 同时输出文本推理+未来画面预测
python infer_onevl.py \
    --model_path /path/to/OneVL-checkpoint \
    --test_set_path test_data/navsim_test.json \
    --image_base_path "" \
    --output_path output/navsim/results_explain.json \
    --device cuda:0 \
    --num_latent 2 --num_latent_vis 4 \
    --max_new_tokens 1024 --answer_prefix "[" --prefix_k 0 \
    --decoder_explain --aux_visual_condition \
    --c_thought 2 --max_explain_tokens 1024 \
    --visual_decoder_explain --visual_aux_visual_condition \
    --c_thought_visual 4 --max_visual_tokens 2560
```

### 多卡并行推理（全量测试集推荐）

```bash
export MODEL_PATH=/path/to/OneVL-checkpoint
export TEST_SET_PATH=test_data/navsim_test.json
export OUTPUT_PATH=output/navsim/navsim_results.json

bash run_infer.sh
```

启动脚本自动检测可用显卡，拆分测试数据集分片并行运算，运行结束后自动合并推理结果。

### 各数据集专属运行脚本

```bash
bash scripts/infer_navsim.sh       # NAVSIM数据集
bash scripts/infer_ar1.sh          # APR1数据集（仅轨迹预测）
bash scripts/infer_roadwork.sh     # ROADWork数据集
bash scripts/infer_impromptu.sh    # Impromptu数据集
```

### 图文双解释推理运行

```bash
bash scripts/infer_ar1_explain.sh  # APR1数据集 图文解释演示
```

### 结果评估

AR1、Impromptu、ROADWork 数据集可直接调用内置评估脚本测算指标：

```bash
# APR1评估
python eval_results.py ar1 \
    --results_json output/ar1/ar1_results.json \
    --test_jsonl test_data/ar1_test.jsonl

# Impromptu评估
python eval_results.py impromptu \
    --results_json output/impromptu/impromptu_results.json \
    --test_jsonl test_data/impromptu_test.jsonl

# ROADWork评估
python eval_results.py roadwork \
    --json_path output/roadwork/roadwork_results.json
```

NAVSIM数据集需转换结果格式后，使用官方评测代码计算分数：

```bash
python output/navsim/convert_to_eval.py \
    --input_path output/navsim/navsim_results.json \
    --ref_path output/navsim/navsim_results_eval.json \
    --output_path output/navsim/navsim_results_for_eval.json
```

---

## 未来画面可视化解析

开启 `--visual_decoder_explain` 参数推理后，输出的 JSON 包含编码预测的未来帧视觉标记的 `visual_decoder_explain` 字段。执行可视化脚本即可还原实景图像：

```bash
source venv/onevl/bin/activate

python scripts/visualize_predict_image_tokens.py \
    --predict_json output/ar1_explain/ar1_results_explain.json \
    --out_dir output/ar1_explain_visualize \
    --model_root /path/to/emu35_model_root \
    -n 20 \
    --device cuda:0
```

单样本输出目录结构：

```
output/ar1_explain_visualize/
└── sample_0000/
    ├── input_00.jpg                  # 原始车载相机画面
    ├── input_01.jpg
    ├── ...
    ├── decoded_from_tokens_00.png    # 0.5秒后预测画面
    ├── decoded_from_tokens_01.png    # 1秒后预测画面
    └── meta.json                     # 推理文本与样本信息
```

该脚本使用独立的 `vq_decoder/` 模块（捆绑的 Emu3.5 IBQ VQ-VAE）—— 不需要外部 Emu3.5 仓库依赖。

`--model_root` 必须包含 `Emu3.5-VisionTokenizer/config.yaml` 和 `Emu3.5-VisionTokenizer/model.ckpt` 。
可从 [BAAI/Emu3.5-VisionTokenizer(文心大模型Emu3.5分词库)](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer) 下载。

---

## 测试数据格式规范

### 数组JSON格式（NAVSIM、ROADWork）

```json
[
  {
    "messages": [{"role": "user", "content": "<image>结合当前画面，预测车辆行驶轨迹……"}],
    "images": ["path/to/frame.jpg"],
    "GT": "[[1.0, 0.0], [2.5, 0.1], ...]"
  }
]
```

### 行式 JSONL 格式（APR1、Impromptu）

每行存储单个样本数据，字段结构与上述一致。

---

## 全局环境变量说明

所有运行脚本均可识别以下环境配置参数：

| 变量名 | 默认值 | 参数说明 |
|----------|---------|-------------|
| `MODEL_PATH` | *(必填)* | OneVL模型权重存放路径 |
| `TEST_SET_PATH` | *(必填)* | 测试集JSON/JSONL文件路径 |
| `OUTPUT_PATH` | `<模型路径>/infer_results/onevl_merged.json` | 推理结果保存路径 |
| `IMAGE_BASE_PATH` | `""` | 图片相对路径前缀补全地址 |
| `NUM_LATENT` | `2` | 语言隐令牌数量 |
| `NUM_LATENT_VIS` | `4` | 视觉隐令牌数量 |
| `MAX_NEW_TOKENS` | `1024` | 轨迹文本最大生成长度 |
| `ANSWER_PREFIX` | `""` | 结果前缀标识 |
| `PREFIX_K` | `0` | 前置填充真实轨迹点数，仅ROADWork任务使用 |
| `DECODER_EXPLAIN` | `false` | 开启语言推理解释功能 |
| `AUX_VISUAL_CONDITION` | `true` | 语言解码依托视觉特征输入 |
| `C_THOUGHT` | `2` | 语言解码器读取隐令牌个数 |
| `MAX_EXPLAIN_TOKENS` | `1024` | 推理文本最大生成字数 |
| `VISUAL_DECODER_EXPLAIN` | `false` | 开启未来画面预测功能 |
| `VISUAL_AUX_VISUAL_CONDITION` | `true` | 视觉解码依托视觉特征输入 |
| `C_THOUGHT_VISUAL` | `4` | 视觉解码器读取隐令牌个数 |
| `MAX_VISUAL_TOKENS` | `2560` | 画面令牌最大生成数量 |

---

## 引用格式

若该项目对你的研究有所帮助，可引用如下文献：

```bibtex
@article{lu2026onevl,
  title={OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation},
  author={Lu, Jinghui and Guan, Jiayi and Huang, Zhijian and Li, Jinlong and Li, Guang and Kong, Lingdong and Li, Yingyan and Wang, Han and Xu, Shaoqing and Luo, Yuechen and others},
  journal={arXiv preprint arXiv:2604.18486},
  year={2026},
  url={https://arxiv.org/abs/2604.18486}
}
```

---

## 开源协议

本项目基于 **Apache 2.0** 协议开源。

模型权重基于 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)，视觉分词器来自 [Emu3.5-VisionTokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer)；使用时请同步遵守对应开源许可协议。

---

## 致谢

- [Qwen3-VL(通义千问3视觉模型)](https://github.com/QwenLM/Qwen3-VL) — 基础视觉语言主干网络
- [Emu3.5](https://github.com/baaivision/Emu3) — 图像矢量量化分词工具
- [AdaThinkDrive](https://github.com/luo-yc17/AdaThinkDrive/tree/main) — NAVSIM数据集推理标注数据
- [NAVSIM](https://github.com/autonomousvision/navsim)、[ROADWork](https://github.com/anuragxel/roadwork-dataset)、[Impromptu](https://github.com/ahydchh/Impromptu-VLA) — 自动驾驶评测基准数据集