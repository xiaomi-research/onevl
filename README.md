<div align="center">

# <img src="assets/onevl_logo_new.png" alt="OneVL Logo" height="48" style="vertical-align:middle"/> OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanations

[![Tech Report](https://img.shields.io/badge/Tech%20Report-arXiv-red?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.18486/)
[![Project Page](https://img.shields.io/badge/Project%20Page-blue?style=flat-square&logo=googlechrome)](https://xiaomi-embodied-intelligence.github.io/OneVL/)
[![Model Weights](https://img.shields.io/badge/Model%20Weights-HuggingFace-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/collections/xiaomi-research/onevl-models/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)](LICENSE)

</div>

[English](README.md)、[简体中文](README_CN.md)

---
## 📅 Changelog
- 🔥 **2026.05.22** Released *OneVL NAVSIM MLP head* model, supporting faster waypoint decoding, visual explanation and text explanation.
- 🔥 **2026.05.11** Released [*OneVL training code*](https://github.com/GeorgeLuImmortal/OneVL_training/tree/main).

## Overview

**OneVL** is a Vision-Language-Action (VLA) framework for autonomous driving that achieves **state-of-the-art trajectory prediction accuracy** with **inference latency matching answer-only AR models**. It overcomes the fundamental limitations of prior latent Chain-of-Thought (CoT) methods by introducing dual-modal auxiliary decoders that supervise compact latent tokens to encode both linguistic reasoning and future scene dynamics.

### Three CoT Paradigms

<div align="center">
<img src="assets/comparison.png" alt="Comparison of three CoT paradigms" width="90%"/>
</div>

> **(a) Explicit CoT** generates a full reasoning chain before the answer — interpretable but slow. **(b) Implicit CoT** compresses reasoning into opaque latent vectors — fast but not interpretable. **(c) OneVL (ours)** uses visual latent tokens `v` and language latent tokens `l`; during training, dual auxiliary decoders decode these into future frames and CoT text respectively. At inference, decoders are discarded and latents are **prefilled** into the prompt — matching the speed of (b) while recovering the interpretability of (a) in both vision and language.

### Architecture

<div align="center">
<img src="assets/framework.png" alt="OneVL architecture" width="90%"/>
</div>

> During training, hidden states at visual latent positions are routed to the **Visual Aux. Decoder** (predicts future-frame visual tokens at t+0.5s and t+1.0s) and at language latent positions to the **Language Aux. Decoder** (reconstructs CoT text). Both decoders are discarded at inference; all latent tokens are **prefilled** into the prompt, matching answer-only AR prediction latency.

OneVL augments **Qwen3-VL-4B-Instruct** with:

- **Latent Token Interface** — 4 visual latent tokens + 2 language latent tokens placed in the assistant response before the answer, using existing vocabulary tokens (no new special tokens).
- **Visual Auxiliary Decoder** — Predicts future-frame visual tokens at t+0.5s and t+1.0s from visual latent hidden states (Emu3.5 IBQ, 131k codebook), acting as a **world model** supervision signal.
- **Language Auxiliary Decoder** — Reconstructs explicit CoT reasoning text from language latent hidden states, conditioned on ViT visual features.
- **Prefill Inference** — Both decoders are discarded at inference; latent tokens are processed in one parallel pass with only the trajectory generated autoregressively.

### Key Innovations

- **Dual-Modal Auxiliary Decoders**: A *language auxiliary decoder* reconstructs human-readable CoT reasoning from language latent tokens; a *visual auxiliary decoder* predicts future scene frames from visual latent tokens, acting as a **world model** that grounds the latents in physical scene dynamics.
- **Prefill Inference**: All latent tokens are prefilled into the prompt context in a single parallel pass — **1.5× faster than explicit CoT on NAVSIM, 2.3× faster on ROADWork** — with latency essentially identical to answer-only AR prediction.
- **Compression Drives Generalization**: OneVL is the **only latent CoT method that outperforms explicit autoregressive CoT** across all four benchmarks.

---

## Open-Source Status

| Component | Status |
|-----------|--------|
| 📄 Technical Report | ✅ [Tech report](https://arxiv.org/abs/2604.18486) |
| ⚖️ Model Weights | ✅ [Weights](https://huggingface.co/collections/xiaomi-research/onevl-models) |
| 🔍 Inference Code | ✅ [Code](https://github.com/xiaomi-research/onevl)|
| 🏋️ Training Code | ✅ [Code](https://github.com/GeorgeLuImmortal/OneVL_training/tree/main) |

---

## Results

### Accuracy–Efficiency Pareto (NAVSIM & ROADWork)

<div align="center">
<img src="assets/teaser_bar.png" alt="Teaser: Accuracy-Efficiency Pareto across benchmarks" width="90%"/>
</div>

> OneVL lands in the **green-shaded optimal corner** (lowest latency, best metric) on both benchmarks. All prior latent CoT methods (COCONUT, CODI, SIM-CoT) underperform even the AR Answer baseline on driving tasks — a critical failure that OneVL overcomes.

### NAVSIM — Full Comparison

| Method | Model Size | PDM-score ↑ | Latency (s) ↓ | Interpretability |
|--------|:----------:|:-----------:|:-------------:|:----------------:|
| AdaThinkDrive | 8B | 86.20 | — | Language |
| LaST-VLA | 8B | 87.30 | — | — |
| AR Answer | 4B | 87.47 | <u>4.49</u> | — |
| AR CoT+Answer | 4B | <u>88.29</u> | 6.58 | Language |
| COCONUT | 4B | 84.84 | 5.93 | — |
| CODI | 4B | 83.92 | 8.62 | — |
| SIM-CoT | 4B | 84.21 | 10.86 | Language |
| **OneVL** | **4B** | **88.84** | **4.46** | **Vision + Language** |

### ROADWork — Full Comparison

| Method | ADE (px) ↓ | FDE (px) ↓ | Latency (s) ↓ | Interpretability |
|--------|:----------:|:----------:|:-------------:|:----------------:|
| YNet | 22.68 | 80.78 | — | — |
| AR Answer | 15.98 | 40.29 | <u>4.74</u> | — |
| AR CoT+Answer | <u>13.18</u> | <u>29.98</u> | 10.74 | Language |
| COCONUT | 15.44 | 38.60 | 6.06 | — |
| CODI | 16.45 | 44.28 | 6.73 | — |
| SIM-CoT | 16.49 | 44.32 | 6.19 | Language |
| **OneVL** | **12.49** | **28.80** | **4.71** | **Vision + Language** |

### Impromptu — Full Comparison

| Method | ADE (m) ↓ | FDE (m) ↓ | Latency (s) ↓ | Interpretability |
|--------|:---------:|:---------:|:-------------:|:----------------:|
| Impromptu VLA | 1.60 | 4.28 | 6.10 | — |
| AR Answer | 1.46 | 4.03 | <u>4.24</u> | — |
| AR CoT+Answer | <u>1.42</u> | <u>3.96</u> | 6.84 | Language |
| COCONUT | 1.49 | 4.07 | 5.27 | — |
| CODI | 1.86 | 5.18 | 5.24 | — |
| SIM-CoT | 2.43 | 6.10 | 5.09 | Language |
| **OneVL** | **1.34** | **3.70** | **4.02** | **Vision + Language** |

### APR1 — Full Comparison

| Method | ADE (m) ↓ | FDE (m) ↓ | Latency (s) ↓ | Interpretability |
|--------|:---------:|:---------:|:-------------:|:----------------:|
| Cosmos-Reason | <u>2.86</u> | **7.42** | — | Language |
| AR Answer | 3.27 | 9.59 | 3.06 | — |
| AR CoT+Answer | 2.99 | 8.54 | 3.51 | Language |
| COCONUT | 3.29 | 9.48 | 3.76 | — |
| CODI | 3.22 | 9.25 | 3.85 | — |
| SIM-CoT | 3.40 | 9.85 | 3.78 | Language |
| **OneVL** | **2.62** | <u>7.53</u> | **3.26** | **Vision + Language** |

### Text CoT Quality (NAVSIM)

| Method | Meta Action Acc. ↑ | STS Score ↑ | LLM Judge ↑ | Avg. ↑ | Latency (s) ↓ |
|--------|:-----------------:|:-----------:|:-----------:|:------:|:------:|
| AR CoT+Answer | 73.20 | 79.75 | 81.86 | **78.27** | <u>6.58</u> |
| SIM-CoT | 67.20 | 76.25 | 78.73 | 74.06 | 10.86 |
| **OneVL** (lang. aux.) | 71.00 | 78.26 | 79.13 | <u>76.13</u> | **4.46** |

OneVL's language auxiliary decoder recovers 97% of explicit CoT quality while running at answer-only speed.

### Ablation Study (NAVSIM PDM-score)

| Model Variant | Lang. Aux. Dec. | Vis. Aux. Dec. | Staged Train | PDM-score ↑ |
|---------------|:---------------:|:--------------:|:------------:|:-----------:|
| OneVL w/o vis. dec. | ✓ | — | ✓ | 87.97 |
| OneVL w/o lang. dec. | — | ✓ | ✓ | 88.53 |
| OneVL w/o staged train | ✓ | ✓ | — | 67.13 |
| **OneVL (full)** | **✓** | **✓** | **✓** | **88.84** |

Both auxiliary decoders contribute measurably; staged training is essential (without it, performance collapses to 67.13).

---

## Qualitative Examples

### NAVSIM

<div align="center">
<img src="assets/navsim_example1.png" alt="NAVSIM qualitative example" width="95%"/>
</div>

> Each plot overlays ground-truth (green) and predicted (red) trajectories on the front camera view, along with predicted future frames at t+0.5s and t+1.0s decoded from the visual auxiliary decoder, and the language CoT from the language auxiliary decoder.

### ROADWork (Construction Zone Navigation)

<div align="center">
<img src="assets/roadwork_example1.png" alt="ROADWork qualitative example" width="95%"/>
</div>

---

## Environment Setup

**Requirements:** Python 3.10+, CUDA GPU (≥16 GB VRAM recommended for inference with aux decoders).

```bash
# 1. Create and activate virtual environment
uv venv venv/onevl --python 3.12
source venv/onevl/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

Core packages (`requirements.txt`):

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

> **Note:** `transformers ≥ 4.57.0` is required for `Qwen3VLForConditionalGeneration` support.

---

## Inference

### Quick Start (Single GPU)

```bash
source venv/onevl/bin/activate

# Trajectory prediction only (fastest, prefill inference)
python infer_onevl.py \
    --model_path /path/to/OneVL-checkpoint \
    --test_set_path test_data/navsim_test.json \
    --image_base_path ""
    --output_path output/navsim/results.json \
    --device cuda:0 \
    --num_latent 2 --num_latent_vis 4 \
    --max_new_tokens 1024 --answer_prefix "[" --prefix_k 0

# With language explanation (text CoT from language aux decoder)
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

# With both language + visual explanation (text CoT + future frame tokens)
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

### Multi-GPU Inference (recommended for full test sets)

```bash
export MODEL_PATH=/path/to/OneVL-checkpoint
export TEST_SET_PATH=test_data/navsim_test.json
export OUTPUT_PATH=output/navsim/navsim_results.json

bash run_infer.sh
```

The launcher auto-detects available GPUs, shards the test set, runs inference in parallel across all GPUs, and merges results.

### Per-Benchmark Scripts

```bash
bash scripts/infer_navsim.sh       # NAVSIM
bash scripts/infer_navsim_mlp.sh   # NAVSIM MLP head
bash scripts/infer_ar1.sh          # APR1 (trajectory only)
bash scripts/infer_roadwork.sh     # ROADWork
bash scripts/infer_impromptu.sh    # Impromptu
```

### For visual cot/text cot explain
```bash
bash scripts/infer_ar1_explain.sh  # APR1 (language + visual explanations, use APR1 as example)
```

### Evaluation

AR1, Impromptu, and ROADWork can be evaluated directly with the bundled evaluation script:

```bash
# AR1
python eval_results.py ar1 \
    --results_json output/ar1/ar1_results.json \
    --test_jsonl test_data/ar1_test.jsonl

# Impromptu
python eval_results.py impromptu \
    --results_json output/impromptu/impromptu_results.json \
    --test_jsonl test_data/impromptu_test.jsonl

# ROADWork
python eval_results.py roadwork \
    --json_path output/roadwork/roadwork_results.json
```

NAVSIM uses the official NAVSIM evaluation pipeline. First convert OneVL inference results to the NAVSIM test format, then evaluate the converted file with the [NAVSIM(v1.0 branch)](https://github.com/autonomousvision/navsim) codebase:

```bash
python output/navsim/convert_to_eval.py \
    --input_path output/navsim/navsim_results.json \
    --ref_path output/navsim/navsim_results_eval.json \
    --output_path output/navsim/navsim_results_for_eval.json
```


---

## Visualizing Future-Frame Predictions

After running inference with `--visual_decoder_explain`, the output JSON contains `visual_decoder_explain` fields encoding predicted future-frame visual tokens. Use the visualization script to decode them back to images:

```bash
source venv/onevl/bin/activate

python scripts/visualize_predict_image_tokens.py \
    --predict_json output/ar1_explain/ar1_results_explain.json \
    --out_dir output/ar1_explain_visualize \
    --model_root /path/to/emu35_model_root \
    -n 20 \
    --device cuda:0
```

**Output layout per sample:**

```
output/ar1_explain_visualize/
└── sample_0000/
    ├── input_00.jpg                  # original camera frame(s)
    ├── input_01.jpg
    ├── ...
    ├── decoded_from_tokens_00.png    # predicted future frame at t+0.5s
    ├── decoded_from_tokens_01.png    # predicted future frame at t+1.0s
    └── meta.json                     # CoT text + metadata
```

The script uses the self-contained `vq_decoder/` module (bundled Emu3.5 IBQ VQ-VAE) — no external Emu3.5 repo dependency required.

`--model_root` must contain `Emu3.5-VisionTokenizer/config.yaml` and `Emu3.5-VisionTokenizer/model.ckpt`. Download from [BAAI/Emu3.5-VisionTokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer).

---

## Test Data Format

### JSON array (NAVSIM, ROADWork)

```json
[
  {
    "messages": [{"role": "user", "content": "<image>Based on the current image, predict ..."}],
    "images": ["path/to/frame.jpg"],
    "GT": "[[1.0, 0.0], [2.5, 0.1], ...]"
  }
]
```

### JSONL (APR1, Impromptu)

One JSON object per line, same schema as above.

---

**Environment variables** accepted by all scripts:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *(required)* | Path to the OneVL checkpoint |
| `TEST_SET_PATH` | *(required)* | Test JSON / JSONL file |
| `OUTPUT_PATH` | `<MODEL_PATH>/infer_results/onevl_merged.json` | Where to write merged results |
| `IMAGE_BASE_PATH` | `""` | Prepended to relative image paths |
| `NUM_LATENT` | `2` | Number of language latent tokens |
| `NUM_LATENT_VIS` | `4` | Number of visual latent tokens |
| `MAX_NEW_TOKENS` | `1024` | Max answer tokens to generate |
| `ANSWER_PREFIX` | `""` | Prefix after `<answer>` (e.g. `[` for NAVSIM, `[[` for APR1) |
| `PREFIX_K` | `0` |  Prefill first K GT waypoints after `<answer>` (default: 0), only used on ROADWork |
| `DECODER_EXPLAIN` | `false` | Enable language auxiliary decoder |
| `AUX_VISUAL_CONDITION` | `true` | *(if DECODER_EXPLAIN=true)* Condition language aux decoder on ViT features (`--aux_visual_condition`) |
| `C_THOUGHT` | `2` | *(if DECODER_EXPLAIN=true)* Number of latent tokens read by language aux decoder |
| `MAX_EXPLAIN_TOKENS` | `1024` | *(if DECODER_EXPLAIN=true)* Max tokens generated by language aux decoder |
| `VISUAL_DECODER_EXPLAIN` | `false` | Enable visual auxiliary decoder |
| `VISUAL_AUX_VISUAL_CONDITION` | `true` | *(if VISUAL_DECODER_EXPLAIN=true)* Condition visual aux decoder on ViT features (`--visual_aux_visual_condition`) |
| `C_THOUGHT_VISUAL` | `4` | *(if VISUAL_DECODER_EXPLAIN=true)* Number of latent tokens read by visual aux decoder |
| `MAX_VISUAL_TOKENS` | `2560` | *(if VISUAL_DECODER_EXPLAIN=true)* Max visual tokens generated by visual aux decoder |

--- 

## Citation

If you find this work useful, please cite:

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

## License

This project is released under the [Apache 2.0 License](LICENSE).

Model weights are built on [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) and the visual tokenizer is from [Emu3.5-VisionTokenizer](https://huggingface.co/BAAI/Emu3.5-VisionTokenizer); please refer to their respective licenses as well.

---

## Acknowledgements

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — backbone VLM
- [Emu3.5](https://github.com/baaivision/Emu3) — IBQ visual tokenizer
- [AdaThinkDrive](https://github.com/luo-yc17/AdaThinkDrive/tree/main) — NAVSIM CoT annotations
- [NAVSIM](https://github.com/autonomousvision/navsim), [ROADWork](https://github.com/anuragxel/roadwork-dataset), [Impromptu](https://github.com/ahydchh/Impromptu-VLA) — evaluation benchmarks
