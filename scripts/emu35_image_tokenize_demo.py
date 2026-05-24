#!/usr/bin/env python3
"""
Emu3.5 图片量化到 token / 从 token 反量化到图片的简易脚本。
参考官方 Emu3.5 代码，只做：图片 -> VQ 编码 -> token id -> 保存 txt；
从 txt 读 token id -> VQ 解码 -> 图片。

使用前请激活环境: source lujinghui/venv/emu35/bin/activate
本地模型目录: lujinghui/models/emu35 (需包含 Emu3.5-VisionTokenizer 的 config.yaml + model.ckpt，以及 Emu3.5-Image 的 tokenizer)
"""

import argparse
import os
import sys

# 把 Emu3.5 官方代码加入路径（与脚本所在目录平级的 Emu3.5 仓库）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMU35_SRC = os.path.join(SCRIPT_DIR, "Emu3.5", "src")
if os.path.isdir(EMU35_SRC):
    sys.path.insert(0, EMU35_SRC)
else:
    # 若脚本被复制到别处，可设置环境变量
    EMU35_SRC = os.environ.get("EMU35_SRC", "")
    if EMU35_SRC:
        sys.path.insert(0, EMU35_SRC)

import re
import numpy as np
import torch
from PIL import Image

# Emu3.5 图像 special tokens（与官方一致，解码时可直接解析 txt 无需加载 tokenizer）
BOI_TOKEN = "<|image start|>"
EOI_TOKEN = "<|image end|>"
EOL_TOKEN = "<|extra_200|>"
IMG_TOKEN = "<|image token|>"
VISUAL_TOKEN_PATTERN = re.compile(r"<\|visual token (\d+)\|>")


def _get_embed_dim(vq_model):
    """从 VQ 模型拿到 embed_dim，decode_code 的 shape 里要用。"""
    if hasattr(vq_model, "quantize") and hasattr(vq_model.quantize, "embed_dim"):
        return vq_model.quantize.embed_dim
    if hasattr(vq_model, "config"):
        return getattr(vq_model.config, "embed_dim", 256)
    return 256


def image_to_tokens(image_path, vq_model, image_area=512 * 512, device="cuda:0"):
    """
    图片 -> 量化 token 网格 (H, W)。
    """
    from utils.input_utils import smart_resize

    image = Image.open(image_path).convert("RGB")
    image = smart_resize(image, area=image_area, ds_factor=16)
    w, h = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image_tensor = torch.tensor((np.array(image) / 127.5 - 1.0)).to(device, dtype).permute(2, 0, 1)
    _, _, info = vq_model.encode(image_tensor[None])
    # info 为 (None, None, ind)，ind 为 flatten 的索引
    ind = info[-1]
    token_h, token_w = h // 16, w // 16
    token_grid = ind.view(1, token_h, token_w)[0]  # (H, W)
    return token_grid


def tokens_to_image(token_grid, vq_model, embed_dim=256, device="cuda:0"):
    """
    token 网格 (H, W) -> 解码为 PIL Image。
    token_grid: numpy 或 torch (H, W)，dtype 为整数。
    """
    if isinstance(token_grid, np.ndarray):
        token_grid = torch.from_numpy(token_grid).long()
    token_grid = token_grid.to(device)
    h, w = token_grid.shape
    # decode_code 需要 shape=(batch, h, w, embed_dim)
    image = vq_model.decode_code(
        token_grid[None], shape=(1, h, w, embed_dim)
    ).float()
    image = image[0].permute(1, 2, 0)
    image = Image.fromarray(
        ((image + 1.0) * 127.5).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
    )
    return image


def format_image_string_with_special_tokens(token_grid, tokenizer=None, newline_after_eol=True):
    """
    将 token 网格格式化为带 special tokens 的字符串：
    <|image start|>H*W<|image token|><|visual token 000001|>...<|extra_200|>
    下一行 visual tokens...<|extra_200|>
    ...<|image end|>
    newline_after_eol=True 时在每行末尾的 EOL 后加真实换行，便于阅读。
    """
    if isinstance(token_grid, torch.Tensor):
        token_grid = token_grid.cpu().numpy()
    boi = tokenizer.boi_token if tokenizer else BOI_TOKEN
    eoi = tokenizer.eoi_token if tokenizer else EOI_TOKEN
    eol = tokenizer.eol_token if tokenizer else EOL_TOKEN
    img = tokenizer.img_token if tokenizer else IMG_TOKEN
    h, w = token_grid.shape
    rows = []
    for i in range(h):
        row_str = "".join(
            "<|visual token {:0>6d}|>".format(int(token_grid[i, j])) for j in range(w)
        )
        rows.append(row_str)
    sep = eol + ("\n" if newline_after_eol else "")
    body = sep.join(rows)
    return f"{boi}{h}*{w}{img}{body}{eoi}"


def save_token_ids_txt(token_grid, out_txt_path, tokenizer=None):
    """
    把 token 网格保存为 txt，以文字形式并包含 special tokens：
    <|image start|>H*W<|image token|><|visual token 000001|>...<|extra_200|>换行...<|image end|>
    tokenizer 用于编码时传入（含 boi/eoi/eol/img）；解码读 txt 不需要 tokenizer。
    """
    if isinstance(token_grid, torch.Tensor):
        token_grid = token_grid.cpu().numpy()
    text = format_image_string_with_special_tokens(token_grid, tokenizer)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    h, w = token_grid.shape
    print(f"Token ids saved to {out_txt_path} (shape {h} x {w}, with BOI/EOI/EOL)")


def parse_token_block(content):
    """
    从单块字符串解析 token 网格（<|image start|>...<|image end|>）。
    与 load_token_ids_txt 第一分支逻辑一致，用于从 future_image_tokens 等多块拼接字符串中解析单块。
    """
    content = content.strip()
    if not (content.startswith(BOI_TOKEN) and content.endswith(EOI_TOKEN)):
        raise ValueError("Content is not a valid image token block (missing BOI/EOI)")
    inner = content[len(BOI_TOKEN) : -len(EOI_TOKEN)]
    if IMG_TOKEN in inner:
        inner = inner.split(IMG_TOKEN, 1)[1]
    rows_str = inner.split(EOL_TOKEN)
    grid = []
    for r in rows_str:
        ids = VISUAL_TOKEN_PATTERN.findall(r)
        if ids:
            grid.append([int(x) for x in ids])
    if not grid:
        raise ValueError("No visual tokens found in block")
    W = len(grid[0])
    for i in range(1, len(grid)):
        row = grid[i]
        if len(row) > W:
            grid[i] = row[:W]
        elif len(row) < W:
            prev = grid[i - 1]
            pad_token = prev[-1] if prev else 0
            grid[i] = row + [pad_token] * (W - len(row))
    return np.array(grid, dtype=np.int64)


def load_token_ids_txt(txt_path):
    """
    从 txt 读回 token 网格。支持两种格式：
    1) 带 special tokens 的文本：<|image start|>H*W<|image token|>...<|image end|>
    2) 旧版纯数字：第一行 "H W"，之后每行 W 个 id（空格分隔）
    """
    with open(txt_path, encoding="utf-8") as f:
        content = f.read()
    content = content.strip()
    # 判断是否为带 special tokens 的格式
    if content.startswith(BOI_TOKEN) and content.endswith(EOI_TOKEN):
        # 去掉头尾得到中间： H*W<|image token|>... 或直接 ...<|visual token ...|>...
        inner = content[len(BOI_TOKEN) : -len(EOI_TOKEN)]
        # 去掉开头的 "H*W<|image token|>"
        if IMG_TOKEN in inner:
            inner = inner.split(IMG_TOKEN, 1)[1]
        # 按换行符 EOL 分行
        rows_str = inner.split(EOL_TOKEN)
        grid = []
        for r in rows_str:
            ids = VISUAL_TOKEN_PATTERN.findall(r)
            if ids:
                grid.append([int(x) for x in ids])
        if not grid:
            raise ValueError(f"No visual tokens found in {txt_path}")

        # 异常处理：统一每行 visual token 数量（以第一行长度为 W）
        # 多了截断，少了用上一行的最后一个 token 复制填充
        W = len(grid[0])
        for i in range(1, len(grid)):
            row = grid[i]
            if len(row) > W:
                grid[i] = row[:W]
            elif len(row) < W:
                prev = grid[i - 1]
                pad_token = prev[-1] if prev else 0
                grid[i] = row + [pad_token] * (W - len(row))

        return np.array(grid, dtype=np.int64)
    # 旧版：首行 "H W"
    lines = content.splitlines()
    if not lines:
        raise ValueError(f"Empty file {txt_path}")
    parts = lines[0].split()
    print(parts)
    if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError(f"Expected first line 'H W' in {txt_path}")
    h, w = int(parts[0]), int(parts[1])
    grid = []
    for i in range(1, 1 + h):
        if i >= len(lines):
            raise ValueError(f"Expected {h} data lines in {txt_path}")
        row = list(map(int, lines[i].split()))
        if len(row) != w:
            raise ValueError(f"Row {i} has {len(row)} values, expected {w}")
        grid.append(row)
    return np.array(grid, dtype=np.int64)


def load_vision_tokenizer(model_root, device="cuda:0"):
    """加载 VQ 模型。优先本地 config.yaml + model.ckpt，否则从 HF 拉取。"""
    import os.path as osp
    from vision_tokenizer import build_vision_tokenizer

    vq_path = os.path.join(model_root, "Emu3.5-VisionTokenizer")
    ckpt_path = os.path.join(vq_path, "model.ckpt")
    if osp.isfile(ckpt_path):
        return build_vision_tokenizer("ibq", vq_path, device=device)
    try:
        return build_vision_tokenizer("ibq", "BAAI/Emu3.5-VisionTokenizer", device=device)
    except Exception as e:
        raise RuntimeError(
            f"请在 {vq_path} 下放置 model.ckpt（可从 HuggingFace BAAI/Emu3.5-VisionTokenizer 下载），或确保可访问 HF。\n{e}"
        )


def load_text_tokenizer(model_root):
    """加载文本 tokenizer（仅解码图像字符串时需要；本脚本只做 token id 的 txt，可不加载）。"""
    import os.path as osp
    tokenizer_path = os.path.join(model_root, "Emu3.5-Image")
    # 优先使用 Emu3.5 自带的 tokenizer 源码，避免 HF 目录缺 tokenization_emu3.py
    tokenizer_emu3_dir = os.path.join(SCRIPT_DIR, "Emu3.5", "src", "tokenizer_emu3_ibq")
    if os.path.isfile(os.path.join(tokenizer_emu3_dir, "tokenization_emu3.py")):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_emu3_dir,
            special_tokens_file=osp.join(tokenizer_emu3_dir, "emu3_vision_tokens.txt"),
            trust_remote_code=True,
        )
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            special_tokens_file=osp.join(tokenizer_path, "emu3_vision_tokens.txt"),
            trust_remote_code=True,
        )
    tokenizer.eol_token = "<|extra_200|>"
    tokenizer.img_token = "<|image token|>"
    tokenizer.boi_token = "<|image start|>"
    tokenizer.eoi_token = "<|image end|>"
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Emu3.5 image <-> token 量化/反量化")
    parser.add_argument("--model_root", type=str, default="opendata/roadworks/models/emu35",
                        help="模型根目录，默认 lujinghui/models/emu35")
    parser.add_argument("--image", type=str, default=None, help="输入图片路径（编码时必填）")
    parser.add_argument("--token_txt", type=str, default="lujinghui/los_angeles_tokens_text.txt",
                        help="token id 的 txt 路径：编码时写出；解码时读入")
    parser.add_argument("--out_image", type=str, default="decoded.png", help="解码得到的图片保存路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_area", type=int, default=512 * 512,
                        help="编码时目标像素面积，越小 token 越少（默认 256*256；原 512*512 约 4 倍 token）")
    args = parser.parse_args()

    # 默认模型根目录：相对脚本所在目录的 lujinghui/models/emu35
    print(f"Loading model from {args.model_root}")
    vq_model = load_vision_tokenizer(args.model_root, device=args.device)
    embed_dim = _get_embed_dim(vq_model)

    if args.image:
        # 编码：图片 -> token，并保存为带 BOI/EOI/EOL 的文本 txt
        token_grid = image_to_tokens(
            args.image, vq_model,
            image_area=args.image_area, device=args.device
        )
        out_txt = args.token_txt or (args.image.rsplit(".", 1)[0] + "_tokens.txt")
        tokenizer = load_text_tokenizer(args.model_root)
        save_token_ids_txt(token_grid, out_txt, tokenizer=tokenizer)
        # 可选：立刻用同一批 token 解码回图片做简单测试
        if args.out_image:
            recon = tokens_to_image(token_grid, vq_model, embed_dim=embed_dim, device=args.device)
            recon.save(args.out_image)
            print(f"Reconstructed image saved to {args.out_image}")
    elif args.token_txt:
        # 解码：从 txt 读 token id -> 图片
        token_grid = load_token_ids_txt(args.token_txt)
        out_image = args.out_image or args.token_txt.replace(".txt", "_decoded.png")
        image = tokens_to_image(token_grid, vq_model, embed_dim=embed_dim, device=args.device)
        image.save(out_image)
        print(f"Decoded image saved to {out_image}")
    else:
        parser.print_help()
        print("\n示例:")
        print("  编码并保存 token、并解码回图片:")
        print('  python emu35_image_tokenize_demo.py --image path/to/img.jpg --token_txt out_tokens.txt --out_image out_recon.png')
        print("  仅从 token txt 解码:")
        print('  python emu35_image_tokenize_demo.py --token_txt out_tokens.txt --out_image out.png')


if __name__ == "__main__":
    main()
