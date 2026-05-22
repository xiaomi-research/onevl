"""
OneVL (Latent CoT) standalone inference script for Qwen3-VL.

Supports:
  - Standard latent-token inference (generate answer from latent prefix)
  - Optional aux decoder explain: decode the latent hidden states into
    explicit reasoning text using the trained auxiliary text decoder
  - Optional visual aux decoder explain: decode latent states into
    future visual tokens using the trained visual auxiliary decoder
  - Optional MLP float head: regress numeric waypoints directly from the
    last-text-latent hidden state (skip language-model decoding)
  - Single-image and multi-image samples (auto-detected from test set)

The trained checkpoint stores all weights (base model + aux decoder + projections)
in the same safetensors files.  This script extracts sub-module weights by prefix
and is fully self-contained — no training framework dependency.
"""

import sys
import os
import json
import ast
import argparse
import glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    AutoConfig,
)
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# GT / waypoint parsing for coordinate prefill (--prefix_k)
# ---------------------------------------------------------------------------

def parse_gt_waypoints(gt_str):
    if not gt_str or not isinstance(gt_str, str):
        return []
    s = gt_str.strip()
    if not s:
        return []
    try:
        data = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        try:
            data = ast.literal_eval('[' + s + ']')
        except (SyntaxError, ValueError):
            return []
    if not data:
        return []
    if isinstance(data[0], (int, float)):
        return [list(data)]
    return [list(p) for p in data]


def format_gt_prefix_points(points):
    parts = []
    for p in points:
        inner = ", ".join(
            str(int(x)) if isinstance(x, float) and x == int(x) else str(x)
            for x in p
        )
        parts.append(f"[{inner}]")
    return ", ".join(parts) + ","


# ---------------------------------------------------------------------------
# Latent-position detection helpers (original-vocab mode)
# ---------------------------------------------------------------------------

def _get_latent_pattern_ids(tokenizer):
    def _single_id(text):
        enc = tokenizer.encode(text, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None
    return {
        'latent_keyword_id': _single_id('latent'),
        'pipe_id': _single_id('|'),
        'vis_suffix_id': _single_id('-vis'),
    }


def _get_marker_component_ids(tokenizer):
    texts = ['<', '>', '|', '><', 'latent', 'start', 'end', '-lat', 'ent', '-vis']
    ids = set()
    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return ids


def _find_latent_keyword_positions(ids_list, latent_keyword_id, pipe_id):
    positions = []
    n = len(ids_list)
    for i in range(1, n - 1):
        if (ids_list[i] == latent_keyword_id
                and ids_list[i - 1] == pipe_id
                and ids_list[i + 1] == pipe_id):
            positions.append(i)
    return positions


def _find_visual_latent_keyword_positions(ids_list, latent_keyword_id, pipe_id, vis_suffix_id):
    if vis_suffix_id is None:
        return []
    positions = []
    n = len(ids_list)
    for i in range(1, n - 1):
        if (ids_list[i] == latent_keyword_id
                and ids_list[i - 1] == pipe_id
                and ids_list[i + 1] == vis_suffix_id):
            positions.append(i)
    return positions


def _find_text_latent_block_start(ids_list, pipe_id, vis_suffix_id, tokenizer):
    def _first_id(text):
        enc = tokenizer.encode(text, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None
    start_id = _first_id('start')
    neglat_id = _first_id('-lat')
    ent_id = _first_id('ent')
    if start_id is None or neglat_id is None or ent_id is None:
        return len(ids_list)
    n = len(ids_list)
    for i in range(1, n - 4):
        if (ids_list[i] == pipe_id
                and ids_list[i + 1] == start_id
                and ids_list[i + 2] == neglat_id
                and ids_list[i + 3] == ent_id
                and ids_list[i + 4] == pipe_id
                and ids_list[i - 1] != vis_suffix_id):
            return i
    return len(ids_list)


def _expand_keyword_positions_with_stop(ids_list, keyword_positions,
                                        marker_component_ids, stop_before):
    stop_set = set(stop_before)
    all_positions = []
    used = set()
    for kw_pos in keyword_positions:
        start = kw_pos
        while (start > 0
               and (start - 1) not in stop_set
               and ids_list[start - 1] in marker_component_ids
               and (start - 1) not in used):
            start -= 1
        end = kw_pos
        n = len(ids_list)
        while (end < n - 1
               and (end + 1) not in stop_set
               and ids_list[end + 1] in marker_component_ids
               and (end + 1) not in used):
            end += 1
        for p in range(start, end + 1):
            if p not in used:
                all_positions.append(p)
                used.add(p)
    return all_positions


def compute_inference_latent_positions(
    input_ids_single, tokenizer, pattern_ids, marker_component_ids,
):
    """Return (text_positions, visual_positions) for one sequence.

    Uses original-vocab pattern matching with all sub-tokens and separate
    visual / text latent blocks.
    """
    ids_list = (input_ids_single.tolist()
                if hasattr(input_ids_single, 'tolist') else input_ids_single)
    lkw = pattern_ids['latent_keyword_id']
    pipe = pattern_ids['pipe_id']
    vis_suffix_id = pattern_ids.get('vis_suffix_id')

    text_kw = _find_latent_keyword_positions(ids_list, lkw, pipe)
    vis_kw = (_find_visual_latent_keyword_positions(
        ids_list, lkw, pipe, vis_suffix_id) if vis_suffix_id else [])

    text_block_start = _find_text_latent_block_start(
        ids_list, pipe, vis_suffix_id, tokenizer)
    stop_txt = set(text_kw)
    vis_pos_full = (_expand_keyword_positions_with_stop(
        ids_list, vis_kw, marker_component_ids, stop_txt)
        if vis_kw else [])
    vis_pos = [p for p in vis_pos_full if p < text_block_start]
    text_pos = _expand_keyword_positions_with_stop(
        ids_list, text_kw, marker_component_ids, vis_pos)
    text_pos = [p for p in text_pos if p >= text_block_start]
    return text_pos, vis_pos


# ---------------------------------------------------------------------------
# Checkpoint loading utilities
# ---------------------------------------------------------------------------

def collect_state_dict_from_safetensors(ckpt_dir, prefix):
    result = {}
    for sf in sorted(glob.glob(os.path.join(ckpt_dir, '*.safetensors'))):
        sd = load_file(sf)
        for k, v in sd.items():
            if k.startswith(prefix):
                result[k[len(prefix):]] = v
    return result


def build_aux_decoder_from_checkpoint(ckpt_dir, prefix, aux_base_model_path,
                                      device, dtype):
    config = AutoConfig.from_pretrained(aux_base_model_path,
                                        trust_remote_code=True)
    model_type = getattr(config, 'model_type', '')
    if 'qwen3_vl' in model_type:
        from transformers import Qwen3VLForConditionalGeneration as Cls
    elif 'qwen2_vl' in model_type:
        from transformers import Qwen2VLForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls

    print(f"[INFO] Building aux decoder from {aux_base_model_path}")
    model = Cls.from_pretrained(aux_base_model_path, dtype=dtype,
                                trust_remote_code=True)

    sd = collect_state_dict_from_safetensors(ckpt_dir, prefix)
    if sd:
        embed_key = 'model.language_model.embed_tokens.weight'
        if embed_key in sd:
            ckpt_vocab = sd[embed_key].shape[0]
            cur_vocab = model.model.language_model.embed_tokens.weight.shape[0]
            if ckpt_vocab != cur_vocab:
                print(f"[INFO] Resizing aux decoder embeddings: "
                      f"{cur_vocab} -> {ckpt_vocab}")
                model.resize_token_embeddings(ckpt_vocab)

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            lm_head_missing = [k for k in missing if 'lm_head.weight' in k]
            if lm_head_missing and hasattr(model, 'lm_head') and hasattr(model, 'model'):
                model.lm_head.weight = model.model.language_model.embed_tokens.weight
            unresolved = [k for k in missing if k not in lm_head_missing]
            if unresolved:
                print(f"[WARN] Unresolved missing keys: {unresolved}")
        print(f"[INFO] Loaded {len(sd)} weights with prefix '{prefix}'")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")
    else:
        print(f"[WARN] No weights found with prefix '{prefix}' in {ckpt_dir}")

    model.to(device).eval()
    return model


def build_projection_from_checkpoint(ckpt_dir, prefix, in_dim, out_dim,
                                     device, dtype):
    proj = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.GELU(),
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
    )
    sd = collect_state_dict_from_safetensors(ckpt_dir, prefix)
    if sd:
        proj.load_state_dict(sd)
        print(f"[INFO] Loaded projection weights with prefix '{prefix}'")
    else:
        print(f"[WARN] No projection weights for prefix '{prefix}', random init")
    proj.to(device=device, dtype=dtype).eval()
    return proj


# ---------------------------------------------------------------------------
# Float MLP head (regress numeric waypoints from latent hidden states)
# ---------------------------------------------------------------------------

class FloatMLPHead(nn.Module):
    """MLP head that predicts float coordinates from a single hidden state."""

    def __init__(self, hidden_size, output_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, output_dim),
        )

    def forward(self, hidden_state):
        return self.net(hidden_state.to(self.net[0].weight.dtype))


def build_float_head_from_checkpoint(ckpt_dir, head_type, base_hidden,
                                     device, dtype, output_dim=24):
    """Build a float head (MLP only) and load weights from the checkpoint."""
    if head_type != 'mlp':
        raise ValueError(
            f"Only 'mlp' float head is supported in this script, got {head_type!r}")

    head = FloatMLPHead(base_hidden, output_dim)
    sd = collect_state_dict_from_safetensors(
        ckpt_dir, '_latent_cot_float_head.')
    if sd:
        missing, unexpected = head.load_state_dict(sd, strict=False)
        print(f"[INFO] Loaded float head ({head_type}) weights "
              f"({len(sd)} tensors)")
        if missing:
            print(f"[WARN] Float head missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Float head unexpected keys: {unexpected}")
    else:
        print("[WARN] No float head weights found with prefix "
              "'_latent_cot_float_head.', using random init")
    head.to(device=device, dtype=dtype).eval()
    return head


def find_float_head_position_infer(ids_list, tokenizer, latent_positions):
    """Locate the hidden-state position fed to the float head.

    Matches training: take the position immediately before the first
    ``answer`` token that appears after the last latent position.
    """
    answer_enc = tokenizer.encode('answer', add_special_tokens=False)
    if not answer_enc:
        return None
    answer_id = answer_enc[0]
    last_latent = max(latent_positions) if latent_positions else 0
    for i in range(last_latent + 1, len(ids_list)):
        if ids_list[i] == answer_id:
            return i - 1 if i > 0 else i
    return None


@torch.no_grad()
def predict_float_with_head(
    float_head, input_ids, hidden_states, tokenizer, text_positions_list,
    output_dim=24,
):
    """Run the MLP float head on the hidden state at the answer-anchor position.

    Returns a list (per-batch) of either ``None`` or a list of waypoints
    each of length 3 (x, y, heading).
    """
    last_hidden = hidden_states[-1]
    batch_size = input_ids.size(0)
    results = []
    for b in range(batch_size):
        positions = text_positions_list[b]
        ids_list = input_ids[b].tolist()
        pos = find_float_head_position_infer(ids_list, tokenizer, positions)
        if pos is None:
            results.append(None)
            continue
        hidden = last_hidden[b, pos, :].unsqueeze(0).float()
        pred = float_head(hidden)
        waypoints = pred.squeeze(0).reshape(-1, 3).tolist()
        results.append(waypoints)
    return results


# ---------------------------------------------------------------------------
# Aux-decoder helpers
# ---------------------------------------------------------------------------

def get_aux_input_embeddings(aux_decoder):
    if hasattr(aux_decoder, 'model') and hasattr(aux_decoder.model,
                                                   'get_input_embeddings'):
        return aux_decoder.model.get_input_embeddings()
    return aux_decoder.get_input_embeddings()


def call_aux_decoder_lm(aux_decoder, inputs_embeds, use_cache=False,
                         past_key_values=None):
    """Forward through the aux decoder's language_model + lm_head directly."""
    if (hasattr(aux_decoder, 'model')
            and hasattr(aux_decoder.model, 'language_model')
            and hasattr(aux_decoder, 'lm_head')):
        lm_out = aux_decoder.model.language_model(
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
        hidden = (lm_out.last_hidden_state
                  if hasattr(lm_out, 'last_hidden_state') else lm_out[0])
        logits = aux_decoder.lm_head(hidden)
        new_past = (lm_out.past_key_values
                    if use_cache and hasattr(lm_out, 'past_key_values')
                    else None)
        return logits, new_past
    out = aux_decoder(
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        past_key_values=past_key_values,
    )
    logits = out.logits if hasattr(out, 'logits') else out[0]
    new_past = (out.past_key_values
                if use_cache and hasattr(out, 'past_key_values') else None)
    return logits, new_past


def extract_visual_embeds(student_embeds, input_ids, image_token_id,
                          video_token_id=None):
    vis_mask = (input_ids == image_token_id)
    if video_token_id is not None:
        vis_mask = vis_mask | (input_ids == video_token_id)
    if not vis_mask.any():
        return None
    return student_embeds[vis_mask]


# ---------------------------------------------------------------------------
# Aux-decoder decoding functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_latent_with_aux(
    model, aux_decoder, latent_proj, input_ids, hidden_states,
    processor, device,
    text_positions_list=None,
    use_visual_condition=False, image_token_id=None, video_token_id=None,
    max_explain_tokens=512,
    vit_embeds=None,
):
    tokenizer = (processor.tokenizer
                 if hasattr(processor, 'tokenizer') else processor)
    latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
    last_hidden = hidden_states[-1]
    batch_size = input_ids.size(0)
    results = []
    aux_embedding = get_aux_input_embeddings(aux_decoder)

    for b in range(batch_size):
        if text_positions_list is not None:
            positions = text_positions_list[b]
        else:
            positions = (input_ids[b] == latent_token_id).nonzero(
                as_tuple=True)[0].tolist()
        if not positions:
            results.append("")
            continue

        latent_embeds = last_hidden[b, positions, :]
        if latent_proj is not None:
            latent_embeds = latent_proj(latent_embeds)

        parts = []
        if use_visual_condition and image_token_id is not None:
            if vit_embeds is not None:
                student_embeds_b = vit_embeds[b]
            else:
                embed_fn = (model.model.get_input_embeddings()
                            if hasattr(model, 'model')
                            else model.get_input_embeddings())
                student_embeds_b = embed_fn(input_ids[b])
            vit_cond = extract_visual_embeds(
                student_embeds_b, input_ids[b], image_token_id, video_token_id)
            if vit_cond is not None:
                parts.append(vit_cond)
        parts.append(latent_embeds)
        combined = torch.cat(parts, dim=0).unsqueeze(0)

        generated_ids = []
        past_kv = None
        cur_embeds = combined
        for _ in range(max_explain_tokens):
            logits, past_kv = call_aux_decoder_lm(
                aux_decoder, cur_embeds, use_cache=True,
                past_key_values=past_kv)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())
            if next_id.item() == tokenizer.eos_token_id:
                break
            cur_embeds = aux_embedding(next_id).unsqueeze(1)

        results.append(
            tokenizer.decode(generated_ids, skip_special_tokens=True))

    return results


@torch.no_grad()
def decode_latent_with_visual_aux(
    model, visual_aux_decoder, visual_latent_proj, input_ids, hidden_states,
    processor, device,
    visual_positions_list=None,
    use_visual_condition=False, image_token_id=None, video_token_id=None,
    max_visual_tokens=512,
    vit_embeds=None,
    vis_aux_tokenizer=None,
):
    tokenizer = (processor.tokenizer
                 if hasattr(processor, 'tokenizer') else processor)
    last_hidden = hidden_states[-1]

    if vis_aux_tokenizer is None:
        vis_aux_tokenizer = tokenizer

    batch_size = input_ids.size(0)
    results = []
    aux_embedding = get_aux_input_embeddings(visual_aux_decoder)

    for b in range(batch_size):
        if visual_positions_list is not None:
            positions = visual_positions_list[b]
        else:
            latent_token_id = tokenizer.convert_tokens_to_ids('<|latent|>')
            positions = (input_ids[b] == latent_token_id).nonzero(
                as_tuple=True)[0].tolist()
        if not positions:
            results.append("")
            continue

        latent_embeds = last_hidden[b, positions, :]
        if visual_latent_proj is not None:
            latent_embeds = visual_latent_proj(latent_embeds)

        parts = []
        if use_visual_condition and image_token_id is not None:
            if vit_embeds is not None:
                student_embeds_b = vit_embeds[b]
            else:
                embed_fn = (model.model.get_input_embeddings()
                            if hasattr(model, 'model')
                            else model.get_input_embeddings())
                student_embeds_b = embed_fn(input_ids[b])
            vit_cond = extract_visual_embeds(
                student_embeds_b, input_ids[b], image_token_id, video_token_id)
            if vit_cond is not None:
                parts.append(vit_cond)
        parts.append(latent_embeds)
        combined = torch.cat(parts, dim=0).unsqueeze(0)

        generated_ids = []
        past_kv = None
        cur_embeds = combined
        eos_id = vis_aux_tokenizer.eos_token_id
        for _ in range(max_visual_tokens):
            logits, past_kv = call_aux_decoder_lm(
                visual_aux_decoder, cur_embeds, use_cache=True,
                past_key_values=past_kv)
            next_id = logits[:, -1, :].argmax(dim=-1)
            generated_ids.append(next_id.item())
            if next_id.item() == eos_id:
                break
            cur_embeds = aux_embedding(next_id).unsqueeze(1)

        results.append(
            vis_aux_tokenizer.decode(generated_ids, skip_special_tokens=True))

    return results


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_set(path):
    """Load a test set from JSON (list) or JSONL (one dict per line)."""
    if path.endswith('.jsonl'):
        data = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    with open(path, 'r') as f:
        return json.load(f)


def resolve_image_path(rel_path, base_path):
    """Prepend base_path to a relative image path if base_path is given."""
    if base_path and not os.path.isabs(rel_path):
        return os.path.join(base_path, rel_path)
    return rel_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OneVL (Latent CoT) inference for Qwen3-VL")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained OneVL checkpoint")
    parser.add_argument("--test_set_path", type=str, required=True,
                        help="Path to the test set (JSON or JSONL)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save inference results (JSON)")
    parser.add_argument("--image_base_path", type=str, default="",
                        help="Base directory prepended to relative image paths")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_latent", type=int, default=2,
                        help="Number of text latent tokens in prefix")
    parser.add_argument("--num_latent_vis", type=int, default=4,
                        help="Number of visual latent tokens in prefix")
    parser.add_argument("--max_new_tokens", type=int, default=1024)

    # Aux text decoder
    parser.add_argument("--decoder_explain", action="store_true")
    parser.add_argument("--aux_model_path", type=str, default=None,
                        help="Architecture source for text aux decoder "
                             "(defaults to --model_path)")
    parser.add_argument("--aux_visual_condition", action="store_true")
    parser.add_argument("--c_thought", type=int, default=2)
    parser.add_argument("--max_explain_tokens", type=int, default=512)

    # Aux visual decoder
    parser.add_argument("--visual_decoder_explain", action="store_true")
    parser.add_argument("--visual_aux_model_path", type=str, default=None,
                        help="Architecture source for visual aux decoder "
                             "(defaults to --model_path)")
    parser.add_argument("--visual_aux_tokenizer_path", type=str, default=None,
                        help="Path to visual aux tokenizer with visual-token "
                             "vocab (defaults to visual_tokenizer/ next to "
                             "this script)")
    parser.add_argument("--visual_aux_visual_condition", action="store_true")
    parser.add_argument("--c_thought_visual", type=int, default=4)
    parser.add_argument("--max_visual_tokens", type=int, default=1024)

    parser.add_argument("--answer_prefix", type=str, default="[[",
                        help="Token(s) after <answer> in assistant prefix "
                             "(e.g. '[[' for nested lists, '[' for navsim)")
    parser.add_argument("--prefix_k", type=int, default=0,
                        help="Prefill first K GT waypoints after <answer>")

    # Float head (MLP only)
    parser.add_argument("--float_head_type", type=str, default=None,
                        choices=['mlp'],
                        help="If set, also run the MLP float head to regress "
                             "waypoints from the latent hidden state. "
                             "(flow-matching head is intentionally unsupported)")
    parser.add_argument("--float_head_output_dim", type=int, default=24,
                        help="Float head output dim (default 24 = 8 waypoints * 3)")

    args = parser.parse_args()
    device = args.device
    dtype = torch.bfloat16

    # ---- Load model ----
    print(f"[INFO] Loading model from {args.model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, trust_remote_code=True)
    model.to(device).eval()

    processor = AutoProcessor.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    MAX_IMAGE_SIZE = 1792
    processor.image_processor.max_pixels = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE
    processor.image_processor.size["longest_edge"] = MAX_IMAGE_SIZE * MAX_IMAGE_SIZE

    image_token_id = getattr(model.config, 'image_token_id', None)
    video_token_id = getattr(model.config, 'video_token_id', None)

    tokenizer = (processor.tokenizer
                 if hasattr(processor, 'tokenizer') else processor)
    pattern_ids = _get_latent_pattern_ids(tokenizer)
    marker_component_ids = _get_marker_component_ids(tokenizer)

    # ---- Hidden sizes ----
    base_hidden = (model.config.text_config.hidden_size
                   if hasattr(model.config, 'text_config')
                   else model.config.hidden_size)

    # ---- Build latent prefix string ----
    ans_pfx = args.answer_prefix
    if args.num_latent_vis > 0:
        latent_block = (
            "<|start-latent-vis|>"
            + "<|latent-vis|>" * args.num_latent_vis
            + "<|end-latent-vis|><|start-latent|>"
            + "<|latent|>" * args.num_latent
            + f"<|end-latent|><answer>{ans_pfx}"
        )
    else:
        latent_block = (
            "<|start-latent|>"
            + "<|latent|>" * args.num_latent
            + f"<|end-latent|><answer>{ans_pfx}"
        )
    assistant_prefix = latent_block

    # ---- Build aux decoders from checkpoint ----
    aux_decoder = None
    latent_proj = None
    if args.decoder_explain:
        aux_arch_path = args.aux_model_path or args.model_path
        print(f"[INFO] Text aux decoder architecture from: {aux_arch_path}")
        aux_decoder = build_aux_decoder_from_checkpoint(
            args.model_path, '_latent_cot_aux_decoder.',
            aux_arch_path, device, dtype)
        aux_cfg = AutoConfig.from_pretrained(aux_arch_path,
                                             trust_remote_code=True)
        aux_hidden = (aux_cfg.text_config.hidden_size
                      if hasattr(aux_cfg, 'text_config')
                      else aux_cfg.hidden_size)
        latent_proj = build_projection_from_checkpoint(
            args.model_path, '_latent_cot_latent_proj.',
            base_hidden, aux_hidden, device, dtype)

    visual_aux_decoder = None
    visual_latent_proj = None
    vis_aux_tokenizer = None
    if args.visual_decoder_explain:
        vis_arch_path = args.visual_aux_model_path or args.model_path
        print(f"[INFO] Visual aux decoder architecture from: {vis_arch_path}")
        visual_aux_decoder = build_aux_decoder_from_checkpoint(
            args.model_path, '_latent_cot_visual_aux_decoder.',
            vis_arch_path, device, dtype)
        vis_cfg = AutoConfig.from_pretrained(vis_arch_path,
                                             trust_remote_code=True)
        vis_hidden = (vis_cfg.text_config.hidden_size
                      if hasattr(vis_cfg, 'text_config')
                      else vis_cfg.hidden_size)
        visual_latent_proj = build_projection_from_checkpoint(
            args.model_path, '_latent_cot_visual_latent_proj.',
            base_hidden, vis_hidden, device, dtype)

        # Load the visual aux tokenizer (has 131k visual token vocab)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vis_tok_path = (args.visual_aux_tokenizer_path
                        or os.path.join(script_dir, 'visual_tokenizer'))
        try:
            from transformers import AutoTokenizer
            vis_aux_tokenizer = AutoTokenizer.from_pretrained(
                vis_tok_path, trust_remote_code=True)
            print(f"[INFO] Loaded visual aux tokenizer from {vis_tok_path} "
                  f"(vocab_size={vis_aux_tokenizer.vocab_size})")
        except Exception as e:
            print(f"[WARN] Could not load visual aux tokenizer from "
                  f"{vis_tok_path}: {e}")
            print("[WARN] Falling back to main model tokenizer for visual "
                  "aux decoding")

    # ---- Build float head (MLP) from checkpoint ----
    float_head = None
    if args.float_head_type is not None:
        float_head = build_float_head_from_checkpoint(
            args.model_path, args.float_head_type, base_hidden, device, dtype,
            output_dim=args.float_head_output_dim)

    # ---- Load test set ----
    test_set = load_test_set(args.test_set_path)
    print(f"[INFO] Loaded {len(test_set)} samples from {args.test_set_path}")

    # ---- Inference loop ----
    output_list = []
    need_hidden = (aux_decoder is not None
                   or visual_aux_decoder is not None
                   or float_head is not None)

    for idx, item in enumerate(test_set):
        output_dict = {}

        prompt = item["messages"][0]["content"].replace("<image>", "")
        image_paths_raw = item["images"]
        image_paths = [resolve_image_path(p, args.image_base_path)
                       for p in image_paths_raw]

        messages = [{"role": "user", "content": []}]
        for img_path in image_paths:
            messages[0]["content"].append({"type": "image", "image": img_path})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        text += assistant_prefix

        if args.prefix_k > 0:
            gt_src = item.get("GT")
            if not gt_src and len(item.get("messages", [])) > 1:
                gt_src = item["messages"][1].get("content", "")
            pts = parse_gt_waypoints(gt_src)
            if pts:
                k = min(args.prefix_k, len(pts))
                text += format_gt_prefix_points(pts[:k])

        try:
            imgs = [Image.open(p).convert("RGB") for p in image_paths]
        except Exception as e:
            print(f"[WARN] Skipping sample {idx}: {e}")
            continue

        inputs = processor(
            text=[text], images=imgs, return_tensors="pt", padding=True
        ).to(device)

        # ---- Optional: forward pass for hidden states (aux decoders) ----
        vit_embeds = None
        if need_hidden:
            _captured = {}

            def _capture_hook(module, args, kwargs):
                ie = kwargs.get('inputs_embeds')
                if ie is not None:
                    _captured['embeds'] = ie.detach()
                return None

            _hook = model.model.language_model.register_forward_pre_hook(
                _capture_hook, with_kwargs=True)
            torch.cuda.synchronize()
            _fwd_t0 = time.time()
            fwd_out = model(
                **inputs, output_hidden_states=True, return_dict=True)
            torch.cuda.synchronize()
            output_dict["forward_latency"] = time.time() - _fwd_t0
            _hook.remove()
            hidden_states = fwd_out.hidden_states
            vit_embeds = _captured.get('embeds')

            bs = inputs['input_ids'].size(0)
            text_positions_list = []
            visual_positions_list = []
            for b in range(bs):
                tp, vp = compute_inference_latent_positions(
                    inputs['input_ids'][b], tokenizer,
                    pattern_ids, marker_component_ids,
                )
                text_positions_list.append(tp)
                visual_positions_list.append(vp)

            if aux_decoder is not None:
                explains = decode_latent_with_aux(
                    model, aux_decoder, latent_proj,
                    inputs['input_ids'], hidden_states, processor, device,
                    text_positions_list=text_positions_list,
                    use_visual_condition=args.aux_visual_condition,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    max_explain_tokens=args.max_explain_tokens,
                    vit_embeds=vit_embeds,
                )
                if explains and explains[0]:
                    output_dict["decoder_explain"] = explains[0]

            if visual_aux_decoder is not None:
                vis_explains = decode_latent_with_visual_aux(
                    model, visual_aux_decoder, visual_latent_proj,
                    inputs['input_ids'], hidden_states, processor, device,
                    visual_positions_list=visual_positions_list,
                    use_visual_condition=args.visual_aux_visual_condition,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    max_visual_tokens=args.max_visual_tokens,
                    vit_embeds=vit_embeds,
                    vis_aux_tokenizer=vis_aux_tokenizer,
                )
                if vis_explains and vis_explains[0]:
                    output_dict["visual_decoder_explain"] = vis_explains[0]

            if float_head is not None:
                torch.cuda.synchronize()
                _fh_t0 = time.time()
                float_preds = predict_float_with_head(
                    float_head, inputs['input_ids'], hidden_states,
                    tokenizer, text_positions_list,
                    output_dim=args.float_head_output_dim,
                )
                torch.cuda.synchronize()
                output_dict["float_head_latency"] = time.time() - _fh_t0
                if float_preds and float_preds[0] is not None:
                    output_dict["float_pred"] = float_preds[0]
                    if idx < 3:
                        print(f"  [FloatHead] waypoints: {float_preds[0]}")
                        print(f"  [FloatHead] latency: "
                              f"{output_dict['float_head_latency']:.4f}s")

            del fwd_out, hidden_states
            torch.cuda.empty_cache()

        # ---- Generate answer ----
        torch.cuda.synchronize()
        t0 = time.time()

        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        torch.cuda.synchronize()
        latency = time.time() - t0

        generated_ids = gen_outputs.sequences
        generated_ids_trimmed = [
            out[len(inp):] for inp, out
            in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        output_dict["latency"] = latency
        output_dict["messages"] = messages
        output_dict["GT"] = item.get("GT", "")
        output_dict["output_text"] = output_text[0]

        scores = gen_outputs.scores
        entropies = []
        for step_logits in scores:
            probs = F.softmax(step_logits.float(), dim=-1)
            entropies.append(
                -torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
        entropies_tensor = torch.stack(entropies).transpose(0, 1)
        avg_entropy = entropies_tensor.mean(dim=1)

        transition_scores = model.compute_transition_scores(
            generated_ids, scores, normalize_logits=True)
        avg_log_prob = transition_scores.mean(dim=1)
        seq_confidence = torch.exp(avg_log_prob)

        output_dict["avg_entropy"] = avg_entropy.item()
        output_dict["avg_log_prob"] = avg_log_prob.item()
        output_dict["seq_confidence"] = seq_confidence.item()

        if idx < 3 or idx % 100 == 0:
            print(f"\n=== Sample {idx} ===")
            print(f"  Output: {output_text[0][:200]}")
            print(f"  Entropy: {avg_entropy.item():.4f}, "
                  f"Confidence: {seq_confidence.item():.2%}")
            if output_dict.get("decoder_explain"):
                print(f"  Explain: {output_dict['decoder_explain'][:200]}")
            if output_dict.get("visual_decoder_explain"):
                print(f"  VisExplain: "
                      f"{output_dict['visual_decoder_explain'][:200]}")
            if output_dict.get("float_pred") is not None:
                print(f"  FloatPred: {output_dict['float_pred']}")

        output_list.append(output_dict)

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            json.dump(output_list, f, indent=4, ensure_ascii=False)

    print(f"\n[INFO] Done. {len(output_list)} results -> {args.output_path}")


if __name__ == "__main__":
    main()
