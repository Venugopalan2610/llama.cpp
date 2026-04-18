#!/usr/bin/env python3
"""Convert QTIP-quantized OLMoE .pt files + HF model to GGUF.

Usage:
    python convert_qtip_olmoe_to_gguf.py \
        --hf-model cache/model/olmoe-1b-7b-0125 \
        --quant-dir cache/quantized \
        --output olmoe-qtip-2b.gguf

Reads:
  - Non-quantized weights (embeddings, norms, router) from HF safetensors
  - Quantized .pt files from quant-dir (bitstreams, start_states, sign_l, sign_r, W_scale)

Tile layout convention:
  Each QTIP block = one 16x16 tile (256 elements) from the weight matrix.
  Blocks stored in row_tile-major order: tile (rt, ct) at flat index rt * n_col_tiles + ct.
  The 256 decoded values within a tile are in row-major 16x16 order.
  sign_l and sign_r stored as separate F32 tensors.
"""

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

TILE_SIZE = 16
QTIP_BLOCK_SIZE = 256
QTIP_TYPE_ID = 41
BLOCK_BYTES = 68  # sizeof(block_qtip_2b)


def float32_to_fp16_bits(f: float) -> int:
    arr = np.array([f], dtype=np.float32).astype(np.float16)
    return int(np.frombuffer(arr.tobytes(), dtype=np.uint16)[0])


def pack_tiles_vectorized(bitstreams: np.ndarray, start_states: np.ndarray,
                          scale_f16: int, m: int, n: int) -> bytes:
    """Pack all tiles for a weight matrix into QTIP block format.

    Args:
        bitstreams: (n_col_tiles, n_row_tiles, 128) uint8
        start_states: (n_col_tiles, n_row_tiles) int32
        scale_f16: fp16-encoded scale (uint16)
        m: output dim, n: input dim

    Returns:
        bytes of packed blocks in row_tile-major order.
    """
    n_col_tiles = n // TILE_SIZE
    n_row_tiles = m // TILE_SIZE
    total_blocks = n_row_tiles * n_col_tiles

    buf = bytearray(total_blocks * BLOCK_BYTES)

    for rt in range(n_row_tiles):
        for ct in range(n_col_tiles):
            flat_idx = rt * n_col_tiles + ct
            codes = bitstreams[ct, rt, :]  # .pt is (col_tile, row_tile, step)
            ss = int(start_states[ct, rt])

            # Pre-apply first code to initial state
            state0 = ((ss << 4) | int(codes[0])) & 0xFFFF

            off = flat_idx * BLOCK_BYTES
            # d: fp16 scale (2 bytes LE)
            buf[off] = scale_f16 & 0xFF
            buf[off + 1] = (scale_f16 >> 8) & 0xFF
            # trellis_data[0..1]: initial state (2 bytes LE)
            buf[off + 2] = state0 & 0xFF
            buf[off + 3] = (state0 >> 8) & 0xFF
            # trellis_data[2..65]: 127 x 4-bit codes, bit-packed
            for j1 in range(127):  # codes[1] to codes[127]
                code = int(codes[j1 + 1]) & 0xF
                bit_pos = 16 + j1 * 4
                byte_idx = bit_pos // 8
                bit_off = bit_pos % 8
                buf_pos = off + 2 + byte_idx
                buf[buf_pos] |= code << bit_off
                if bit_off > 4 and (buf_pos + 1) < off + BLOCK_BYTES:
                    buf[buf_pos + 1] |= code >> (8 - bit_off)

    return bytes(buf)


def convert_qtip_weight(pt_path: str) -> tuple:
    """Convert one .pt quantized weight to packed QTIP blocks + sign vectors.

    Returns (block_data, sign_l, sign_r, shape) where shape = (m, n).
    """
    saved = torch.load(pt_path, map_location='cpu', weights_only=False)

    bitstreams = saved['bitstreams']       # (n_col_tiles, n_row_tiles, 128) uint8
    start_states = saved['start_states']   # (n_col_tiles, n_row_tiles) int32
    sign_l = saved['sign_l'].astype(np.float32)
    sign_r = saved['sign_r'].astype(np.float32)
    W_scale = float(saved['W_scale'])
    m, n = saved['shape']

    n_col_tiles = n // TILE_SIZE
    n_row_tiles = m // TILE_SIZE
    assert bitstreams.shape == (n_col_tiles, n_row_tiles, 128)

    scale_f16 = float32_to_fp16_bits(W_scale)
    block_data = pack_tiles_vectorized(bitstreams, start_states, scale_f16, m, n)
    return block_data, sign_l, sign_r, (m, n)


def convert_expert_stack(quant_dir: str, layer_idx: int, proj: str,
                         n_experts: int) -> tuple:
    """Convert all experts for one projection into stacked data.

    Returns (stacked_blocks, sign_l_stacked, sign_r_stacked, shape_3d).
    shape_3d = (ne0, ne1, ne2) in ggml order = (input_dim, output_dim, n_expert).
    """
    all_blocks = []
    all_sign_l = []
    all_sign_r = []
    ref_shape = None

    for eidx in range(n_experts):
        fname = f"expert_{eidx:02d}_{proj}.pt"
        pt_path = os.path.join(quant_dir, f"L{layer_idx:02d}", fname)

        block_data, sign_l, sign_r, shape = convert_qtip_weight(pt_path)
        if ref_shape is None:
            ref_shape = shape
        else:
            assert shape == ref_shape

        all_blocks.append(block_data)
        all_sign_l.append(sign_l)
        all_sign_r.append(sign_r)

    m, n = ref_shape
    stacked_blocks = b''.join(all_blocks)
    sign_l_stacked = np.stack(all_sign_l, axis=0)   # (n_expert, m) - ggml {m, n_expert}
    sign_r_stacked = np.stack(all_sign_r, axis=0)   # (n_expert, n) - ggml {n, n_expert}
    return stacked_blocks, sign_l_stacked, sign_r_stacked, (n, m, n_experts)


def load_hf_tensors(hf_model_dir: str) -> dict:
    tensors = {}
    for sf_file in sorted(Path(hf_model_dir).glob("*.safetensors")):
        with safe_open(str(sf_file), framework="numpy") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    return tensors


# Tensor name mappings
HF_GLOBAL = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}

HF_LAYER = {
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
}

ATTN_PROJS = {
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_output",
}

EXPERT_PROJS = {
    "gate_proj": "ffn_gate_exps",
    "up_proj": "ffn_up_exps",
    "down_proj": "ffn_down_exps",
}


def add_qtip_tensor(writer, name: str, block_data: bytes,
                    m: int, n: int, n_expert: int = 0):
    """Add a QTIP_2B tensor to the GGUF writer.

    m = output_dim, n = input_dim.
    ggml shape: ne[0]=n, ne[1]=m, [ne[2]=n_expert].

    The byte array is reshaped so its last dim = (n/256)*68, which is a
    multiple of type_size (68). quant_shape_from_byte_shape then computes
    the tensor element shape automatically.
    """
    QTIP_2B = gguf.GGMLQuantizationType(QTIP_TYPE_ID)
    arr = np.frombuffer(block_data, dtype=np.uint8)

    bytes_per_row = (n // QTIP_BLOCK_SIZE) * BLOCK_BYTES
    if n_expert > 0:
        arr = arr.reshape(n_expert, m, bytes_per_row)
    else:
        arr = arr.reshape(m, bytes_per_row)

    writer.add_tensor(name, arr, raw_dtype=QTIP_2B)


def main():
    parser = argparse.ArgumentParser(description="Convert QTIP-quantized OLMoE to GGUF")
    parser.add_argument("--hf-model", required=True, help="Path to HF model directory")
    parser.add_argument("--quant-dir", required=True, help="Path to quantized .pt files")
    parser.add_argument("--output", required=True, help="Output GGUF file path")
    args = parser.parse_args()

    print(f"Loading HF model from {args.hf_model}")
    hf_tensors = load_hf_tensors(args.hf_model)
    print(f"  Loaded {len(hf_tensors)} tensors")

    with open(os.path.join(args.hf_model, "config.json")) as f:
        config = json.load(f)

    n_layers = config["num_hidden_layers"]
    n_experts = config["num_experts"]
    n_experts_per_tok = config["num_experts_per_tok"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    vocab_size = config["vocab_size"]
    rms_norm_eps = config.get("rms_norm_eps", 1e-5)
    rope_theta = config.get("rope_theta", 10000.0)
    context_length = config.get("max_position_embeddings", 4096)

    # Determine actual tokenizer vocab size (may differ from config due to padding)
    tokenizer_path = os.path.join(args.hf_model, "tokenizer.json")
    n_vocab_actual = vocab_size
    tok = None
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path) as f:
            tok = json.load(f)
        if "model" in tok and "vocab" in tok["model"]:
            n_vocab_actual = len(tok["model"]["vocab"])
            if n_vocab_actual != vocab_size:
                print(f"  Note: trimming embeddings from {vocab_size} to {n_vocab_actual} "
                      f"(tokenizer vocab size)")

    writer = gguf.GGUFWriter(args.output, "olmoe")

    # Metadata
    writer.add_name("OLMoE-1B-7B QTIP-2bit")
    writer.add_block_count(n_layers)
    writer.add_context_length(context_length)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_expert_count(n_experts)
    writer.add_expert_used_count(n_experts_per_tok)
    writer.add_vocab_size(n_vocab_actual)
    writer.add_expert_weights_scale(1.0)

    # Tokenizer
    if tok is not None and "model" in tok and "vocab" in tok["model"]:
        vocab_dict = tok["model"]["vocab"]
        tokens = sorted(vocab_dict.items(), key=lambda x: x[1])
        token_list = [t[0].encode("utf-8") for t in tokens]
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(token_list)
        if "merges" in tok["model"] and tok["model"]["merges"]:
            raw_merges = tok["model"]["merges"]
            # Merges can be strings ("Ġ Ġ") or lists (["Ġ", "Ġ"])
            if raw_merges and isinstance(raw_merges[0], list):
                merges = [" ".join(m).encode("utf-8") for m in raw_merges]
            else:
                merges = [m.encode("utf-8") for m in raw_merges]
            writer.add_token_merges(merges)

    # Global non-quantized tensors
    print("\n=== Non-quantized tensors ===")
    for hf_name, gguf_name in HF_GLOBAL.items():
        if hf_name in hf_tensors:
            data = hf_tensors[hf_name].astype(np.float32)
            # Trim embedding/output to actual tokenizer vocab size
            if hf_name in ("model.embed_tokens.weight", "lm_head.weight"):
                if data.shape[0] > n_vocab_actual:
                    data = data[:n_vocab_actual]
            print(f"  {gguf_name}: {data.shape}")
            writer.add_tensor(gguf_name, data)

    # Per-layer non-quantized tensors
    for il in range(n_layers):
        for hf_suffix, gguf_suffix in HF_LAYER.items():
            hf_name = f"model.layers.{il}.{hf_suffix}"
            gguf_name = f"blk.{il}.{gguf_suffix}"
            if hf_name in hf_tensors:
                data = hf_tensors[hf_name].astype(np.float32)
                writer.add_tensor(gguf_name, data)

    # Quantized attention weights
    print("\n=== Quantized attention weights ===")
    t0 = time.time()
    for il in range(n_layers):
        for py_proj, gguf_prefix in ATTN_PROJS.items():
            pt_path = os.path.join(args.quant_dir, f"L{il:02d}", f"attn_{py_proj}.pt")
            if not os.path.exists(pt_path):
                # Fallback: use HF weight as F32
                hf_name = f"model.layers.{il}.self_attn.{py_proj}.weight"
                if hf_name in hf_tensors:
                    data = hf_tensors[hf_name].astype(np.float32)
                    writer.add_tensor(f"blk.{il}.{gguf_prefix}.weight", data)
                continue

            block_data, sign_l, sign_r, (m, n) = convert_qtip_weight(pt_path)

            tname = f"blk.{il}.{gguf_prefix}.weight"
            add_qtip_tensor(writer, tname, block_data, m, n)

            # sign_r: (n,) F32, sign_l: (m,) F32
            writer.add_tensor(f"blk.{il}.{gguf_prefix}.sign_r.weight", sign_r)
            writer.add_tensor(f"blk.{il}.{gguf_prefix}.sign_l.weight", sign_l)

        if (il + 1) % 4 == 0:
            print(f"  Layers 0-{il} done ({time.time()-t0:.0f}s)")

    # Quantized expert weights
    print("\n=== Quantized expert weights ===")
    for il in range(n_layers):
        lt0 = time.time()
        for py_proj, gguf_prefix in EXPERT_PROJS.items():
            stacked, sign_l_st, sign_r_st, (ne0, ne1, ne2) = convert_expert_stack(
                args.quant_dir, il, py_proj, n_experts)

            tname = f"blk.{il}.{gguf_prefix}.weight"
            # ne0=input_dim, ne1=output_dim, ne2=n_expert
            add_qtip_tensor(writer, tname, stacked, m=ne1, n=ne0, n_expert=ne2)

            # sign_r: (n_expert, input_dim) F32 → ggml {input_dim, n_expert}
            writer.add_tensor(f"blk.{il}.{gguf_prefix}.sign_r.weight", sign_r_st)
            # sign_l: (n_expert, output_dim) F32 → ggml {output_dim, n_expert}
            writer.add_tensor(f"blk.{il}.{gguf_prefix}.sign_l.weight", sign_l_st)

        elapsed = time.time() - lt0
        print(f"  Layer {il}: {elapsed:.0f}s ({time.time()-t0:.0f}s total)")

    # Write
    print(f"\nWriting GGUF to {args.output}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(args.output)
    print(f"Done! {args.output} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
