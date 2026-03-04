"""
Goal 4: Architectural Bottleneck Analysis
Maps benchmark measurements to architectural causes, computes arithmetic
intensity, and classifies components as compute-bound vs memory-bound.

Usage:
    python -m analysis.bottleneck_analysis [--data-dir DATA_DIR]
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

from benchmarks.utils import get_output_prefix, get_platform_tag

import numpy as np

from benchmarks.config import DATA_DIR, RESULTS_DIR


def find_latest(data_dir: str, pattern: str) -> Optional[str]:
    """Find the most recently modified file matching a glob pattern in data_dir."""
    matches = glob.glob(os.path.join(data_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


# ============================================================
# LLaMA Architecture Parameters
# ============================================================

# LLaMA 3.2 1B architecture (from HuggingFace config.json)
LLAMA_1B = {
    "name": "LLaMA-3.2-1B",
    "params_billions": 1.0,
    "hidden_dim": 2048,
    "num_layers": 16,
    "num_heads": 32,
    "num_kv_heads": 8,        # GQA: 8 key-value heads
    "head_dim": 64,
    "intermediate_dim": 8192,  # MLP hidden size
    "vocab_size": 128256,
}

# LLaMA 3.2 3B architecture (from HuggingFace config.json)
LLAMA_3B = {
    "name": "LLaMA-3.2-3B",
    "params_billions": 3.0,
    "hidden_dim": 3072,
    "num_layers": 28,
    "num_heads": 24,
    "num_kv_heads": 8,        # GQA: 8 key-value heads
    "head_dim": 128,
    "intermediate_dim": 8192,
    "vocab_size": 128256,
}

# Bytes per element for each quantization
BYTES_PER_ELEMENT = {
    "F16": 2.0,
    "Q8_0": 1.0,
    "Q4_K_M": 0.5,
}


# ============================================================
# Arithmetic Intensity Computation
# ============================================================

def compute_attention_flops(seq_len: int, arch: Dict) -> float:
    """Compute FLOPs for a single attention layer (decode step, batch=1)."""
    d = arch["hidden_dim"]
    h = arch["num_heads"]
    d_k = arch["head_dim"]
    kv_dim = arch["num_kv_heads"] * d_k  # GQA: K,V project to smaller dim

    # QKV projections with GQA:
    #   Q: [1, d] x [d, d] = 2*d*d  (full hidden_dim)
    #   K: [1, d] x [d, kv_dim] = 2*d*kv_dim
    #   V: [1, d] x [d, kv_dim] = 2*d*kv_dim
    qkv_flops = 2 * d * d + 2 * 2 * d * kv_dim

    # Attention scores: [h, 1, d_k] x [h, d_k, seq_len] = h * 2 * d_k * seq_len
    score_flops = h * 2 * d_k * seq_len

    # Attention output: [h, 1, seq_len] x [h, seq_len, d_k] = h * 2 * seq_len * d_k
    out_flops = h * 2 * seq_len * d_k

    # Softmax: exp + sum + divide over seq_len per head = h * 3 * seq_len
    softmax_flops = h * 3 * seq_len

    # Output projection: [1, d] x [d, d] = 2 * d * d
    proj_flops = 2 * d * d

    return qkv_flops + score_flops + softmax_flops + out_flops + proj_flops


def compute_attention_bytes(seq_len: int, arch: Dict, quant: str = "F16") -> float:
    """Compute bytes accessed for attention (decode step, batch=1)."""
    d = arch["hidden_dim"]
    d_k = arch["head_dim"]
    kv_dim = arch["num_kv_heads"] * d_k  # GQA: K,V use fewer heads
    bpe = BYTES_PER_ELEMENT[quant]

    # QKV weight matrices with GQA: Q is [d, d], K and V are [d, kv_dim]
    qkv_bytes = (d * d + 2 * d * kv_dim) * bpe

    # KV-cache read: 2 * seq_len * kv_dim (always in FP16 regardless of weight quant)
    kv_bytes = 2 * seq_len * kv_dim * 2.0  # KV-cache stored in FP16

    # Output projection weights: d * d
    proj_bytes = d * d * bpe

    return qkv_bytes + kv_bytes + proj_bytes


def compute_mlp_flops(arch: Dict) -> float:
    """Compute FLOPs for a single MLP layer (decode step, batch=1)."""
    d = arch["hidden_dim"]
    ff = arch["intermediate_dim"]

    # gate_proj: [1, d] x [d, ff] = 2 * d * ff
    # up_proj:   [1, d] x [d, ff] = 2 * d * ff
    # down_proj: [1, ff] x [ff, d] = 2 * ff * d
    # SiLU + element-wise multiply: ~2 * ff
    return 3 * 2 * d * ff + 2 * ff


def compute_mlp_bytes(arch: Dict, quant: str = "F16") -> float:
    """Compute bytes accessed for MLP (decode step, batch=1)."""
    d = arch["hidden_dim"]
    ff = arch["intermediate_dim"]
    bpe = BYTES_PER_ELEMENT[quant]

    # gate_proj + up_proj + down_proj weights
    return (2 * d * ff + ff * d) * bpe


def compute_lm_head_flops(arch: Dict) -> float:
    """Compute FLOPs for LM head (vocab projection)."""
    d = arch["hidden_dim"]
    v = arch["vocab_size"]
    return 2 * d * v


def compute_lm_head_bytes(arch: Dict, quant: str = "F16") -> float:
    """Compute bytes accessed for LM head."""
    d = arch["hidden_dim"]
    v = arch["vocab_size"]
    bpe = BYTES_PER_ELEMENT[quant]
    return d * v * bpe


# ============================================================
# Full Model Analysis
# ============================================================

def analyze_arithmetic_intensity(
    arch: Dict,
    seq_len: int = 512,
    quant: str = "F16",
) -> Dict[str, Dict]:
    """
    Compute arithmetic intensity for each component.
    AI = FLOPs / Bytes. Low AI = memory-bound, High AI = compute-bound.
    """
    n_layers = arch["num_layers"]

    components = {}

    # Attention (per layer, then total)
    attn_flops = compute_attention_flops(seq_len, arch)
    attn_bytes = compute_attention_bytes(seq_len, arch, quant)
    components["attention_per_layer"] = {
        "flops": attn_flops,
        "bytes": attn_bytes,
        "arithmetic_intensity": attn_flops / attn_bytes if attn_bytes > 0 else 0,
    }
    components["attention_total"] = {
        "flops": attn_flops * n_layers,
        "bytes": attn_bytes * n_layers,
        "arithmetic_intensity": attn_flops / attn_bytes if attn_bytes > 0 else 0,
    }

    # MLP (per layer, then total)
    mlp_flops = compute_mlp_flops(arch)
    mlp_bytes = compute_mlp_bytes(arch, quant)
    components["mlp_per_layer"] = {
        "flops": mlp_flops,
        "bytes": mlp_bytes,
        "arithmetic_intensity": mlp_flops / mlp_bytes if mlp_bytes > 0 else 0,
    }
    components["mlp_total"] = {
        "flops": mlp_flops * n_layers,
        "bytes": mlp_bytes * n_layers,
        "arithmetic_intensity": mlp_flops / mlp_bytes if mlp_bytes > 0 else 0,
    }

    # LM head
    head_flops = compute_lm_head_flops(arch)
    head_bytes = compute_lm_head_bytes(arch, quant)
    components["lm_head"] = {
        "flops": head_flops,
        "bytes": head_bytes,
        "arithmetic_intensity": head_flops / head_bytes if head_bytes > 0 else 0,
    }

    # Total decode step
    total_flops = (attn_flops + mlp_flops) * n_layers + head_flops
    total_bytes = (attn_bytes + mlp_bytes) * n_layers + head_bytes
    components["total_decode_step"] = {
        "flops": total_flops,
        "bytes": total_bytes,
        "arithmetic_intensity": total_flops / total_bytes if total_bytes > 0 else 0,
    }

    return components


def classify_bottleneck(arithmetic_intensity: float) -> str:
    """Classify a component as compute-bound or memory-bound."""
    # Rough thresholds (operations per byte):
    # CPU: typically ~10 FLOP/byte peak → AI < 10 is memory-bound
    # GPU: typically ~100 FLOP/byte peak → AI < 50 is memory-bound
    if arithmetic_intensity < 2.0:
        return "strongly memory-bound"
    elif arithmetic_intensity < 10.0:
        return "memory-bound"
    elif arithmetic_intensity < 50.0:
        return "mixed (platform-dependent)"
    else:
        return "compute-bound"


# ============================================================
# Analysis from Benchmark Data
# ============================================================

def analyze_scaling_trends(data_dir: str) -> Dict:
    """Analyze scaling trends from benchmark CSV/JSON data."""
    results = {}

    # Load sequence length scaling results (find most recent file)
    seq_json = find_latest(data_dir, "scaling_*_all.json")
    if seq_json:
        with open(seq_json) as f:
            scaling_data = json.load(f)

        # Analyze TTFT scaling with sequence length
        if "sequence_length" in scaling_data:
            seq_results = scaling_data["sequence_length"]
            lengths = [r["prompt_length"] for r in seq_results]
            ttfts = [r["ttft_mean_ms"] for r in seq_results]
            token_latencies = [r["token_mean_ms"] for r in seq_results]

            if len(lengths) >= 2:
                # Check if TTFT scales linearly (ratio of TTFT to prompt length)
                ratios = [t / l for t, l in zip(ttfts, lengths)]
                results["ttft_scaling"] = {
                    "lengths": lengths,
                    "ttft_ms": ttfts,
                    "ttft_per_token_ms": ratios,
                    "scaling_pattern": "linear" if np.std(ratios) / np.mean(ratios) < 0.3 else "super-linear",
                }

                # Check if decode latency grows with sequence length
                if len(token_latencies) >= 2:
                    growth = (token_latencies[-1] - token_latencies[0]) / token_latencies[0] * 100
                    results["decode_scaling"] = {
                        "lengths": lengths,
                        "token_ms": token_latencies,
                        "growth_pct": growth,
                        "cause": "KV-cache size growth → increased memory bandwidth demand"
                        if growth > 10 else "minimal growth (KV-cache fits in cache hierarchy)",
                    }

    # Load decomposition results (find most recent file)
    decomp_json = find_latest(data_dir, "decomposition_*.json")
    if decomp_json:
        with open(decomp_json) as f:
            decomp_data = json.load(f)

        pcts = decomp_data.get("category_percentages", {})
        if pcts:
            # Identify dominant component
            dominant = max(pcts, key=pcts.get)
            results["dominant_component"] = {
                "component": dominant,
                "percentage": pcts[dominant],
                "interpretation": _interpret_dominant(dominant, pcts),
            }

    return results


def _interpret_dominant(component: str, pcts: Dict) -> str:
    """Provide architectural interpretation of the dominant component."""
    interpretations = {
        "attention": (
            "Attention dominates latency. This is expected for longer sequences "
            "where the KV-cache grows large and requires significant memory bandwidth "
            "to read during each decode step. FlashAttention (Paper 1) addresses this "
            "by tiling attention computation to reduce HBM↔SRAM transfers."
        ),
        "mlp": (
            "MLP (feed-forward network) dominates latency. This is typical for shorter "
            "sequences where KV-cache is small but the large weight matrices (gate_proj, "
            "up_proj, down_proj) require substantial memory bandwidth to load. "
            "Quantization provides near-linear speedup by reducing bytes per weight."
        ),
        "lm_head": (
            "The LM head (vocabulary projection) is a significant fraction of latency. "
            "This is due to the large vocabulary size (128K tokens) requiring a "
            "[hidden_dim x vocab_size] matrix multiplication per token."
        ),
        "embedding": (
            "Embedding lookup is a negligible fraction — this is a simple table lookup."
        ),
    }
    return interpretations.get(component, f"{component} dominates — further analysis needed.")


# ============================================================
# Report Generation
# ============================================================

def generate_report(arch: Dict, seq_len: int, quant: str, data_dir: str) -> str:
    """Generate a text report of the bottleneck analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"ARCHITECTURAL BOTTLENECK ANALYSIS")
    lines.append(f"Model: {arch['name']} | Sequence Length: {seq_len} | Quantization: {quant}")
    lines.append("=" * 70)

    # Arithmetic intensity analysis
    lines.append(f"\n--- Arithmetic Intensity Analysis (Decode Step) ---")
    lines.append(f"{'Component':<25} {'FLOPs':>12} {'Bytes':>12} {'AI (F/B)':>10} {'Classification':>25}")
    lines.append("-" * 85)

    components = analyze_arithmetic_intensity(arch, seq_len, quant)
    for name, data in components.items():
        ai = data["arithmetic_intensity"]
        classification = classify_bottleneck(ai)
        lines.append(
            f"{name:<25} {data['flops']:>12.0f} {data['bytes']:>12.0f} "
            f"{ai:>10.2f} {classification:>25}"
        )

    # KV-cache size analysis
    lines.append(f"\n--- KV-Cache Analysis ---")
    kv_dim = arch["num_kv_heads"] * arch["head_dim"]
    kv_size_bytes = 2 * seq_len * kv_dim * arch["num_layers"] * 2  # 2 tensors (K+V), kv_dim per head group, 2 bytes per FP16 element
    kv_size_mb = kv_size_bytes / (1024 * 1024)
    lines.append(f"KV-cache size at seq_len={seq_len}: {kv_size_mb:.2f} MB")
    lines.append(f"KV-cache size at seq_len=2048: {kv_size_mb * 2048 / seq_len:.2f} MB")

    # Typical cache sizes for context
    lines.append(f"\nTypical cache hierarchy sizes:")
    lines.append(f"  L1 cache:  ~64 KB  — KV-cache {'fits' if kv_size_mb < 0.064 else 'exceeds'}")
    lines.append(f"  L2 cache:  ~1 MB   — KV-cache {'fits' if kv_size_mb < 1 else 'exceeds'}")
    lines.append(f"  L3 cache:  ~8 MB   — KV-cache {'fits' if kv_size_mb < 8 else 'exceeds'}")
    lines.append(f"  L3 (M-chip): ~16 MB — KV-cache {'fits' if kv_size_mb < 16 else 'exceeds'}")

    # Weight memory analysis
    lines.append(f"\n--- Weight Memory Analysis ---")
    bpe = BYTES_PER_ELEMENT[quant]
    total_weight_mb = arch["params_billions"] * 1e9 * bpe / (1024 * 1024)
    lines.append(f"Total weight size ({quant}): {total_weight_mb:.0f} MB")
    lines.append(f"Per-token bandwidth needed: weight load + KV-cache read")
    lines.append(f"  → This is why decode is memory-bandwidth-bound")

    # Hardware bandwidth comparison (Papers 1 & 5: IO-bound verification)
    lines.append(f"\n--- IO-Bound Verification (Hardware Bandwidth) ---")
    total_bytes_per_token = components["total_decode_step"]["bytes"]
    total_bytes_gb = total_bytes_per_token / (1024 ** 3)

    platform_bandwidths = {
        "Intel CPU (DDR4)": 50,
        "Apple M-chip (unified)": 100,
        "NVIDIA T4": 300,
        "NVIDIA A100": 2000,
    }

    lines.append(f"Total bytes accessed per decode step: {total_bytes_per_token / 1e6:.2f} MB")
    lines.append(f"{'Platform':<25} {'BW (GB/s)':>10} {'Ideal ms/token':>15} {'Bottleneck':>15}")
    lines.append("-" * 68)
    for plat, bw_gbps in platform_bandwidths.items():
        ideal_ms = (total_bytes_gb / bw_gbps) * 1000
        bottleneck = "memory BW" if components["total_decode_step"]["arithmetic_intensity"] < 10 else "compute"
        lines.append(f"  {plat:<23} {bw_gbps:>10} {ideal_ms:>14.2f}ms {bottleneck:>15}")

    lines.append(f"\nConclusion: Decode is memory-bandwidth-bound on all platforms.")
    lines.append(f"  Measured latency should be close to ideal (bandwidth-limited) time.")
    lines.append(f"  Ref: FlashAttention (Paper 1) IO-awareness, LLMCompass (Paper 5) IO-bound analysis")

    # Scaling predictions
    lines.append(f"\n--- Scaling Predictions ---")
    lines.append(f"1. TTFT (prefill): Scales ~linearly with prompt length (compute-bound, parallelizable)")
    lines.append(f"2. Per-token (decode): Grows with seq length due to KV-cache bandwidth")
    lines.append(f"3. Model size (1B→3B): ~3x latency increase (proportional to parameter count)")
    lines.append(f"4. Quantization: Near-linear speedup (Q4 ~2x faster than Q8, ~4x faster than F16)")

    # Data-driven analysis
    trends = analyze_scaling_trends(data_dir)
    if trends:
        lines.append(f"\n--- Empirical Observations ---")
        for key, value in trends.items():
            lines.append(f"\n{key}:")
            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")

    lines.append(f"\n--- Key Takeaways ---")
    lines.append(f"• Decode phase is MEMORY-BOUND (low arithmetic intensity ~1-2 FLOP/byte)")
    lines.append(f"• Prefill phase is COMPUTE-BOUND (high arithmetic intensity, batch parallelism)")
    lines.append(f"• KV-cache bandwidth is the primary bottleneck for long sequences")
    lines.append(f"• Quantization directly reduces memory bandwidth → proportional speedup")
    lines.append(f"• Relevant papers: FlashAttention (tiling for IO), PagedAttention (KV management),")
    lines.append(f"  LLMCompass (IO-bound analysis), DynamoLLM (energy-efficient inference)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Architectural Bottleneck Analysis")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--quant", type=str, default="F16", choices=list(BYTES_PER_ELEMENT.keys()))
    parser.add_argument("--model", type=str, default="1B", choices=["1B", "3B"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    arch = LLAMA_1B if args.model == "1B" else LLAMA_3B

    report = generate_report(arch, args.seq_len, args.quant, args.data_dir)
    print(report)

    # Save report
    if args.output:
        output_path = args.output
    else:
        tag = get_platform_tag()
        tag_dir = os.path.join(RESULTS_DIR, tag)
        os.makedirs(tag_dir, exist_ok=True)
        prefix = get_output_prefix()
        output_path = os.path.join(tag_dir, f"bottleneck_{prefix}.txt")

    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {output_path}")

    # Also save structured data as JSON
    components = analyze_arithmetic_intensity(arch, args.seq_len, args.quant)
    trends = analyze_scaling_trends(args.data_dir)
    json_data = {
        "model": arch["name"],
        "seq_len": args.seq_len,
        "quantization": args.quant,
        "arithmetic_intensity": components,
        "scaling_trends": trends,
    }

    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"Structured data saved to {json_path}")


if __name__ == "__main__":
    main()
