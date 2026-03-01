"""
Visualization scripts for benchmark results.
Generates all plots needed for the presentation and survey paper.

Usage:
    python -m analysis.plot_results [--data-dir DATA_DIR] [--output-dir RESULTS_DIR]
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for all platforms
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarks.config import DATA_DIR, RESULTS_DIR
from benchmarks.utils import get_output_prefix


def find_latest(data_dir: str, pattern: str) -> Optional[str]:
    """Find the most recently modified file matching a glob pattern in data_dir."""
    matches = glob.glob(os.path.join(data_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

# Style configuration
sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = sns.color_palette("husl", 8)


def load_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def load_json(filepath: str) -> Dict:
    with open(filepath) as f:
        return json.load(f)


# ============================================================
# Plot 1: Per-Token Latency Timeline
# ============================================================

def plot_per_token_timeline(json_path: str, output_dir: str, prefix: str = ""):
    """Plot per-token latency over generation (shows warm-up and steady state)."""
    data = load_json(json_path)

    for result in data if isinstance(data, list) else [data]:
        raw_tokens = result.get("raw_mean_token") or result.get("token_times_ms", [])
        if not raw_tokens:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(raw_tokens)), raw_tokens, marker=".", markersize=3, linewidth=1)
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"Per-Token Generation Latency\n{result.get('model', 'Model')}")
        ax.axhline(y=np.mean(raw_tokens), color="r", linestyle="--",
                    label=f"Mean: {np.mean(raw_tokens):.2f}ms")
        ax.legend()
        plt.tight_layout()

        name = result.get("model_key", "model").replace("/", "_")
        fname = f"token_timeline_{name}_{prefix}.png" if prefix else f"token_timeline_{name}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        print(f"Saved: {fname}")


# ============================================================
# Plot 2: Latency Decomposition Stacked Bar
# ============================================================

def plot_decomposition(json_path: str, output_dir: str, prefix: str = ""):
    """Plot latency decomposition as stacked bar chart (percentages)."""
    data = load_json(json_path)
    pcts = data.get("category_percentages", {})
    if not pcts:
        print("No decomposition data found")
        return

    categories = list(pcts.keys())
    values = [pcts[c] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(categories, values, color=COLORS[:len(categories)])
    ax.set_xlabel("% of Total Latency")
    ax.set_title(f"Latency Decomposition — {data.get('model', 'Model')}")

    # Add percentage labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    fname = f"decomposition_breakdown_{prefix}.png" if prefix else "decomposition_breakdown.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ============================================================
# Plot 3: Sequence Length Scaling
# ============================================================

def plot_sequence_scaling(csv_path: str, output_dir: str, prefix: str = ""):
    """Plot TTFT and per-token latency vs sequence length."""
    df = load_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # TTFT vs sequence length
    ax1.plot(df["prompt_length"], df["ttft_mean_ms"], "o-", color=COLORS[0], linewidth=2)
    ax1.fill_between(
        df["prompt_length"],
        df["ttft_mean_ms"] - df["ttft_std_ms"],
        df["ttft_mean_ms"] + df["ttft_std_ms"],
        alpha=0.2, color=COLORS[0],
    )
    ax1.set_xlabel("Prompt Length (tokens)")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to First Token vs Sequence Length")

    # Per-token latency vs sequence length
    ax2.plot(df["prompt_length"], df["token_mean_ms"], "o-", color=COLORS[1], linewidth=2)
    ax2.fill_between(
        df["prompt_length"],
        df["token_mean_ms"] - df["token_std_ms"],
        df["token_mean_ms"] + df["token_std_ms"],
        alpha=0.2, color=COLORS[1],
    )
    ax2.set_xlabel("Prompt Length (tokens)")
    ax2.set_ylabel("Per-Token Latency (ms)")
    ax2.set_title("Decode Latency vs Sequence Length")

    plt.tight_layout()
    fname = f"scaling_sequence_length_{prefix}.png" if prefix else "scaling_sequence_length.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ============================================================
# Plot 4: Model Size Comparison
# ============================================================

def plot_model_size_comparison(csv_path: str, output_dir: str, prefix: str = ""):
    """Plot side-by-side bars for 1B vs 3B model."""
    df = load_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(df))
    labels = df["model"].tolist()

    ax1.bar(x, df["ttft_mean_ms"], color=COLORS[:len(df)], yerr=df["ttft_std_ms"], capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to First Token by Model Size")

    ax2.bar(x, df["token_mean_ms"], color=COLORS[:len(df)], yerr=df["token_std_ms"], capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Per-Token Latency (ms)")
    ax2.set_title("Decode Latency by Model Size")

    plt.tight_layout()
    fname = f"scaling_model_size_{prefix}.png" if prefix else "scaling_model_size.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ============================================================
# Plot 5: Quantization Impact
# ============================================================

def plot_quantization_impact(csv_path: str, output_dir: str, prefix: str = ""):
    """Plot grouped bars for Q4 vs Q8 vs F16."""
    df = load_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = range(len(df))
    labels = df["quantization"].tolist()

    for ax, metric, title in zip(
        axes,
        ["ttft_mean_ms", "token_mean_ms", "tokens_per_sec"],
        ["TTFT (ms)", "Per-Token Latency (ms)", "Throughput (tokens/sec)"],
    ):
        ax.bar(x, df[metric].astype(float), color=COLORS[:len(df)])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(title)
        ax.set_title(title)

    plt.suptitle("Impact of Quantization on Inference Latency", fontsize=14)
    plt.tight_layout()
    fname = f"scaling_quantization_{prefix}.png" if prefix else "scaling_quantization.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ============================================================
# Plot 6: Cross-Platform Comparison
# ============================================================

def plot_cross_platform(csv_paths: List[str], output_dir: str, prefix: str = ""):
    """Plot grouped bars comparing platforms (Intel vs M-chip vs GPU)."""
    frames = []
    for path in csv_paths:
        if os.path.exists(path):
            frames.append(load_csv(path))
    if not frames:
        print("No cross-platform data found")
        return

    df = pd.concat(frames, ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    platforms = df["platform"].unique()
    x = np.arange(len(platforms))
    width = 0.35

    # Use first matching row per platform (simplification)
    ttft_vals = [df[df["platform"] == p]["ttft_mean_ms"].astype(float).iloc[0] for p in platforms]
    token_vals = [df[df["platform"] == p]["token_mean_ms"].astype(float).iloc[0] for p in platforms]

    ax1.bar(x, ttft_vals, width, color=COLORS[:len(platforms)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms, rotation=15, ha="right")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Time to First Token by Platform")

    ax2.bar(x, token_vals, width, color=COLORS[:len(platforms)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(platforms, rotation=15, ha="right")
    ax2.set_ylabel("Per-Token Latency (ms)")
    ax2.set_title("Decode Latency by Platform")

    plt.tight_layout()
    fname = f"cross_platform_comparison_{prefix}.png" if prefix else "cross_platform_comparison.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prefix = get_output_prefix()

    # Generate available plots based on what data files exist
    # Uses glob patterns to find platform-tagged and timestamped files
    data_dir = args.data_dir

    # Sequence length scaling
    seq_csv = find_latest(data_dir, "scaling_*_sequence_length.csv") or find_latest(data_dir, "scaling_*_sequence.csv")
    if seq_csv:
        print(f"Using: {os.path.basename(seq_csv)}")
        plot_sequence_scaling(seq_csv, args.output_dir, prefix)

    # Model size scaling
    model_csv = find_latest(data_dir, "scaling_*_model_size.csv")
    if model_csv:
        print(f"Using: {os.path.basename(model_csv)}")
        plot_model_size_comparison(model_csv, args.output_dir, prefix)

    # Quantization scaling
    quant_csv = find_latest(data_dir, "scaling_*_quantization.csv")
    if quant_csv:
        print(f"Using: {os.path.basename(quant_csv)}")
        plot_quantization_impact(quant_csv, args.output_dir, prefix)

    # Decomposition
    decomp_json = find_latest(data_dir, "decomposition_*.json")
    if decomp_json:
        print(f"Using: {os.path.basename(decomp_json)}")
        plot_decomposition(decomp_json, args.output_dir, prefix)

    # Benchmark harness results
    bench_json = find_latest(data_dir, "benchmark_*.json")
    if bench_json:
        print(f"Using: {os.path.basename(bench_json)}")
        plot_per_token_timeline(bench_json, args.output_dir, prefix)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
