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
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for all platforms
import matplotlib.pyplot as plt
import seaborn as sns

from benchmarks.config import DATA_DIR, RESULTS_DIR


def find_latest(data_dir: str, pattern: str) -> Optional[str]:
    """Find the most recently modified file matching a glob pattern in data_dir."""
    matches = glob.glob(os.path.join(data_dir, pattern))
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _extract_platform_tag(filepath: str) -> Optional[str]:
    """Extract platform tag from auto-tagged filename.

    'benchmark_gpu_A100-SXM4-80GB_20260304_024845.csv' → 'gpu_A100-SXM4-80GB'
    'scaling_mac_arm64_m4pro_20260303_181334_sequence_length.csv' → 'mac_arm64_m4pro'
    """
    basename = os.path.basename(filepath)
    m = re.match(r'^(?:benchmark|scaling|decomposition)_(.+?)_\d{8}_\d{6}', basename)
    return m.group(1) if m else None

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
        plat = result.get("platform", "")
        title_suffix = f" — {plat}" if plat else ""
        ax.set_title(f"Per-Token Generation Latency{title_suffix}\n{result.get('model', 'Model')}")
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
    plat = data.get("platform", "")
    title_suffix = f" — {plat}" if plat else ""
    ax.set_title(f"Latency Decomposition{title_suffix}\n{data.get('model', 'Model')}")

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

    plat = df["platform"].iloc[0] if "platform" in df.columns else ""
    title_suffix = f" — {plat}" if plat else ""

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
    ax1.set_title(f"Time to First Token vs Sequence Length{title_suffix}")

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
    ax2.set_title(f"Decode Latency vs Sequence Length{title_suffix}")

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

    plat = df["platform"].iloc[0] if "platform" in df.columns else ""
    title_suffix = f" — {plat}" if plat else ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = range(len(df))
    labels = df["model"].tolist()

    ax1.bar(x, df["ttft_mean_ms"], color=COLORS[:len(df)], yerr=df["ttft_std_ms"], capsize=5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title(f"Time to First Token by Model Size{title_suffix}")

    ax2.bar(x, df["token_mean_ms"], color=COLORS[:len(df)], yerr=df["token_std_ms"], capsize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylabel("Per-Token Latency (ms)")
    ax2.set_title(f"Decode Latency by Model Size{title_suffix}")

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

    plat = df["platform"].iloc[0] if "platform" in df.columns else ""
    title_suffix = f" — {plat}" if plat else ""
    plt.suptitle(f"Impact of Quantization on Inference Latency{title_suffix}", fontsize=14)
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
    df["ttft_mean_ms"] = df["ttft_mean_ms"].astype(float)
    df["token_mean_ms"] = df["token_mean_ms"].astype(float)

    # Filter to a common prompt length for fair comparison
    common_lengths = [512, 256, 128, 1024]
    chosen_len = None
    for cl in common_lengths:
        if cl in df["prompt_length"].astype(int).values:
            chosen_len = cl
            break
    if chosen_len is not None:
        df = df[df["prompt_length"].astype(int) == chosen_len]

    # Filter to a common model for fair comparison (1B Q4)
    if "model_key" in df.columns:
        q4_df = df[df["model_key"].str.contains("1b") & df["model_key"].str.contains("q4")]
        if not q4_df.empty:
            df = q4_df

    platforms = df["platform"].unique()
    x = np.arange(len(platforms))

    # Average across rows per platform (in case of duplicates)
    ttft_vals = [df[df["platform"] == p]["ttft_mean_ms"].mean() for p in platforms]
    token_vals = [df[df["platform"] == p]["token_mean_ms"].mean() for p in platforms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(x, ttft_vals, 0.5, color=COLORS[:len(platforms)])
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms, rotation=15, ha="right")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title(f"Time to First Token by Platform (prompt={chosen_len})")
    for bar, val in zip(bars1, ttft_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}", ha="center", fontsize=9)

    bars2 = ax2.bar(x, token_vals, 0.5, color=COLORS[:len(platforms)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(platforms, rotation=15, ha="right")
    ax2.set_ylabel("Per-Token Latency (ms)")
    ax2.set_title(f"Decode Latency by Platform (prompt={chosen_len})")
    for bar, val in zip(bars2, token_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    fname = f"cross_platform_comparison_{prefix}.png" if prefix else "cross_platform_comparison.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ============================================================
# Hardware specs for roofline and bandwidth analysis
# ============================================================

PLATFORM_HW = {
    "Mac-M4PRO": {"bw_gbps": 273, "peak_gflops": 4500},
    "GPU-NVIDIA A100-SXM4-80GB": {"bw_gbps": 2039, "peak_gflops": 312000},
    "Windows-x86": {"bw_gbps": 50, "peak_gflops": 500},
}


def _match_platform_hw(platform_label: str) -> Optional[Dict]:
    """Fuzzy-match a platform label to PLATFORM_HW entry."""
    for key in PLATFORM_HW:
        if key.lower().replace("-", "").replace("_", "") in \
           platform_label.lower().replace("-", "").replace("_", ""):
            return PLATFORM_HW[key]
        if platform_label.lower().replace("-", "").replace("_", "") in \
           key.lower().replace("-", "").replace("_", ""):
            return PLATFORM_HW[key]
    return None


# ============================================================
# Plot 7: Cross-Platform Sequence Scaling Overlay
# ============================================================

def plot_cross_sequence_scaling(seq_csvs: Dict[str, Optional[str]],
                                output_dir: str, prefix: str = ""):
    """Overlay TTFT and decode scaling curves from multiple platforms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plotted = False

    for i, (tag, csv_path) in enumerate(seq_csvs.items()):
        if not csv_path:
            continue
        df = load_csv(csv_path)
        plat = df["platform"].iloc[0] if "platform" in df.columns else tag
        color = COLORS[i % len(COLORS)]

        ax1.plot(df["prompt_length"], df["ttft_mean_ms"], "o-",
                 color=color, label=plat, linewidth=2)
        ax1.fill_between(df["prompt_length"],
                         df["ttft_mean_ms"] - df["ttft_std_ms"],
                         df["ttft_mean_ms"] + df["ttft_std_ms"],
                         alpha=0.15, color=color)

        ax2.plot(df["prompt_length"], df["token_mean_ms"], "o-",
                 color=color, label=plat, linewidth=2)
        ax2.fill_between(df["prompt_length"],
                         df["token_mean_ms"] - df["token_std_ms"],
                         df["token_mean_ms"] + df["token_std_ms"],
                         alpha=0.15, color=color)
        plotted = True

    if not plotted:
        plt.close()
        print("No sequence scaling data for cross-platform overlay")
        return

    ax1.set_xlabel("Prompt Length (tokens)")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("TTFT Scaling — Cross-Platform")
    ax1.legend()

    ax2.set_xlabel("Prompt Length (tokens)")
    ax2.set_ylabel("Per-Token Latency (ms)")
    ax2.set_title("Decode Latency Scaling — Cross-Platform")
    ax2.legend()

    plt.tight_layout()
    fname = f"cross_sequence_scaling_{prefix}.png" if prefix else "cross_sequence_scaling.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ============================================================
# Plot 8: Roofline Model
# ============================================================

def plot_roofline(bottleneck_jsons: Dict[str, Optional[str]],
                  bench_csvs: List[str], output_dir: str, prefix: str = ""):
    """Roofline plot: arithmetic intensity vs achieved GFLOP/s per platform."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Draw roofline ceilings for each known platform
    ai_range = np.logspace(-2, 4, 200)
    for i, (plat_label, hw) in enumerate(PLATFORM_HW.items()):
        bw = hw["bw_gbps"]       # GB/s
        peak = hw["peak_gflops"]  # GFLOP/s
        # Roofline: min(peak, bw * AI)
        roof = np.minimum(peak, bw * ai_range)
        color = COLORS[i % len(COLORS)]
        ax.loglog(ai_range, roof, "--", color=color, linewidth=1.5,
                  label=f"{plat_label} roofline", alpha=0.6)

    # Load benchmark CSVs to get tokens_per_sec per platform
    bench_frames = []
    for path in bench_csvs:
        if os.path.exists(path):
            bench_frames.append(load_csv(path))
    bench_df = pd.concat(bench_frames, ignore_index=True) if bench_frames else pd.DataFrame()

    # Plot measured workload points from bottleneck JSONs
    for i, (tag, json_path) in enumerate(bottleneck_jsons.items()):
        if not json_path or not os.path.exists(json_path):
            continue
        data = load_json(json_path)
        ai_data = data.get("arithmetic_intensity", {}).get("total_decode_step", {})
        ai_val = ai_data.get("arithmetic_intensity")
        total_flops = ai_data.get("flops")
        if ai_val is None or total_flops is None:
            continue

        # Get tokens/sec from benchmark CSV for this platform
        tps = None
        if not bench_df.empty:
            # Match platform by tag substring
            for _, row in bench_df.iterrows():
                plat_str = str(row.get("platform", ""))
                if tag.replace("_", "").lower() in plat_str.replace("-", "").replace("_", "").lower() or \
                   plat_str.replace("-", "").replace("_", "").lower() in tag.replace("_", "").lower():
                    tps = float(row["tokens_per_sec"])
                    plat_label = plat_str
                    break

        if tps is None:
            continue

        achieved_gflops = total_flops * tps / 1e9
        color = COLORS[i % len(COLORS)]
        ax.loglog(ai_val, achieved_gflops, "o", color=color, markersize=12,
                  markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ax.annotate(plat_label, (ai_val, achieved_gflops),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Achieved Performance (GFLOP/s)")
    ax.set_title("Roofline Model — LLaMA Decode Workload")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fname = f"roofline_{prefix}.png" if prefix else "roofline.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


# ============================================================
# Plot 9: Throughput vs Memory Bandwidth
# ============================================================

def plot_throughput_vs_bandwidth(bench_csvs: List[str],
                                 output_dir: str, prefix: str = ""):
    """Scatter plot: memory bandwidth (GB/s) vs measured tokens/sec."""
    frames = []
    for path in bench_csvs:
        if os.path.exists(path):
            frames.append(load_csv(path))
    if not frames:
        print("No benchmark data for throughput vs bandwidth plot")
        return

    df = pd.concat(frames, ignore_index=True)

    # Collect (bandwidth, tokens/sec, platform_label) per platform
    points = []
    for plat_label in df["platform"].unique():
        hw = _match_platform_hw(plat_label)
        if hw is None:
            continue
        plat_df = df[df["platform"] == plat_label]
        tps = plat_df["tokens_per_sec"].astype(float).mean()
        points.append((hw["bw_gbps"], tps, plat_label))

    if len(points) < 2:
        print("Need at least 2 platforms for throughput vs bandwidth plot")
        return

    bws = [p[0] for p in points]
    tps_vals = [p[1] for p in points]
    labels = [p[2] for p in points]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(bws, tps_vals, s=120, color=COLORS[:len(points)],
               edgecolors="black", linewidths=1.5, zorder=5)

    for bw, tps, label in points:
        ax.annotate(label, (bw, tps), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

    # Best-fit line
    bws_arr = np.array(bws)
    tps_arr = np.array(tps_vals)
    coeffs = np.polyfit(bws_arr, tps_arr, 1)
    fit_x = np.linspace(0, max(bws_arr) * 1.1, 100)
    fit_y = np.polyval(coeffs, fit_x)
    r_sq = 1 - np.sum((tps_arr - np.polyval(coeffs, bws_arr))**2) / \
               np.sum((tps_arr - np.mean(tps_arr))**2)
    ax.plot(fit_x, fit_y, "--", color="gray", linewidth=1.5,
            label=f"Linear fit (R²={r_sq:.3f})")

    ax.set_xlabel("Memory Bandwidth (GB/s)")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput vs Memory Bandwidth — Cross-Platform")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    fname = f"throughput_vs_bandwidth_{prefix}.png" if prefix else "throughput_vs_bandwidth.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark plots")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--platforms", nargs="+",
                        help="Platform tags to include (e.g. mac_arm64_m4pro gpu_A100-SXM4-80GB)")
    parser.add_argument("--select", action="store_true",
                        help="Interactive platform selection menu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data_dir = args.data_dir

    # Discover unique platform tags from benchmark CSVs
    all_bench_csvs = sorted(glob.glob(os.path.join(data_dir, "benchmark_*.csv")))
    tags = set()
    for csv_path in all_bench_csvs:
        tag = _extract_platform_tag(csv_path)
        if tag:
            tags.add(tag)

    if not tags:
        print("No benchmark data found in", data_dir)
        return

    # Platform selection
    if args.platforms:
        selected = set(args.platforms)
        invalid = selected - tags
        if invalid:
            print(f"Warning: unknown platform tags ignored: {invalid}")
        tags = tags & selected
    elif args.select:
        sorted_tags = sorted(tags)
        print("Available platforms:")
        for i, t in enumerate(sorted_tags, 1):
            print(f"  {i}) {t}")
        choice = input("\nSelect platforms (space-separated numbers, or 'all'): ").strip()
        if choice.lower() != "all":
            indices = [int(x) - 1 for x in choice.split()]
            tags = {sorted_tags[i] for i in indices if 0 <= i < len(sorted_tags)}

    if not tags:
        print("No platforms selected")
        return

    # Filter bench CSVs to selected platforms only
    all_bench_csvs = [c for c in all_bench_csvs if _extract_platform_tag(c) in tags]

    print(f"Platforms: {', '.join(sorted(tags))}")

    # Generate per-platform plots into results/{tag}/
    for tag in sorted(tags):
        plat_dir = os.path.join(args.output_dir, tag)
        os.makedirs(plat_dir, exist_ok=True)
        print(f"\n=== Generating plots for {tag} ===")

        seq_csv = find_latest(data_dir, f"scaling_{tag}_*_sequence_length.csv") or \
                  find_latest(data_dir, f"scaling_{tag}_*_sequence.csv")
        if seq_csv:
            print(f"  Using: {os.path.basename(seq_csv)}")
            plot_sequence_scaling(seq_csv, plat_dir, tag)

        model_csv = find_latest(data_dir, f"scaling_{tag}_*_model_size.csv")
        if model_csv:
            print(f"  Using: {os.path.basename(model_csv)}")
            plot_model_size_comparison(model_csv, plat_dir, tag)

        quant_csv = find_latest(data_dir, f"scaling_{tag}_*_quantization.csv")
        if quant_csv:
            print(f"  Using: {os.path.basename(quant_csv)}")
            plot_quantization_impact(quant_csv, plat_dir, tag)

        decomp_jsons = sorted(glob.glob(os.path.join(data_dir, f"decomposition_{tag}_*.json")))
        decomp_jsons = [f for f in decomp_jsons if "_steps" not in f]
        if decomp_jsons:
            print(f"  Using: {os.path.basename(decomp_jsons[-1])}")
            plot_decomposition(decomp_jsons[-1], plat_dir, tag)

        bench_jsons = sorted(glob.glob(os.path.join(data_dir, f"benchmark_{tag}_*.json")))
        for bench_json in bench_jsons:
            print(f"  Using: {os.path.basename(bench_json)}")
            plot_per_token_timeline(bench_json, plat_dir, tag)

    # Cross-platform comparison → results/{tag1}_vs_{tag2}_vs_.../
    if len(tags) >= 2:
        cross_name = "_vs_".join(sorted(tags))
        cross_dir = os.path.join(args.output_dir, cross_name)
        os.makedirs(cross_dir, exist_ok=True)
        print(f"\n=== Cross-platform comparison ({cross_name}) ===")
        plot_cross_platform(all_bench_csvs, cross_dir, cross_name)

        # Sequence scaling overlay
        seq_csvs = {}
        for tag in sorted(tags):
            seq_csvs[tag] = find_latest(data_dir, f"scaling_{tag}_*_sequence_length.csv") or \
                            find_latest(data_dir, f"scaling_{tag}_*_sequence.csv")
        plot_cross_sequence_scaling(seq_csvs, cross_dir, cross_name)

        # Roofline model
        bottleneck_jsons = {}
        for tag in sorted(tags):
            bottleneck_jsons[tag] = find_latest(
                os.path.join(args.output_dir, tag), "bottleneck_*.json")
        plot_roofline(bottleneck_jsons, all_bench_csvs, cross_dir, cross_name)

        # Throughput vs memory bandwidth
        plot_throughput_vs_bandwidth(all_bench_csvs, cross_dir, cross_name)

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
