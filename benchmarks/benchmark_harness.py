"""
Goal 1: Benchmark Harness
Measures TTFT, per-token latency, and end-to-end response time
using llama-cpp-python with GGUF models.

Usage:
    python -m benchmarks.benchmark_harness [--model MODEL_KEY] [--prompt-length N] [--trials N]

Works on all platforms: Mac (Metal), Windows (AVX2), Colab (CUDA).
"""

import argparse
import contextlib
import os
import time
from typing import List, Dict


@contextlib.contextmanager
def suppress_c_stderr():
    """Redirect C-level stderr (fd 2) to /dev/null to silence llama.cpp init messages."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    old_fd = os.dup(2)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(old_fd, 2)
        os.close(old_fd)

from llama_cpp import Llama

from benchmarks.config import (
    MODELS, PROMPT_LENGTHS, OUTPUT_LENGTH, WARMUP_RUNS,
    NUM_TRIALS, CONTEXT_SIZE,
)
from benchmarks.utils import (
    get_gguf_model_path, get_platform_label, get_output_prefix, detect_platform,
    generate_prompt_text, filter_outliers_iqr, compute_stats,
    save_results_csv, save_results_json, CPUTimer,
)


def run_single_benchmark(
    llm: Llama,
    prompt_text: str,
    output_length: int,
) -> Dict:
    """
    Run a single inference benchmark and return detailed timing data.
    Expects a pre-loaded Llama model; resets KV cache before each run.

    Returns dict with:
        - ttft_ms: Time to first token (prompt evaluation)
        - token_times_ms: List of per-token generation times
        - e2e_ms: End-to-end total time
        - tokens_generated: Number of tokens actually generated
    """
    llm.reset()  # Clear KV cache from any previous trial

    timer = CPUTimer()

    # Tokenize the prompt
    prompt_tokens = llm.tokenize(prompt_text.encode("utf-8"))

    # --- Measure TTFT (prompt evaluation) ---
    e2e_start = time.perf_counter_ns()
    timer.start()
    llm.eval(prompt_tokens)
    ttft_ms = timer.stop()

    # --- Measure per-token generation ---
    token_times_ms = []
    tokens_generated = 0

    for i in range(output_length):
        timer.start()
        # Sample next token
        token = llm.sample(
            top_k=40,
            top_p=0.95,
            temp=0.8,
            repeat_penalty=1.1,
        )
        # Check for end-of-sequence
        if token == llm.token_eos():
            token_time = timer.stop()
            token_times_ms.append(token_time)
            tokens_generated += 1
            break
        # Evaluate the new token (feeds it into KV-cache)
        llm.eval([token])
        token_time = timer.stop()
        token_times_ms.append(token_time)
        tokens_generated += 1

    e2e_ms = (time.perf_counter_ns() - e2e_start) / 1_000_000

    return {
        "ttft_ms": ttft_ms,
        "token_times_ms": token_times_ms,
        "e2e_ms": e2e_ms,
        "tokens_generated": tokens_generated,
    }


def benchmark_configuration(
    model_key: str,
    prompt_length: int,
    output_length: int = OUTPUT_LENGTH,
    num_trials: int = NUM_TRIALS,
    warmup_runs: int = WARMUP_RUNS,
) -> Dict:
    """
    Run a full benchmark for one model + prompt length configuration.
    Includes warm-up runs and outlier filtering.
    """
    model_cfg = MODELS[model_key]
    model_path = get_gguf_model_path(model_key)
    platform_label = get_platform_label()

    # Detect GPU layers
    platform_info = detect_platform()
    n_gpu_layers = -1 if platform_info["platform_type"] == "cuda_gpu" else 0

    # Generate prompt text of target length
    prompt_text = generate_prompt_text(prompt_length)

    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_cfg['name']}")
    print(f"Platform: {platform_label}")
    print(f"Prompt length: ~{prompt_length} tokens")
    print(f"Output length: {output_length} tokens")
    print(f"Trials: {warmup_runs} warmup + {num_trials} measured")
    print(f"{'='*60}")

    # Load model once — suppresses C-level Metal init spam
    print("Loading model...")
    with suppress_c_stderr():
        llm = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    print("Model loaded.")

    # --- Warm-up runs ---
    print(f"Running {warmup_runs} warm-up iterations...")
    for warmup_idx in range(warmup_runs):
        run_single_benchmark(llm, prompt_text, output_length)
        print(f"  Warm-up {warmup_idx+1}/{warmup_runs} done")

    # --- Measurement runs ---
    trial_results = []
    print(f"Running {num_trials} measurement trials...")
    for trial in range(num_trials):
        result = run_single_benchmark(llm, prompt_text, output_length)
        trial_results.append(result)

        # Per-token stats for this trial
        mean_token = compute_stats(result["token_times_ms"])["mean"]
        print(
            f"  Trial {trial+1}/{num_trials}: "
            f"TTFT={result['ttft_ms']:.2f}ms, "
            f"mean_token={mean_token:.2f}ms, "
            f"e2e={result['e2e_ms']:.2f}ms, "
            f"tokens={result['tokens_generated']}"
        )

    # --- Aggregate results ---
    all_ttft = [r["ttft_ms"] for r in trial_results]
    all_e2e = [r["e2e_ms"] for r in trial_results]
    all_mean_token = [
        compute_stats(r["token_times_ms"])["mean"] for r in trial_results
    ]

    # Filter outliers
    filtered_ttft = filter_outliers_iqr(all_ttft)
    filtered_e2e = filter_outliers_iqr(all_e2e)
    filtered_mean_token = filter_outliers_iqr(all_mean_token)

    ttft_stats = compute_stats(filtered_ttft)
    e2e_stats = compute_stats(filtered_e2e)
    token_stats = compute_stats(filtered_mean_token)

    # Compute tokens/sec
    tokens_per_sec = 1000.0 / token_stats["mean"] if token_stats["mean"] > 0 else 0

    del llm  # Free model memory after all trials

    print(f"\n--- Results (after IQR outlier filtering) ---")
    print(f"TTFT:      mean={ttft_stats['mean']:.2f}ms, std={ttft_stats['std']:.2f}ms")
    print(f"Token:     mean={token_stats['mean']:.2f}ms, p95={token_stats['p95']:.2f}ms")
    print(f"E2E:       mean={e2e_stats['mean']:.2f}ms")
    print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")

    return {
        "platform": platform_label,
        "model": model_cfg["name"],
        "model_key": model_key,
        "params": model_cfg["params"],
        "quantization": model_cfg["quantization"],
        "prompt_length": prompt_length,
        "output_length": output_length,
        "num_trials": num_trials,
        "num_trials_after_filtering": len(filtered_ttft),
        "ttft_mean_ms": ttft_stats["mean"],
        "ttft_std_ms": ttft_stats["std"],
        "ttft_p95_ms": ttft_stats["p95"],
        "token_mean_ms": token_stats["mean"],
        "token_median_ms": token_stats["median"],
        "token_std_ms": token_stats["std"],
        "token_p95_ms": token_stats["p95"],
        "token_p99_ms": token_stats["p99"],
        "e2e_mean_ms": e2e_stats["mean"],
        "e2e_std_ms": e2e_stats["std"],
        "tokens_per_sec": tokens_per_sec,
        # Raw trial data for later analysis
        "raw_ttft": all_ttft,
        "raw_mean_token": all_mean_token,
        "raw_e2e": all_e2e,
    }


def run_full_benchmark(
    model_keys: List[str] = None,
    prompt_lengths: List[int] = None,
) -> List[Dict]:
    """Run benchmark across all model + prompt length combinations."""
    if model_keys is None:
        model_keys = ["llama-3.2-1b-q4"]
    if prompt_lengths is None:
        prompt_lengths = PROMPT_LENGTHS

    all_results = []
    for model_key in model_keys:
        for prompt_len in prompt_lengths:
            result = benchmark_configuration(model_key, prompt_len)
            all_results.append(result)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="LLaMA Token-Generation Latency Benchmark")
    parser.add_argument(
        "--model", type=str, default="llama-3.2-1b-q4",
        choices=list(MODELS.keys()),
        help="Model configuration key",
    )
    parser.add_argument(
        "--prompt-length", type=int, default=None,
        help="Single prompt length to test (default: all configured lengths)",
    )
    parser.add_argument("--trials", type=int, default=NUM_TRIALS, help="Number of measurement trials")
    parser.add_argument("--warmup", type=int, default=WARMUP_RUNS, help="Number of warm-up runs")
    parser.add_argument("--output-tokens", type=int, default=OUTPUT_LENGTH, help="Tokens to generate")
    parser.add_argument("--all-models", action="store_true", help="Run all model configurations")
    prefix = get_output_prefix()
    parser.add_argument("--output", type=str, default=f"benchmark_{prefix}.csv", help="Output filename")
    args = parser.parse_args()

    # Determine what to run
    if args.all_models:
        model_keys = list(MODELS.keys())
    else:
        model_keys = [args.model]

    prompt_lengths = [args.prompt_length] if args.prompt_length else PROMPT_LENGTHS

    print("Platform info:", detect_platform())
    print(f"Models: {model_keys}")
    print(f"Prompt lengths: {prompt_lengths}")

    all_results = []
    for model_key in model_keys:
        for prompt_len in prompt_lengths:
            result = benchmark_configuration(
                model_key, prompt_len,
                output_length=args.output_tokens,
                num_trials=args.trials,
                warmup_runs=args.warmup,
            )
            all_results.append(result)

    # Save CSV (without raw arrays — those go to JSON)
    csv_results = []
    for r in all_results:
        csv_row = {k: v for k, v in r.items() if not k.startswith("raw_")}
        csv_results.append(csv_row)
    save_results_csv(csv_results, args.output)

    # Save full results with raw data as JSON
    json_filename = args.output.replace(".csv", ".json")
    save_results_json(all_results, json_filename)


if __name__ == "__main__":
    main()
