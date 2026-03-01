"""
Goal 3: Scaling Analysis
Studies how latency scales with sequence length, model size, and precision.

Experiments:
    1. Sequence length scaling: 128 → 256 → 512 → 1024 → 2048
    2. Model size scaling: 1B vs 3B
    3. Quantization scaling: Q4_K_M vs Q8_0 vs F16

Usage:
    python -m benchmarks.scaling_analysis [--experiment EXPERIMENT]
"""

import argparse
from typing import List, Dict

from benchmarks.config import (
    MODELS, SCALING_SEQUENCE_LENGTHS, SCALING_MODELS,
    SCALING_QUANTIZATIONS, OUTPUT_LENGTH, NUM_TRIALS, WARMUP_RUNS,
)
from benchmarks.benchmark_harness import benchmark_configuration
from benchmarks.utils import (
    detect_platform, get_output_prefix, save_results_csv, save_results_json,
)


def run_sequence_length_scaling(
    model_key: str = "llama-3.2-1b-q4",
    sequence_lengths: List[int] = None,
    num_trials: int = NUM_TRIALS,
) -> List[Dict]:
    """
    Experiment 1: How does latency scale with sequence (prompt) length?
    Keeps model and quantization fixed, varies prompt length.
    """
    if sequence_lengths is None:
        sequence_lengths = SCALING_SEQUENCE_LENGTHS

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT 1: Sequence Length Scaling")
    print(f"# Model: {MODELS[model_key]['name']}")
    print(f"# Lengths: {sequence_lengths}")
    print(f"{'#'*60}")

    results = []
    for seq_len in sequence_lengths:
        result = benchmark_configuration(
            model_key=model_key,
            prompt_length=seq_len,
            num_trials=num_trials,
        )
        result["experiment"] = "sequence_length_scaling"
        results.append(result)

    return results


def run_model_size_scaling(
    model_keys: List[str] = None,
    prompt_length: int = 512,
    num_trials: int = NUM_TRIALS,
) -> List[Dict]:
    """
    Experiment 2: How does latency scale with model size?
    Compares 1B vs 3B at the same quantization and prompt length.
    """
    if model_keys is None:
        model_keys = SCALING_MODELS

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT 2: Model Size Scaling")
    print(f"# Models: {[MODELS[k]['name'] for k in model_keys]}")
    print(f"# Prompt length: {prompt_length}")
    print(f"{'#'*60}")

    results = []
    for model_key in model_keys:
        result = benchmark_configuration(
            model_key=model_key,
            prompt_length=prompt_length,
            num_trials=num_trials,
        )
        result["experiment"] = "model_size_scaling"
        results.append(result)

    return results


def run_quantization_scaling(
    model_keys: List[str] = None,
    prompt_length: int = 512,
    num_trials: int = NUM_TRIALS,
) -> List[Dict]:
    """
    Experiment 3: How does latency scale with quantization precision?
    Compares Q4_K_M vs Q8_0 vs F16 for the same model.
    """
    if model_keys is None:
        model_keys = SCALING_QUANTIZATIONS

    print(f"\n{'#'*60}")
    print(f"# EXPERIMENT 3: Quantization Scaling")
    print(f"# Variants: {[MODELS[k]['name'] for k in model_keys]}")
    print(f"# Prompt length: {prompt_length}")
    print(f"{'#'*60}")

    results = []
    for model_key in model_keys:
        result = benchmark_configuration(
            model_key=model_key,
            prompt_length=prompt_length,
            num_trials=num_trials,
        )
        result["experiment"] = "quantization_scaling"
        results.append(result)

    return results


def run_all_experiments(num_trials: int = NUM_TRIALS) -> Dict[str, List[Dict]]:
    """Run all three scaling experiments."""
    all_results = {}

    all_results["sequence_length"] = run_sequence_length_scaling(num_trials=num_trials)
    all_results["model_size"] = run_model_size_scaling(num_trials=num_trials)
    all_results["quantization"] = run_quantization_scaling(num_trials=num_trials)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="LLaMA Scaling Analysis")
    parser.add_argument(
        "--experiment", type=str, default="all",
        choices=["all", "sequence", "model_size", "quantization"],
        help="Which experiment to run",
    )
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--prompt-length", type=int, default=512)
    prefix = get_output_prefix()
    parser.add_argument("--output", type=str, default=f"scaling_{prefix}")
    args = parser.parse_args()

    print(f"Platform: {detect_platform()}")

    if args.experiment == "all":
        results = run_all_experiments(num_trials=args.trials)
        # Save each experiment separately
        for exp_name, exp_results in results.items():
            csv_data = [{k: v for k, v in r.items() if not k.startswith("raw_")}
                        for r in exp_results]
            save_results_csv(csv_data, f"{args.output}_{exp_name}.csv")
        # Save all as JSON
        save_results_json(results, f"{args.output}_all.json")

    elif args.experiment == "sequence":
        results = run_sequence_length_scaling(num_trials=args.trials)
        csv_data = [{k: v for k, v in r.items() if not k.startswith("raw_")} for r in results]
        save_results_csv(csv_data, f"{args.output}_sequence.csv")

    elif args.experiment == "model_size":
        results = run_model_size_scaling(
            prompt_length=args.prompt_length, num_trials=args.trials
        )
        csv_data = [{k: v for k, v in r.items() if not k.startswith("raw_")} for r in results]
        save_results_csv(csv_data, f"{args.output}_model_size.csv")

    elif args.experiment == "quantization":
        results = run_quantization_scaling(
            prompt_length=args.prompt_length, num_trials=args.trials
        )
        csv_data = [{k: v for k, v in r.items() if not k.startswith("raw_")} for r in results]
        save_results_csv(csv_data, f"{args.output}_quantization.csv")


if __name__ == "__main__":
    main()
