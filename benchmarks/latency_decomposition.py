"""
Goal 2: Latency Decomposition
Decomposes per-token latency into architectural components using
HuggingFace Transformers with PyTorch forward hooks.

Components timed:
    - Embedding lookup
    - Attention (QKV projection, softmax, output projection)
    - KV-cache read/write (implicit in attention with use_cache=True)
    - MLP / Feed-Forward Network
    - LayerNorm + residual connections
    - LM head (vocabulary projection)
    - Sampling / decoding logic

Usage:
    python -m benchmarks.latency_decomposition [--model HF_MODEL_ID] [--tokens N]

IMPORTANT:
    - Uses attn_implementation="eager" to avoid fused attention kernels
    - Reports component times as PERCENTAGES (hooks add sync overhead)
    - For absolute latency numbers, use benchmark_harness.py
"""

import argparse
import time
from collections import defaultdict
from typing import Dict, List

import torch
import numpy as np

from benchmarks.config import DEFAULT_HF_MODEL
from benchmarks.utils import (
    detect_platform, get_platform_label, get_output_prefix,
    compute_stats, save_results_csv, save_results_json,
)


# ============================================================
# Hook-based Layer Timing
# ============================================================

class LayerTimer:
    """Manages forward hooks to time individual model components."""

    def __init__(self, use_cuda: bool = False):
        self.use_cuda = use_cuda
        self.timings = defaultdict(list)  # name -> list of elapsed_ns per step
        self._current_step = defaultdict(float)  # name -> start_ns for current step
        self._hooks = []

    def _make_pre_hook(self, name: str):
        """Create a pre-forward hook that records start time."""
        def hook(module, input):
            if self.use_cuda:
                torch.cuda.synchronize()
            self._current_step[name] = time.perf_counter_ns()
        return hook

    def _make_post_hook(self, name: str):
        """Create a post-forward hook that records elapsed time."""
        def hook(module, input, output):
            if self.use_cuda:
                torch.cuda.synchronize()
            elapsed_ns = time.perf_counter_ns() - self._current_step[name]
            self.timings[name].append(elapsed_ns)
        return hook

    def register_hooks(self, model):
        """Register timing hooks on all relevant model components."""
        hook_targets = self._identify_hook_targets(model)
        for name, module in hook_targets:
            h1 = module.register_forward_pre_hook(self._make_pre_hook(name))
            h2 = module.register_forward_hook(self._make_post_hook(name))
            self._hooks.extend([h1, h2])
        print(f"Registered timing hooks on {len(hook_targets)} components")
        return [name for name, _ in hook_targets]

    def _identify_hook_targets(self, model):
        """Identify which modules to hook based on LLaMA architecture."""
        targets = []
        for name, module in model.named_modules():
            # Embedding
            if name.endswith("embed_tokens"):
                targets.append(("embedding", module))
            # Individual transformer layers — attention
            elif name.endswith(".self_attn") and not any(
                sub in name for sub in ["q_proj", "k_proj", "v_proj", "o_proj"]
            ):
                layer_idx = name.split(".")[2] if "layers" in name else "?"
                targets.append((f"layer{layer_idx}.attention", module))
            # Individual transformer layers — MLP
            elif name.endswith(".mlp") and not any(
                sub in name for sub in ["gate_proj", "up_proj", "down_proj"]
            ):
                layer_idx = name.split(".")[2] if "layers" in name else "?"
                targets.append((f"layer{layer_idx}.mlp", module))
            # LayerNorm (input)
            elif name.endswith(".input_layernorm"):
                layer_idx = name.split(".")[2] if "layers" in name else "?"
                targets.append((f"layer{layer_idx}.input_layernorm", module))
            # LayerNorm (post-attention)
            elif name.endswith(".post_attention_layernorm"):
                layer_idx = name.split(".")[2] if "layers" in name else "?"
                targets.append((f"layer{layer_idx}.post_attn_layernorm", module))
            # Final norm
            elif name == "model.norm":
                targets.append(("final_norm", module))
            # LM head
            elif name == "lm_head":
                targets.append(("lm_head", module))

        return targets

    def clear_step_timings(self):
        """Clear all recorded timings."""
        self.timings.clear()
        self._current_step.clear()

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_step_breakdown(self, step_idx: int) -> Dict[str, float]:
        """Get timing breakdown for a specific generation step (in ms)."""
        breakdown = {}
        for name, times in self.timings.items():
            if step_idx < len(times):
                breakdown[name] = times[step_idx] / 1_000_000  # ns -> ms
        return breakdown

    def get_aggregated_breakdown(self) -> Dict[str, Dict]:
        """Get aggregated timing breakdown across all steps."""
        result = {}
        for name, times in self.timings.items():
            ms_times = [t / 1_000_000 for t in times]
            result[name] = compute_stats(ms_times)
        return result

    def get_category_breakdown(self) -> Dict[str, float]:
        """Aggregate timings by component category (attention, mlp, etc.)."""
        categories = defaultdict(float)
        for name, times in self.timings.items():
            total_ms = sum(t / 1_000_000 for t in times)
            if "attention" in name:
                categories["attention"] += total_ms
            elif "mlp" in name:
                categories["mlp"] += total_ms
            elif "layernorm" in name:
                categories["layernorm"] += total_ms
            elif name == "embedding":
                categories["embedding"] += total_ms
            elif name == "final_norm":
                categories["final_norm"] += total_ms
            elif name == "lm_head":
                categories["lm_head"] += total_ms
        return dict(categories)


# ============================================================
# Autoregressive Generation with Timing
# ============================================================

def run_decomposition(
    model_id: str,
    num_tokens: int = 64,
    prompt_text: str = "Explain the concept of attention in transformer models.",
    device: str = "auto",
) -> Dict:
    """
    Run latency decomposition using HuggingFace model with forward hooks.

    Args:
        model_id: HuggingFace model ID
        num_tokens: Number of tokens to generate
        prompt_text: Input prompt
        device: Device to use ('cpu', 'cuda', 'auto')

    Returns:
        Dictionary with per-step and aggregated timing breakdowns
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    use_cuda = device == "cuda"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_id}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"CRITICAL: Using attn_implementation='eager' for per-component timing")

    # Load model with eager attention (NOT flash/sdpa)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    # Set up padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Register hooks
    layer_timer = LayerTimer(use_cuda=use_cuda)
    hook_names = layer_timer.register_hooks(model)
    print(f"Hooked components: {hook_names[:10]}...")

    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"Prompt tokens: {input_ids.shape[1]}")

    # --- Manual autoregressive generation ---
    print(f"\nGenerating {num_tokens} tokens with per-component timing...")
    past_key_values = None
    sampling_times_ns = []
    step_total_times_ns = []

    for step in range(num_tokens):
        step_start = time.perf_counter_ns()

        with torch.no_grad():
            if past_key_values is not None:
                # After first token, only feed the last generated token
                outputs = model(
                    input_ids=next_token_id,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                # First forward pass processes entire prompt
                outputs = model(
                    input_ids=input_ids,
                    use_cache=True,
                )

        # Sampling (timed separately)
        if use_cuda:
            torch.cuda.synchronize()
        sample_start = time.perf_counter_ns()

        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        if use_cuda:
            torch.cuda.synchronize()
        sampling_times_ns.append(time.perf_counter_ns() - sample_start)

        past_key_values = outputs.past_key_values
        step_total_times_ns.append(time.perf_counter_ns() - step_start)

        # Check EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"  EOS at step {step}")
            break

        if (step + 1) % 10 == 0:
            step_ms = step_total_times_ns[-1] / 1_000_000
            print(f"  Step {step+1}/{num_tokens}: {step_ms:.2f}ms")

    actual_tokens = len(step_total_times_ns)
    print(f"\nGenerated {actual_tokens} tokens")

    # --- Compute results ---
    # Per-step breakdown
    per_step_data = []
    for step in range(actual_tokens):
        step_data = {"step": step}
        step_data.update(layer_timer.get_step_breakdown(step))
        step_data["sampling_ms"] = sampling_times_ns[step] / 1_000_000
        step_data["total_ms"] = step_total_times_ns[step] / 1_000_000
        per_step_data.append(step_data)

    # Category breakdown (percentages)
    category_totals = layer_timer.get_category_breakdown()
    sampling_total = sum(t / 1_000_000 for t in sampling_times_ns)
    category_totals["sampling"] = sampling_total
    grand_total = sum(category_totals.values())

    category_pcts = {}
    if grand_total > 0:
        category_pcts = {k: (v / grand_total) * 100 for k, v in category_totals.items()}

    print(f"\n{'='*50}")
    print(f"LATENCY DECOMPOSITION (% of total)")
    print(f"{'='*50}")
    for cat, pct in sorted(category_pcts.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s}: {pct:6.2f}%  ({category_totals[cat]:.2f}ms total)")
    print(f"  {'TOTAL':25s}: {grand_total:.2f}ms")
    print(f"  {'Per-token avg':25s}: {grand_total/actual_tokens:.2f}ms")

    # Clean up
    layer_timer.remove_hooks()

    return {
        "model": model_id,
        "device": device,
        "platform": get_platform_label(),
        "prompt_tokens": input_ids.shape[1],
        "tokens_generated": actual_tokens,
        "category_totals_ms": category_totals,
        "category_percentages": category_pcts,
        "per_step_data": per_step_data,
        "aggregated": layer_timer.get_aggregated_breakdown(),
    }


def main():
    parser = argparse.ArgumentParser(description="LLaMA Latency Decomposition")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_HF_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_HF_MODEL})",
    )
    parser.add_argument("--tokens", type=int, default=64, help="Tokens to generate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--prompt", type=str,
        default="Explain the concept of attention in transformer models.",
        help="Prompt text",
    )
    prefix = get_output_prefix()
    parser.add_argument("--output", type=str, default=f"decomposition_{prefix}.json")
    args = parser.parse_args()

    print(f"Platform: {detect_platform()}")

    results = run_decomposition(
        model_id=args.model,
        num_tokens=args.tokens,
        prompt_text=args.prompt,
        device=args.device,
    )

    # Save results
    save_results_json(results, args.output)

    # Save per-step data as CSV for plotting
    if results["per_step_data"]:
        csv_filename = args.output.replace(".json", "_steps.csv")
        save_results_csv(results["per_step_data"], csv_filename)


if __name__ == "__main__":
    main()
