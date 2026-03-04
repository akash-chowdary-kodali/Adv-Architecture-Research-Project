"""
Utility functions for benchmarking:
- Platform detection
- Timing helpers (CPU and GPU)
- Statistical analysis (IQR filtering, summary stats)
- Model loading helpers
"""

import os
import platform
import time
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

from benchmarks.config import MODELS, MODELS_DIR, DATA_DIR, PROMPT_SEED_TEXT


# ============================================================
# Platform Detection
# ============================================================

def detect_platform() -> Dict[str, str]:
    """Detect current hardware platform and capabilities."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": str(os.cpu_count()),
    }

    # Detect Apple Silicon
    if info["os"] == "Darwin" and info["machine"] == "arm64":
        info["platform_type"] = "apple_silicon"
        info["accelerator"] = "Metal"
    elif info["os"] == "Windows" or (info["os"] == "Linux" and "x86_64" in info["machine"]):
        info["platform_type"] = "x86_cpu"
        info["accelerator"] = "AVX2"
    else:
        info["platform_type"] = "unknown"
        info["accelerator"] = "none"

    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info["platform_type"] = "cuda_gpu"
            info["accelerator"] = "CUDA"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_mb"] = str(
                torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            )
    except ImportError:
        pass

    return info


def _get_chip_model() -> str:
    """Return a short filesystem-safe chip identifier for the current processor.

    Mac Apple Silicon: 'Apple M4 Pro' → 'm4pro', 'Apple M4' → 'm4'
    Other platforms: returns empty string (chip already encoded via GPU name or OS tag).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        )
        brand = result.stdout.strip()  # e.g. "Apple M4 Pro"
        if brand.startswith("Apple "):
            chip = brand[6:].replace(" ", "").lower()  # "M4 Pro" → "m4pro"
            return chip
    except Exception:
        pass
    return ""


def get_platform_label() -> str:
    """Return a short human-readable label for the current platform (used in CSV output)."""
    info = detect_platform()
    if info["platform_type"] == "cuda_gpu":
        return f"GPU-{info.get('gpu_name', 'unknown')}"
    elif info["platform_type"] == "apple_silicon":
        chip = _get_chip_model()
        return f"Mac-{chip.upper()}" if chip else "Mac-AppleSilicon"
    elif info["platform_type"] == "x86_cpu":
        return f"{info['os']}-x86"
    return "unknown"


def get_platform_tag() -> str:
    """Return a short filesystem-safe tag for output filenames.

    Examples: 'mac_arm64_m4pro', 'mac_arm64_m4', 'windows_x86', 'gpu_T4', 'gpu_A100'
    """
    info = detect_platform()
    if info["platform_type"] == "cuda_gpu":
        gpu = info.get("gpu_name", "unknown").split()[-1]  # e.g. "T4", "A100"
        return f"gpu_{gpu}"
    elif info["platform_type"] == "apple_silicon":
        chip = _get_chip_model()
        return f"mac_arm64_{chip}" if chip else "mac_arm64"
    elif info["platform_type"] == "x86_cpu":
        return f"{info['os'].lower()}_x86"
    return "unknown"


def get_output_prefix() -> str:
    """Return a platform + timestamp prefix for output filenames.

    Example: 'mac_arm64_m4pro_20260303_204532'
    """
    from datetime import datetime
    tag = get_platform_tag()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{tag}_{ts}"


# ============================================================
# Timing Helpers
# ============================================================

class CPUTimer:
    """High-resolution CPU timer using perf_counter_ns."""

    def __init__(self):
        self._start = 0

    def start(self):
        self._start = time.perf_counter_ns()

    def stop(self) -> float:
        """Returns elapsed time in milliseconds."""
        elapsed_ns = time.perf_counter_ns() - self._start
        return elapsed_ns / 1_000_000  # Convert to ms


class GPUTimer:
    """GPU timer using torch.cuda.Event for accurate CUDA timing."""

    def __init__(self):
        import torch
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._start_event.record()

    def stop(self) -> float:
        """Returns elapsed time in milliseconds."""
        import torch
        self._end_event.record()
        torch.cuda.synchronize()
        return self._start_event.elapsed_time(self._end_event)


def get_timer():
    """Return the appropriate timer for the current platform."""
    try:
        import torch
        if torch.cuda.is_available():
            return GPUTimer()
    except ImportError:
        pass
    return CPUTimer()


# ============================================================
# Statistical Analysis
# ============================================================

def filter_outliers_iqr(data: List[float], factor: float = 1.5) -> List[float]:
    """Remove outliers using IQR method."""
    if len(data) < 4:
        return data
    arr = np.array(data)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    filtered = arr[(arr >= lower) & (arr <= upper)]
    return filtered.tolist()


def compute_stats(data: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a list of measurements."""
    if not data:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}
    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


# ============================================================
# Prompt Generation
# ============================================================

def generate_prompt_tokens(tokenizer, target_length: int) -> List[int]:
    """Generate a prompt of approximately target_length tokens by repeating seed text."""
    # Tokenize the seed text once
    seed_tokens = tokenizer.encode(PROMPT_SEED_TEXT)
    if not seed_tokens:
        raise ValueError("Seed text produced no tokens")

    # Repeat to reach target length
    repeated = []
    while len(repeated) < target_length:
        repeated.extend(seed_tokens)

    # Truncate to exact target length
    return repeated[:target_length]


def generate_prompt_text(target_length: int, tokens_per_word: float = 1.3) -> str:
    """Generate prompt text that approximates target_length tokens.
    Used when we don't have a tokenizer (e.g., llama-cpp-python high-level API).
    """
    words = PROMPT_SEED_TEXT.split()
    target_words = int(target_length / tokens_per_word)

    repeated = []
    while len(repeated) < target_words:
        repeated.extend(words)

    return " ".join(repeated[:target_words])


# ============================================================
# Model Path Helpers
# ============================================================

def get_gguf_model_path(model_key: str) -> str:
    """Return local path to a GGUF model file."""
    model_cfg = MODELS[model_key]
    path = os.path.join(MODELS_DIR, model_cfg["gguf_file"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Run: huggingface-cli download {model_cfg['gguf_repo']} "
            f"--include \"{model_cfg['gguf_file']}\" --local-dir {MODELS_DIR}"
        )
    return path


# ============================================================
# Data I/O
# ============================================================

def save_results_csv(data: List[Dict], filename: str):
    """Save benchmark results to CSV in the data/raw directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    if not data:
        print(f"Warning: No data to save to {filepath}")
        return

    fieldnames = data[0].keys()
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Results saved to {filepath}")


def save_results_json(data, filename: str):
    """Save benchmark results to JSON in the data/raw directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filepath}")


def load_results_csv(filename: str) -> List[Dict]:
    """Load benchmark results from CSV."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)
