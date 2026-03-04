"""
Central configuration for all benchmark experiments.
Modify these values to adjust benchmark parameters.
"""

# --- Model Configurations ---
MODELS = {
    "llama-3.2-1b-q4": {
        "name": "LLaMA-3.2-1B-Q4_K_M",
        "gguf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "gguf_file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "quantization": "Q4_K_M",
        "params": "1B",
    },
    "llama-3.2-1b-q8": {
        "name": "LLaMA-3.2-1B-Q8_0",
        "gguf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "gguf_file": "Llama-3.2-1B-Instruct-Q8_0.gguf",
        "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "quantization": "Q8_0",
        "params": "1B",
    },
    "llama-3.2-1b-f16": {
        "name": "LLaMA-3.2-1B-F16",
        "gguf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "gguf_file": "Llama-3.2-1B-Instruct-f16.gguf",
        "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "quantization": "F16",
        "params": "1B",
    },
    "llama-3.2-3b-q4": {
        "name": "LLaMA-3.2-3B-Q4_K_M",
        "gguf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "gguf_file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "hf_repo": "meta-llama/Llama-3.2-3B-Instruct",
        "quantization": "Q4_K_M",
        "params": "3B",
    },
    "llama-3.2-3b-q8": {
        "name": "LLaMA-3.2-3B-Q8_0",
        "gguf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "gguf_file": "Llama-3.2-3B-Instruct-Q8_0.gguf",
        "hf_repo": "meta-llama/Llama-3.2-3B-Instruct",
        "quantization": "Q8_0",
        "params": "3B",
    },
}

# HuggingFace model for latency decomposition (requires Meta approval)
DEFAULT_HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# --- Benchmark Parameters ---
PROMPT_LENGTHS = [128, 256, 512, 1024]
OUTPUT_LENGTH = 128
BATCH_SIZE = 1

# Number of warm-up runs (discarded)
WARMUP_RUNS = 3
# Number of measurement trials
NUM_TRIALS = 10

# Context window size for llama-cpp-python
# Must exceed max prompt length + output length (2048 + 128 = 2176 minimum)
CONTEXT_SIZE = 4096

# --- Scaling Analysis Parameters ---
SCALING_SEQUENCE_LENGTHS = [128, 256, 512, 1024, 2048]
SCALING_MODELS = ["llama-3.2-1b-q4", "llama-3.2-3b-q4"]
SCALING_QUANTIZATIONS = ["llama-3.2-1b-q4", "llama-3.2-1b-q8", "llama-3.2-1b-f16"]

# --- Paths ---
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# --- Standard Prompt Text ---
# A repeatable prompt passage used across all benchmarks.
# We repeat it to reach desired token counts.
PROMPT_SEED_TEXT = (
    "The transformer architecture has revolutionized natural language processing "
    "since its introduction in 2017. The key innovation is the self-attention mechanism, "
    "which allows the model to weigh the importance of different parts of the input sequence "
    "when producing each output element. Unlike recurrent neural networks, transformers "
    "can process all positions in parallel during training, leading to significant speedups. "
    "During inference, however, autoregressive generation requires sequential token production, "
    "making per-token latency a critical performance metric. The KV-cache stores previously "
    "computed key and value tensors to avoid redundant computation, but this cache grows "
    "linearly with sequence length and can become a memory bandwidth bottleneck. "
    "Understanding these architectural trade-offs is essential for optimizing LLM inference "
    "performance across different hardware platforms. "
)
