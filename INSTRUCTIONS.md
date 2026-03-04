# Running Instructions

## Prerequisites

- Python 3.10+
- ~10 GB disk space for models
- HuggingFace account with Meta LLaMA 3.2 access (only needed for Goal 2 — latency decomposition; all other goals work without it)

---

## Platform Setup

### Mac (Apple Silicon)

```bash
bash scripts/setup_mac.sh
source .venv/bin/activate
```

This installs llama-cpp-python with the **Metal** backend for GPU-accelerated inference on Apple Silicon.

### Windows (Intel CPU)

```cmd
scripts\setup_windows.bat
.venv\Scripts\activate
```

This installs llama-cpp-python with **AVX2** support for x86 CPU inference.

### Google Colab (NVIDIA GPU)

Any team member can run this — no local setup required, just a Google account.

1. Open `analysis/notebooks/colab_gpu_benchmark.ipynb` in Google Colab
2. Go to **Runtime > Change runtime type > GPU** (T4 is free, A100 requires Colab Pro)
3. Run all cells top-to-bottom — the notebook handles everything (clone, install, download models, benchmark, plot, download results)

Output files will be auto-tagged with the GPU type (e.g., `benchmark_gpu_T4_20260301_...csv`).

Alternatively, from the terminal:

```bash
bash scripts/setup_colab.sh
```

---

## Team Assignments

Each team member runs benchmarks on their own platform, then runs analysis + plots locally on their own results.

| Person | Platform | What to Run |
|--------|----------|-------------|
| Akash | Mac (M4 Pro) | Goals 1, 2, 3 → then Goal 4 + plots |
| Teammate 2 | Windows (Intel CPU) | Goals 1, 3 → then Goal 4 + plots |
| Teammate 3 | Mac (M4) | Goals 1, 3 → then Goal 4 + plots |
| Any one person | Google Colab (GPU) | Goals 1, 3 → then Goal 4 + plots |

**Notes:**
- **Goal 2 (Latency Decomposition):** Akash only — requires a HuggingFace token with Meta LLaMA 3.2 access.
- **Colab GPU:** Any team member can run the notebook (`analysis/notebooks/colab_gpu_benchmark.ipynb`) — just needs a Google account.
- **Goal 4 + Plots:** Everyone runs these on their own results after finishing Goals 1 and 3.
- All output files are auto-tagged with platform and timestamp (e.g., `benchmark_mac_arm64_...`, `benchmark_windows_x86_...`) so nothing overwrites.

---

## Step 1: Download Models

GGUF models from bartowski (free, no approval needed):

**Mac/Linux:**
```bash
bash scripts/download_models.sh
```

**Windows:**
```cmd
scripts\download_models.bat
```

This downloads:
| Model | Quant | Size |
|-------|-------|------|
| LLaMA 3.2 1B | Q4_K_M | ~0.7 GB |
| LLaMA 3.2 1B | Q8_0 | ~1.1 GB |
| LLaMA 3.2 1B | F16 | ~2.2 GB |
| LLaMA 3.2 3B | Q4_K_M | ~1.8 GB |
| LLaMA 3.2 3B | Q8_0 | ~3.2 GB |

---

## Step 2: Smoke Test

Verify everything works with a quick test (< 1 minute):

```bash
python -m benchmarks.benchmark_harness \
    --model llama-3.2-1b-q4 \
    --prompt-length 128 \
    --trials 1 \
    --warmup 1 \
    --output-tokens 32 \
    --output smoke_test.csv
```

---

## Step 3: Run Benchmarks

### Goal 1 -- Benchmark Harness

Measures TTFT, per-token latency, and end-to-end time.

```bash
# Single model, all prompt lengths
python -m benchmarks.benchmark_harness --model llama-3.2-1b-q4 --trials 10

# All models
python -m benchmarks.benchmark_harness --all-models --trials 10

# Specific prompt length
python -m benchmarks.benchmark_harness --model llama-3.2-1b-q4 --prompt-length 512 --trials 10
```

**Output:** `data/raw/benchmark_{platform}_{timestamp}.csv` + `.json`
Example: `benchmark_mac_arm64_20260228_210001.csv`

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `llama-3.2-1b-q4` | Model key from config |
| `--prompt-length` | all (128, 256, 512, 1024) | Single prompt length to test |
| `--trials` | 10 | Measurement trials per config |
| `--warmup` | 3 | Warm-up runs (discarded) |
| `--output-tokens` | 128 | Tokens to generate per trial |
| `--all-models` | false | Run all configured models |
| `--output` | auto-tagged (see below) | Output filename (override with explicit name) |

### Goal 2 -- Latency Decomposition (Optional)

Per-layer timing using PyTorch forward hooks on the HuggingFace model. Any team member can run this on any platform (Mac, Windows, or Colab).

**Requires:** Meta LLaMA 3.2 access on HuggingFace. Whoever runs this needs to:
1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Run `huggingface-cli login` and paste their token

```bash
# CPU (Mac/Windows)
python -m benchmarks.latency_decomposition --device cpu --tokens 64

# GPU (Colab)
python -m benchmarks.latency_decomposition --device cuda --tokens 64
```

**Output:** `data/raw/decomposition_{platform}_{timestamp}.json` + `_steps.csv`
Example: `decomposition_mac_arm64_20260228_211500.json`

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `meta-llama/Llama-3.2-1B-Instruct` | HuggingFace model ID |
| `--tokens` | 64 | Tokens to generate |
| `--device` | `auto` | Device: auto, cpu, cuda |
| `--prompt` | (built-in) | Custom prompt text |
| `--output` | auto-tagged (see below) | Output filename (override with explicit name) |

### Goal 3 -- Scaling Analysis

Three experiments: sequence length, model size, and quantization scaling.

```bash
# All three experiments
python -m benchmarks.scaling_analysis --experiment all --trials 5

# Individual experiments
python -m benchmarks.scaling_analysis --experiment sequence --trials 5
python -m benchmarks.scaling_analysis --experiment model_size --trials 5
python -m benchmarks.scaling_analysis --experiment quantization --trials 5
```

**Output:** `data/raw/scaling_{platform}_{timestamp}_*.csv` + `_all.json`
Example: `scaling_mac_arm64_20260228_211500_sequence.csv`

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `all` | Which experiment(s) to run |
| `--trials` | 10 | Trials per configuration |
| `--prompt-length` | 512 | Base prompt length (for model_size/quantization) |
| `--output` | auto-tagged (see below) | Output filename prefix (override with explicit name) |

### Goal 4 -- Bottleneck Analysis

Maps measurements to architectural causes. Can run with or without benchmark data.

```bash
# Default (1B, F16, seq_len=512)
python -m analysis.bottleneck_analysis

# Custom config
python -m analysis.bottleneck_analysis --model 1B --quant Q4_K_M --seq-len 1024
python -m analysis.bottleneck_analysis --model 3B --quant Q8_0
```

**Output:** `results/bottleneck_{platform}_{timestamp}.txt` + `.json`
Example: `bottleneck_mac_arm64_20260228_211500.txt`

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `1B` | Architecture: 1B or 3B |
| `--quant` | `F16` | Quantization: Q4_K_M, Q8_0, F16 |
| `--seq-len` | 512 | Sequence length for analysis |
| `--data-dir` | `data/raw` | Directory with benchmark CSVs |
| `--output` | auto | Output file path |

---

## Step 4: Generate Plots

```bash
python -m analysis.plot_results
```

**Output:** Auto-tagged PNG plots in `results/`
Example: `scaling_sequence_length_mac_arm64_20260228_211500.png`

Generated plots: sequence scaling, model comparison, quantization impact, per-token timeline, decomposition breakdown.

---

## Recommended Run Order (Full Benchmark)

```bash
# 1. Benchmark harness (all models) -- ~30-60 min
python -m benchmarks.benchmark_harness --all-models --trials 10

# 2. Latency decomposition (optional, needs HF token) -- ~5-10 min
python -m benchmarks.latency_decomposition --tokens 64

# 3. Scaling analysis -- ~30-60 min
python -m benchmarks.scaling_analysis --experiment all --trials 5

# 4. Bottleneck analysis (uses data from steps 1-3)
python -m analysis.bottleneck_analysis --model 1B --quant Q4_K_M

# 5. Generate plots
python -m analysis.plot_results
```

Steps 1, 3, 4, 5 work on any platform without a HuggingFace token. Step 2 can be run by any team member who has HF access.

---

## Quick Run (Reduced Trials)

For testing or time-constrained runs:

```bash
python -m benchmarks.benchmark_harness --model llama-3.2-1b-q4 --trials 3 --warmup 1
python -m benchmarks.latency_decomposition --tokens 32
python -m benchmarks.scaling_analysis --experiment all --trials 2
python -m analysis.bottleneck_analysis
python -m analysis.plot_results
```

---

## Output File Naming

All benchmark scripts **auto-tag** output files with platform and timestamp so that results from different machines and runs never overwrite each other.

**Format:** `{benchmark|decomposition|scaling}_{platform}_{YYYYMMDD_HHMMSS}.{csv|json}`

**Platform tags:**

| Platform | Tag |
|----------|-----|
| Mac (Apple Silicon) | `mac_arm64` |
| Windows (Intel CPU) | `windows_x86` |
| Colab (NVIDIA T4) | `gpu_T4` |
| Colab (NVIDIA A100) | `gpu_A100` |

**Examples (`data/raw/`):**
```
data/raw/benchmark_mac_arm64_20260228_210001.csv
data/raw/scaling_windows_x86_20260301_143022_sequence.csv
data/raw/decomposition_gpu_T4_20260302_091500.json
```

**Examples (`results/`):**
```
results/bottleneck_mac_arm64_20260228_211500.txt
results/scaling_sequence_length_mac_arm64_20260228_211500.png
results/token_timeline_llama-3.2-1b-q4_gpu_T4_20260302_091500.png
```

The analysis scripts (`plot_results.py`, `bottleneck_analysis.py`) **automatically find the most recent** matching data file in `data/raw/`, so you don't need to specify filenames manually. All output files (data + results) are auto-tagged so results from different platforms and runs never collide. You can override with `--output <name>` if needed.

---

## Output Directories

| Directory | Contents |
|-----------|----------|
| `data/raw/` | Raw CSV/JSON benchmark data (auto-tagged per platform/run) |
| `results/` | Generated plots (PNG) and analysis reports (auto-tagged per platform/run) |
| `models/` | Downloaded GGUF model files (gitignored) |

---

## Troubleshooting

**"No module named benchmarks"** -- Run from the project root directory.

**llama-cpp-python build fails on Mac** -- Ensure Xcode Command Line Tools are installed: `xcode-select --install`

**CUDA not detected on Colab** -- Go to Runtime > Change runtime type > GPU.

**HuggingFace gated model error** -- Run `huggingface-cli login` and ensure you have Meta LLaMA 3.2 access at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

**Out of memory on 3B F16** -- Use Q4_K_M or Q8_0 quantization instead.
