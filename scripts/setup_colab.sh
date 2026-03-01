#!/bin/bash
# Setup script for Google Colab (CUDA GPU)
# Run this in a Colab notebook cell: !bash scripts/setup_colab.sh

set -e

echo "=== Setting up LLaMA Benchmark (Google Colab GPU) ==="

# Install llama-cpp-python with CUDA support (pre-built wheel)
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Install other dependencies
pip install transformers accelerate huggingface-hub
pip install numpy pandas matplotlib seaborn scipy
pip install bitsandbytes  # Only works on CUDA

# Download models
bash scripts/download_models.sh

echo ""
echo "=== Setup complete! ==="
echo "Run benchmark:        !python -m benchmarks.benchmark_harness"
echo "Run decomposition:    !python -m benchmarks.latency_decomposition --device cuda"
