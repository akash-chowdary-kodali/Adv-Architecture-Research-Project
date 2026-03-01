#!/bin/bash
# Setup script for Mac (Apple Silicon with Metal backend)
# Run: bash scripts/setup_mac.sh

set -e

echo "=== Setting up LLaMA Benchmark (Mac Apple Silicon) ==="

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install base dependencies
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy jupyter ipykernel
pip install torch transformers accelerate huggingface-hub

# Install llama-cpp-python with Metal backend
echo "Installing llama-cpp-python with Metal support..."
CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall --no-cache-dir llama-cpp-python

# Download models
echo "Downloading GGUF models..."
bash scripts/download_models.sh

echo ""
echo "=== Setup complete! ==="
echo "Activate environment: source venv/bin/activate"
echo "Run benchmark:        python -m benchmarks.benchmark_harness"
echo "Run decomposition:    python -m benchmarks.latency_decomposition"
