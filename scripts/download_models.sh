#!/bin/bash
# Download GGUF models from bartowski (no Meta approval needed)
# Run: bash scripts/download_models.sh

set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

echo "=== Downloading LLaMA 3.2 1B GGUF models ==="

# Q4_K_M (~0.7GB) - smallest, fastest
echo "Downloading Q4_K_M..."
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
    --local-dir "$MODELS_DIR"

# Q8_0 (~1.1GB) - medium precision
echo "Downloading Q8_0..."
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    --include "Llama-3.2-1B-Instruct-Q8_0.gguf" \
    --local-dir "$MODELS_DIR"

# F16 (~2.2GB) - full precision
echo "Downloading F16..."
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
    --include "Llama-3.2-1B-Instruct-f16.gguf" \
    --local-dir "$MODELS_DIR"

echo ""
echo "=== Downloading LLaMA 3.2 3B GGUF model (for scaling analysis) ==="

# 3B Q4_K_M (~1.8GB) - for model size comparison
echo "Downloading 3B Q4_K_M..."
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
    --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
    --local-dir "$MODELS_DIR"

# 3B Q8_0 (~3.2GB) - for quantization comparison at 3B scale
echo "Downloading 3B Q8_0..."
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
    --include "Llama-3.2-3B-Instruct-Q8_0.gguf" \
    --local-dir "$MODELS_DIR"

echo ""
echo "=== All models downloaded to $MODELS_DIR/ ==="
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "No .gguf files found in $MODELS_DIR"
