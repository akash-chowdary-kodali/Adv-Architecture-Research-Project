#!/bin/bash
# Download GGUF models from bartowski (no Meta approval needed)
# Run: bash scripts/download_models.sh

set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

echo "=== Downloading LLaMA 3.2 GGUF models ==="

python - <<'EOF'
from huggingface_hub import hf_hub_download
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

downloads = [
    # 1B models
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),  # ~0.7 GB
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),    # ~1.1 GB
    ("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-f16.gguf"),     # ~2.2 GB
    # 3B models (for scaling analysis)
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"), # ~1.8 GB
    ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q8_0.gguf"),   # ~3.2 GB
]

for repo_id, filename in downloads:
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.exists(dest):
        print(f"  Skipping {filename} (already exists)")
        continue
    print(f"Downloading {filename}...")
    hf_hub_download(repo_id=repo_id, filename=filename, local_dir=MODELS_DIR)
    print(f"  Done.")

print("\nAll models ready.")
EOF

echo ""
echo "=== Models in $MODELS_DIR/ ==="
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "No .gguf files found in $MODELS_DIR"
