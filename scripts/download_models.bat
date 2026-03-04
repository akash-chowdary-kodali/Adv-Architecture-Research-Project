@echo off
REM Download GGUF models from bartowski (no Meta approval needed)
REM Run: scripts\download_models.bat

set MODELS_DIR=models
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

echo === Downloading LLaMA 3.2 GGUF models ===

REM Write a temp Python script and run it
set TMPSCRIPT=%TEMP%\download_models_tmp.py
(
echo from huggingface_hub import hf_hub_download
echo import os
echo MODELS_DIR = "models"
echo os.makedirs^(MODELS_DIR, exist_ok=True^)
echo downloads = [
echo     ^("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"^),
echo     ^("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"^),
echo     ^("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-f16.gguf"^),
echo     ^("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"^),
echo     ^("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q8_0.gguf"^),
echo ]
echo for repo_id, filename in downloads:
echo     dest = os.path.join^(MODELS_DIR, filename^)
echo     if os.path.exists^(dest^):
echo         print^(f"  Skipping {filename} ^(already exists^)"^)
echo         continue
echo     print^(f"Downloading {filename}..."^)
echo     hf_hub_download^(repo_id=repo_id, filename=filename, local_dir=MODELS_DIR^)
echo     print^(f"  Done."^)
echo print^("\nAll models ready."^)
) > "%TMPSCRIPT%"

python "%TMPSCRIPT%"
del "%TMPSCRIPT%"

echo.
echo === Models in %MODELS_DIR%\ ===
dir /B "%MODELS_DIR%\*.gguf" 2>nul || echo No .gguf files found in %MODELS_DIR%
