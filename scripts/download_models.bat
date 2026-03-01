@echo off
REM Download GGUF models from bartowski (no Meta approval needed)
REM Run: scripts\download_models.bat

set MODELS_DIR=models
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

echo === Downloading LLaMA 3.2 1B GGUF models ===

echo Downloading Q4_K_M...
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" --local-dir "%MODELS_DIR%"

echo Downloading Q8_0...
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF --include "Llama-3.2-1B-Instruct-Q8_0.gguf" --local-dir "%MODELS_DIR%"

echo Downloading F16...
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF --include "Llama-3.2-1B-Instruct-f16.gguf" --local-dir "%MODELS_DIR%"

echo.
echo === Downloading LLaMA 3.2 3B GGUF model (for scaling analysis) ===

echo Downloading 3B Q4_K_M...
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" --local-dir "%MODELS_DIR%"

echo Downloading 3B Q8_0...
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF --include "Llama-3.2-3B-Instruct-Q8_0.gguf" --local-dir "%MODELS_DIR%"

echo.
echo === All models downloaded to %MODELS_DIR%\ ===
dir /B "%MODELS_DIR%\*.gguf" 2>nul || echo No .gguf files found in %MODELS_DIR%
