@echo off
REM Setup script for Windows (Intel CPU with AVX2)
REM Run: scripts\setup_windows.bat

echo === Setting up LLaMA Benchmark (Windows Intel CPU) ===

REM Create virtual environment
python -m venv venv
call venv\Scripts\activate

REM Install base dependencies
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy jupyter ipykernel
pip install torch transformers accelerate huggingface-hub

REM Install llama-cpp-python (AVX2 enabled by default on x86)
echo Installing llama-cpp-python...
pip install llama-cpp-python

REM Download models
echo Downloading GGUF models...
call scripts\download_models.bat

echo.
echo === Setup complete! ===
echo Activate environment: venv\Scripts\activate
echo Run benchmark:        python -m benchmarks.benchmark_harness
echo Run decomposition:    python -m benchmarks.latency_decomposition
