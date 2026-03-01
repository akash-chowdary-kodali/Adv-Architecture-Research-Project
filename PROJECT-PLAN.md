# Project 6: Token-Generation Latency Benchmarking in LLaMA

## Context
Advanced Computer Architecture course research project. Team of 3 members benchmarking token-generation latency in LLaMA-style models, decomposing latency into architectural components, and proposing hardware/system-level improvements. Requires 6+ research papers (2 per person), a final presentation, and a survey paper.

---

## Team Hardware

| Machine | Platform | Use Case |
|---------|----------|----------|
| Windows Intel | x86 CPU | Intel CPU baseline, cache hierarchy analysis |
| Mac M-chip | ARM (Apple Silicon) | Unified memory architecture, KV-cache analysis |
| Mac (TBD chip) | TBD | Secondary testing platform |
| Google Colab | NVIDIA T4/A100 GPU | GPU benchmarking, CUDA profiling |

---

## Research Papers (Verified)

### Person A — Attention & Memory Hierarchy
1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   - Authors: Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, Christopher Ré
   - Venue: NeurIPS 2022
   - PDF: https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html
   - arXiv: https://arxiv.org/abs/2205.14135
   - Key: IO-aware attention using tiling to reduce HBM-SRAM transfers. 3x speedup on GPT-2.

2. **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
   - Authors: Tri Dao
   - Venue: ICLR 2024
   - PDF: https://proceedings.iclr.cc/paper_files/paper/2024/file/98ed250b203d1ac6b24bbcf263e3d4a7-Paper-Conference.pdf
   - OpenReview: https://openreview.net/forum?id=mZn2Xyh9Ec
   - Key: Improved GPU parallelism and work partitioning for attention computation.

### Person B — KV-Cache & Inference Serving
3. **Efficient Memory Management for Large Language Model Serving with PagedAttention**
   - Authors: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, et al.
   - Venue: SOSP 2023
   - PDF: https://dl.acm.org/doi/10.1145/3600006.3613165
   - arXiv: https://arxiv.org/abs/2309.06180
   - Key: PagedAttention for KV-cache memory management. vLLM achieves 2-4x throughput improvement.

4. **ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference**
   - Authors: Hyungjun Oh, Kihong Kim, Jaemin Kim, Sungkyun Kim, Junyeol Lee, Du-Seong Chang, Jiwon Seo
   - Venue: ASPLOS 2024
   - PDF: https://dl.acm.org/doi/10.1145/3620665.3640383
   - arXiv: https://arxiv.org/abs/2404.07947
   - Key: Optimal scheduling for latency-constrained inference. Up to 15.2x throughput over FasterTransformer.

### Person C — Hardware Architecture & Optimization
5. **LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference**
   - Authors: Hengrui Zhang, August Ning, Rohan Baskar Prabhakar, David Wentzlaff
   - Venue: ISCA 2024
   - PDF: https://dl.acm.org/doi/10.1109/ISCA59077.2024.00082
   - Direct PDF: https://augustning.com/assets/papers/llmcompass-isca-2024.pdf
   - Key: Hardware evaluation framework. 4.1% error rate for LLM inference latency estimation. Shows inference is IO-bound.

6. **DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency**
   - Authors: Jovan Stojkovic, Chaojie Zhang, Inigo Goiri, Josep Torrellas, Esha Choukse
   - Venue: HPCA 2025 (Best Paper Award)
   - PDF: https://iacoma.cs.uiuc.edu/iacoma-papers/hpca25_2.pdf
   - IEEE Xplore: https://ieeexplore.ieee.org/xpl/conhome/10946287/proceeding (DOI: 10.1109/HPCA61900.2025.00102)
   - arXiv: https://arxiv.org/abs/2408.00741
   - Key: Energy-management framework. 53% energy savings, 61% cost reduction while meeting latency SLOs.

### Bonus / Shared Papers
7. **InstAttention: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference**
   - Authors: Xiurui Pan, Endian Li, Qiao Li, Shengwen Liang, Yizhou Shan, Ke Zhou, Yingwei Luo, Xiaolin Wang, Jie Zhang
   - Venue: HPCA 2025
   - IEEE Xplore: https://ieeexplore.ieee.org/document/10946721 (DOI: 10.1109/HPCA61900.2025.00113, pp. 1510-1525)
   - arXiv preprint (named "InstInfer"): https://arxiv.org/abs/2409.04992
   - Key: Offloads attention + KV-cache to Computational Storage Drives (CSDs).

8. **Fast Inference from Transformers via Speculative Decoding**
   - Authors: Yaniv Leviathan, Matan Kalman, Yossi Matias
   - Venue: ICML 2023 (Oral)
   - PDF: https://proceedings.mlr.press/v202/leviathan23a/leviathan23a.pdf
   - Proceedings: https://proceedings.mlr.press/v202/leviathan23a.html
   - arXiv: https://arxiv.org/abs/2211.17192
   - Key: 2-3x speedup via draft model + verification. Relevant to bonus section.

### Conference Distribution
- HPCA: 3 papers (6, 7 + DynamoLLM best paper)
- ASPLOS: 1 paper (4)
- ISCA: 1 paper (5)
- ICLR: 1 paper (2)
- NeurIPS: 1 paper (1) — confirmed acceptable by professor
- SOSP: 1 paper (3) — confirmed acceptable by professor
- ICML: 1 paper (8) — confirmed acceptable by professor

---

## Paper-to-Project-Goal Mapping

| Project Goal | Relevant Papers |
|---|---|
| Goal 1: Benchmark Harness | Papers 4 (ExeGPT scheduling), 5 (LLMCompass methodology) |
| Goal 2: Latency Decomposition | Papers 1, 2 (attention), 3 (KV-cache), 7 (storage offloading) |
| Goal 3: Scaling Analysis | Papers 5 (hardware scaling), 6 (cluster scaling) |
| Goal 4: Bottleneck Analysis | Papers 1 (IO-awareness), 5 (IO-bound analysis), 7 (storage bottleneck) |
| Goal 5: Optimization Proposal | Papers 2 (parallelism), 3 (PagedAttention), 8 (speculative decoding) |
| Bonus: Cross-platform | All — Intel vs M-chip vs GPU comparison |
| Bonus: Energy-per-token | Paper 6 (DynamoLLM) |

---

## Team Responsibility Split

| Person | Papers | Project Goals | Machine |
|--------|--------|---------------|---------|
| Person A | 1, 2 (FlashAttention 1 & 2) | Goal 1 (Benchmark Harness) + Goal 3 (Scaling Analysis) | Windows Intel |
| Person B | 3, 4 (PagedAttention, ExeGPT) | Goal 2 (Latency Decomposition) + Goal 4 (Bottleneck Analysis) | Mac M-chip |
| Person C | 5, 6 (LLMCompass, DynamoLLM) | Goal 5 (Optimization Proposal) + Bonus work | Colab GPU |

---

## Implementation Plan

### Approach: Python-Based with Two Backends
1. **llama-cpp-python** (primary): Python bindings for llama.cpp — used for Goals 1, 3 (benchmark harness, scaling). Works cross-platform with GGUF models.
2. **HuggingFace Transformers** (secondary): Used for Goal 2 (latency decomposition). PyTorch forward hooks enable per-component timing.

### Model: LLaMA 3.2 1B Instruct (GGUF)
- Source: `bartowski/Llama-3.2-1B-Instruct-GGUF` (freely available, no Meta approval needed)
- Quantizations: Q4_K_M (~0.7GB), Q8_0 (~1.1GB), F16 (~2.2GB)
- Also: LLaMA 3.2 3B Q4_K_M (~1.8GB) + Q8_0 (~3.2GB) for scaling comparison
- Fallback for HF decomposition: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (ungated)

### Platform Compatibility

| Component | Mac M-chip | Windows Intel (CPU) | Google Colab (GPU) |
|-----------|-----------|-------------------|-------------------|
| llama-cpp-python | Metal backend | AVX2 backend | CUDA backend |
| GGUF models (bartowski) | Free download | Free download | Free download |
| HuggingFace Transformers | CPU/MPS | CPU | CUDA |
| Timing method | perf_counter_ns() | perf_counter_ns() | torch.cuda.Event |

### Project Structure
```
Adv-Architecture-Research-Project/
├── PROJECT-PLAN.md
├── requirements.txt
├── benchmarks/
│   ├── config.py                  # Central configuration
│   ├── benchmark_harness.py       # Goal 1: TTFT, per-token, e2e timing
│   ├── latency_decomposition.py   # Goal 2: Per-layer PyTorch hook timing
│   ├── scaling_analysis.py        # Goal 3: Sequence/model/quantization scaling
│   └── utils.py                   # Platform detection, timers, stats, I/O
├── analysis/
│   ├── plot_results.py            # Generate all plots (matplotlib/seaborn)
│   └── bottleneck_analysis.py     # Goal 4: Map measurements to architecture
├── data/
│   └── raw/                       # Raw CSV/JSON timing data
├── models/                        # Downloaded GGUF model files (gitignored)
├── results/                       # Generated plots and tables
└── scripts/
    ├── setup_mac.sh               # Mac setup (Metal backend)
    ├── setup_windows.bat          # Windows setup (AVX2 backend)
    ├── setup_colab.sh             # Colab setup (CUDA backend)
    └── download_models.sh         # Download GGUF models from bartowski
```

### Phase 1: Setup & Model Download
- Create virtual environment, install dependencies (`pip install -r requirements.txt`)
- Mac: `CMAKE_ARGS="-DGGML_METAL=on" pip install --force-reinstall llama-cpp-python`
- Download GGUF models: `bash scripts/download_models.sh`
- Verify inference: load model with llama-cpp-python, generate 10 tokens

### Phase 2: Benchmark Harness (Goal 1) — `benchmarks/benchmark_harness.py`
- Uses llama-cpp-python low-level API for precise timing
- Measures TTFT (prompt eval), per-token decode latency, end-to-end time
- 3 warm-up + 10 measurement trials per configuration
- IQR-based outlier filtering, computes mean/median/std/p95/p99
- Prompt lengths: 128, 256, 512, 1024 tokens | Output: 128 tokens | Batch: 1
- Saves CSV (aggregate) + JSON (with raw trial data)

### Phase 3: Latency Decomposition (Goal 2) — `benchmarks/latency_decomposition.py`
- Uses HuggingFace Transformers with `attn_implementation="eager"` (no fused kernels)
- PyTorch `register_forward_hook` on: embedding, self_attn, MLP, LayerNorm, lm_head
- Manual autoregressive loop (NOT model.generate) with KV-cache
- Reports component times as **percentages** (hooks add sync overhead)
- Per-step CSV + aggregated JSON output

### Phase 4: Scaling Analysis (Goal 3) — `benchmarks/scaling_analysis.py`
- Experiment 1: Sequence length scaling — 128→256→512→1024→2048 (fixed 1B Q4)
- Experiment 2: Model size scaling — 1B vs 3B at same quantization (Q4_K_M)
- Experiment 3: Quantization scaling — Q4_K_M vs Q8_0 vs F16 (fixed 1B)
- Reuses benchmark_harness.py infrastructure

### Phase 5: Bottleneck Analysis (Goal 4) — `analysis/bottleneck_analysis.py`
- Maps measurements from Phases 2-4 to architectural causes
- Computes arithmetic intensity (FLOPs / Bytes) per component
- Classifies components as compute-bound vs memory-bound
- Key relationships:
  - Per-token latency ↑ with seq length → KV-cache exceeds cache hierarchy
  - Attention dominates at long sequences → memory bandwidth saturation
  - MLP dominates at short sequences → compute-bound FFN
  - Quantization gives near-linear speedup → reduced bandwidth per token

### Phase 6: Optimization Proposal (Goal 5)
- Propose one concrete optimization based on bottleneck analysis
- Candidates: KV-cache quantization, kernel fusion, speculative decoding
- Estimate before/after improvement with calculation methodology

### Phase 7: Visualization — `analysis/plot_results.py`
- Plot 1: Per-token latency timeline (warm-up → steady state)
- Plot 2: Latency decomposition horizontal bar chart (% breakdown)
- Plot 3: Sequence length scaling (TTFT + per-token vs prompt length)
- Plot 4: Model size comparison (1B vs 3B side-by-side)
- Plot 5: Quantization impact (Q4 vs Q8 vs F16 grouped bars)
- Plot 6: Cross-platform comparison (Intel vs M-chip vs GPU)

### Phase 8: Report & Presentation
- Write survey paper synthesizing all 6-8 papers + experimental results
- Build presentation slides following required outline

---

## Presentation Outline
1. Introduction and Background
2. Motivation (why token-generation latency matters)
3. Problem Statement
4. Research Focus
5. Related Research (each person presents their 2 papers)
6. Experimental Methodology
7. Results and Analysis
8. Optimization Proposal
9. Conclusion and Future Work

---

## Running the Benchmarks

```bash
# 1. Setup (Mac example)
bash scripts/setup_mac.sh

# 2. Download models
bash scripts/download_models.sh

# 3. Run benchmark harness (Goal 1)
python -m benchmarks.benchmark_harness --model llama-3.2-1b-q4 --trials 10

# 4. Run latency decomposition (Goal 2)
python -m benchmarks.latency_decomposition --tokens 64

# 5. Run scaling analysis (Goal 3)
python -m benchmarks.scaling_analysis --experiment all --trials 5

# 6. Generate plots
python -m analysis.plot_results

# 7. Bottleneck analysis (Goal 4)
python -m analysis.bottleneck_analysis
```

---

## Action Items
- [x] Confirm with professor: NeurIPS, SOSP, ICML papers are acceptable — DONE
- [x] Implement benchmark codebase — DONE
- [ ] Apply for Meta LLaMA 3.2 access on HuggingFace (all team members)
- [ ] Find out the chip on the third Mac
- [ ] Download GGUF models on all platforms
- [ ] Run benchmarks on all 3 platforms
- [ ] Book presentation time slot
- [ ] Team registration (if not already done)
