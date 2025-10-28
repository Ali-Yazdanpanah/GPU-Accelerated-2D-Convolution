# GPU-Accelerated 2D Convolution

Optimized CUDA kernels for Gaussian-style 2D convolution, paired with a CPU reference implementation and benchmarking utilities for exploring tile configurations and speedups.

## Features

- **CPU baseline:** Portable C++17 implementation (`conv2d_serial.cpp`) for correctness and comparison.
- **GPU kernels:** Naive and tiled CUDA variants (`conv2d_optimized.cu`) leveraging shared memory, coalesced loads, and occupancy-aware launch bounds.
- **Configurable tiles:** Compile-time `TILE_X` / `TILE_Y` macros enable quick experimentation with block shapes.
- **Benchmarking scripts:** Python utilities to compile, run, and visualise performance across sizes and tiles.

## Prerequisites

- NVIDIA GPU with CUDA Compute Capability ≥ 7.0 (default compile target `sm_70`).
- CUDA Toolkit (nvcc + runtime libraries).
- C++17-capable host compiler (e.g., `g++`).
- Python 3.8+ with `matplotlib` (for plotting).

## Quick Start

```bash
# CPU reference
g++ -O3 -march=native -std=c++17 -o conv2d_serial conv2d_serial.cpp
./conv2d_serial 1024 1024 7

# GPU kernel (default tile 32x16)
nvcc -O3 -use_fast_math -arch=sm_70 -o conv2d conv2d_optimized.cu
./conv2d 4096 4096 7

# Experiment with alternative block tile (e.g., 64x8)
nvcc -O3 -use_fast_math -arch=sm_70 -DTILE_X=64 -DTILE_Y=8 -o conv2d_64x8 conv2d_optimized.cu
./conv2d_64x8 4096 4096 7
```

Arguments: `./conv2d [H W K]` with defaults `4096 4096 7`. Kernel size must be odd (1–15 supported by the tiled kernel).

## Benchmarking & Reports

Automated scripts live under `scripts/` (see detailed instructions in `benchmarks/README.md`):

```bash
# Collect runtimes for multiple tiles and sizes
python scripts/benchmark_conv2d.py \
  --sizes 1024x1024 2048x2048 4096x4096 \
  --tiles 32x16 32x32 64x8 \
  --k 7 \
  --repeats 3 \
  --output benchmarks/results

# Generate comparison charts and summary (add --log-scale if needed)
python scripts/plot_benchmarks.py --log-scale
```

Outputs include CSV/JSON data, runtime & speedup PNG charts, and a Markdown summary highlighting the best-performing tile per shape.

## Repository Layout

- `conv2d_serial.cpp` – CPU reference convolution.
- `conv2d_optimized.cu` – CUDA kernels with tunable tile sizes.
- `scripts/` – Benchmark automation (`benchmark_conv2d.py`) and plotting (`plot_benchmarks.py`).
- `benchmarks/` – Generated artifacts + usage guide.
- `README.md` – Project overview (this file).

## Notes

- The tiled kernel relies on constant memory for kernels up to 15×15; adjust if larger filters are needed.
- Shared-memory usage scales with `TILE_X`, `TILE_Y`, and `K`. Ensure the chosen tile fits device limits (threads ≤ 1024, shared memory per block).
- To add new tile shapes, compile with different `-DTILE_X`, `-DTILE_Y` values or extend the benchmarking script’s `--tiles` list.

Happy benchmarking!
