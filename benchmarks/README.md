# Benchmarking Guide

This workflow collects runtime data for the CPU and GPU convolution implementations, then visualises speedups across different tile shapes.

## 1. Run benchmarks

```bash
python scripts/benchmark_conv2d.py \
  --sizes 1024x1024 2048x2048 4096x4096 \
  --tiles 32x16 32x32 64x8 \
  --k 7 \
  --repeats 3 \
  --output benchmarks/results
```

What this does:

- Builds the CPU binary, plus one GPU binary per tile (`benchmarks/bin/`).
- Executes each combination `repeats` times and records mean runtimes.
- Writes `benchmarks/results.csv` and `benchmarks/results.json`.

> **Tip:** Add or remove `--sizes`/`--tiles` to explore other shapes. The script re-compiles only the binaries it needs.

## 2. Generate plots and summary

```bash
python scripts/plot_benchmarks.py \
  --input benchmarks/results.csv \
  --output-dir benchmarks
```

Add `--log-scale` to render runtimes and speedups on a logarithmic Y-axis when CPU/GPU gaps are large:

```bash
python scripts/plot_benchmarks.py --log-scale
```

Outputs:

- `runtime_<HxW>_K<K>.png`: bar chart comparing CPU, GPU naive, GPU tiled runtimes per tile shape.
- `speedup_<HxW>_K<K>.png`: bar chart of speedups.
- `summary.md`: markdown table with all metrics and a note on the fastest tile per shape.

> Make sure `matplotlib` is installed (`pip install matplotlib`) before running the plotting script.

## 3. Interpreting the data

- `cpu_vs_gpu_tiled_speedup` shows CPU runtime divided by tiled GPU runtime (>1.0 means the GPU is faster).
- `gpu_naive_vs_tiled_speedup` isolates the benefit of the tiled kernel over the naive GPU baseline.
- Use the generated markdown summary to spot the best tile shape per input size and kernel.

You can regenerate the CSV/plots at any time after adjusting the tile list, shapes, or kernel size.
