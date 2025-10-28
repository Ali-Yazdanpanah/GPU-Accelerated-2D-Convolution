#!/usr/bin/env python3
"""Generate charts and markdown summaries from conv2d benchmark data."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc


def load_results(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {
                "H": int(raw["H"]),
                "W": int(raw["W"]),
                "K": int(raw["K"]),
                "tile_x": int(raw["tile_x"]),
                "tile_y": int(raw["tile_y"]),
                "repeats": int(raw["repeats"]),
                "cpu_ms": float(raw["cpu_ms"]),
                "gpu_naive_ms": float(raw["gpu_naive_ms"]),
                "gpu_tiled_ms": float(raw["gpu_tiled_ms"]),
                "cpu_vs_gpu_tiled_speedup": float(raw["cpu_vs_gpu_tiled_speedup"]),
                "gpu_naive_vs_tiled_speedup": float(raw["gpu_naive_vs_tiled_speedup"]),
            }
            rows.append(row)
    return rows


def group_by_shape(rows: Iterable[Dict]) -> Dict[Tuple[int, int, int], List[Dict]]:
    groups: Dict[Tuple[int, int, int], List[Dict]] = {}
    for row in rows:
        key = (row["H"], row["W"], row["K"])
        groups.setdefault(key, []).append(row)
    return groups


def plot_runtimes(
    shape_key: Tuple[int, int, int],
    rows: List[Dict],
    outdir: Path,
    log_scale: bool,
) -> Path:
    rows = sorted(rows, key=lambda r: (r["tile_x"], r["tile_y"]))
    labels = [f"{r['tile_x']}x{r['tile_y']}" for r in rows]
    cpu = [r["cpu_ms"] for r in rows]
    gpu_naive = [r["gpu_naive_ms"] for r in rows]
    gpu_tiled = [r["gpu_tiled_ms"] for r in rows]

    x = range(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width for i in x], cpu, width, label="CPU")
    ax.bar(x, gpu_naive, width, label="GPU naive")
    ax.bar([i + width for i in x], gpu_tiled, width, label="GPU tiled")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime (ms)")
    if log_scale:
        ax.set_yscale("log")
    H, W, K = shape_key
    ax.set_title(f"Runtime comparison H={H}, W={W}, K={K}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = outdir / f"runtime_{H}x{W}_K{K}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_speedups(
    shape_key: Tuple[int, int, int],
    rows: List[Dict],
    outdir: Path,
    log_scale: bool,
) -> Path:
    rows = sorted(rows, key=lambda r: (r["tile_x"], r["tile_y"]))
    labels = [f"{r['tile_x']}x{r['tile_y']}" for r in rows]
    cpu_speedup = [r["cpu_vs_gpu_tiled_speedup"] for r in rows]
    gpu_speedup = [r["gpu_naive_vs_tiled_speedup"] for r in rows]

    x = range(len(rows))
    width = 0.3
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], cpu_speedup, width, label="CPU vs GPU tiled")
    ax.bar([i + width / 2 for i in x], gpu_speedup, width, label="GPU naive vs tiled")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup (×)")
    if log_scale:
        ax.set_yscale("log")
    H, W, K = shape_key
    ax.set_title(f"Speedup comparison H={H}, W={W}, K={K}")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path = outdir / f"speedup_{H}x{W}_K{K}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_summary(groups: Dict[Tuple[int, int, int], List[Dict]], outdir: Path) -> Path:
    lines: List[str] = [
        "# conv2d Benchmark Summary",
        "",
        "| Shape (H×W, K) | Tile | CPU (ms) | GPU naive (ms) | GPU tiled (ms) | CPU/Tiled speedup | Naive/Tiled speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda r: (r["tile_x"], r["tile_y"]))
        best = min(rows, key=lambda r: r["gpu_tiled_ms"])
        for row in rows:
            shape_str = f"{row['H']}×{row['W']}, K={row['K']}"
            tile_str = f"{row['tile_x']}×{row['tile_y']}"
            lines.append(
                f"| {shape_str} | {tile_str} | "
                f"{row['cpu_ms']:.3f} | {row['gpu_naive_ms']:.3f} | {row['gpu_tiled_ms']:.3f} | "
                f"{row['cpu_vs_gpu_tiled_speedup']:.2f}× | {row['gpu_naive_vs_tiled_speedup']:.2f}× |"
            )
        lines.append("")
        lines.append(
            f"> Best tile for H={best['H']}, W={best['W']}, K={best['K']}: "
            f"{best['tile_x']}×{best['tile_y']} ({best['gpu_tiled_ms']:.3f} ms)"
        )
        lines.append("")

    summary_path = outdir / "summary.md"
    summary_path.write_text("\n".join(lines))
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmarks/results.csv"),
        help="CSV file produced by benchmark_conv2d.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Where to write plots and summary markdown.",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Render charts with a logarithmic Y-axis to highlight large gaps.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    args.output_dir.mkdir(exist_ok=True, parents=True)
    rows = load_results(args.input)
    groups = group_by_shape(rows)

    runtime_paths = []
    speedup_paths = []
    for key, group in groups.items():
        runtime_paths.append(plot_runtimes(key, group, args.output_dir, args.log_scale))
        speedup_paths.append(plot_speedups(key, group, args.output_dir, args.log_scale))

    summary_path = write_summary(groups, args.output_dir)

    print("[report] Generated plots:")
    for path in runtime_paths + speedup_paths:
        print(f"  - {path}")
    print(f"[report] Summary markdown: {summary_path}")


if __name__ == "__main__":
    main()
