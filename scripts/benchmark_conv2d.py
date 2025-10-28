#!/usr/bin/env python3
"""Benchmark conv2d CPU/GPU implementations across tile sizes and input shapes.

This script automates:
  * building the CPU reference implementation
  * building GPU kernels for each requested tile size
  * executing both binaries for the provided input shapes
  * collecting runtime metrics into a CSV/JSON artifact

Example:
  python scripts/benchmark_conv2d.py \
      --sizes 1024x1024 2048x2048 4096x4096 \
      --tiles 32x16 32x32 64x8 \
      --k 7 --repeats 3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC_CPU = ROOT / "conv2d_serial.cpp"
SRC_GPU = ROOT / "conv2d_optimized.cu"

RESULTS_DIR = ROOT / "benchmarks"
BIN_DIR = RESULTS_DIR / "bin"
DEFAULT_OUTPUT = RESULTS_DIR / "results"


@dataclass(frozen=True)
class HWSize:
    H: int
    W: int

    @classmethod
    def parse(cls, token: str) -> "HWSize":
        match = re.match(r"^(\d+)x(\d+)$", token)
        if not match:
            raise argparse.ArgumentTypeError(f"Invalid size '{token}', expected format HxW.")
        h, w = map(int, match.groups())
        if h <= 0 or w <= 0:
            raise argparse.ArgumentTypeError("Input dimensions must be positive.")
        return cls(h, w)


@dataclass(frozen=True)
class TileShape:
    x: int
    y: int

    @classmethod
    def parse(cls, token: str) -> "TileShape":
        match = re.match(r"^(\d+)x(\d+)$", token)
        if not match:
            raise argparse.ArgumentTypeError(f"Invalid tile '{token}', expected format TxTy.")
        tx, ty = map(int, match.groups())
        if tx <= 0 or ty <= 0:
            raise argparse.ArgumentTypeError("Tile dimensions must be positive.")
        if tx * ty > 1024:
            raise argparse.ArgumentTypeError("Tile area exceeds CUDA block limit (1024 threads).")
        return cls(tx, ty)


def run(cmd: Sequence[str], *, cwd: Path, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
    )


def compile_cpu(binary: Path) -> None:
    print(f"[build] g++ -> {binary.name}")
    run(
        [
            "g++",
            "-O3",
            "-march=native",
            "-std=c++17",
            "-o",
            str(binary),
            str(SRC_CPU),
        ],
        cwd=ROOT,
    )


def compile_gpu(tile: TileShape, binary: Path) -> None:
    print(f"[build] nvcc TILE={tile.x}x{tile.y} -> {binary.name}")
    run(
        [
            "nvcc",
            "-O3",
            "-use_fast_math",
            "-arch=sm_70",
            f"-DTILE_X={tile.x}",
            f"-DTILE_Y={tile.y}",
            "-o",
            str(binary),
            str(SRC_GPU),
        ],
        cwd=ROOT,
    )


def parse_cpu_time(output: str) -> float:
    for line in output.splitlines():
        if "CPU conv2d completed in" in line:
            tokens = line.strip().split()
            return float(tokens[-2])
    raise RuntimeError("Failed to parse CPU time from output:\n" + output)


def parse_gpu_times(output: str) -> Tuple[float, float]:
    naive_ms = tiled_ms = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Naive conv:"):
            naive_ms = float(line.split()[2])
        elif line.startswith("Tiled conv:"):
            tiled_ms = float(line.split()[2])
    if naive_ms is None or tiled_ms is None:
        raise RuntimeError("Failed to parse GPU times from output:\n" + output)
    return naive_ms, tiled_ms


def run_cpu(binary: Path, size: HWSize, k: int, repeats: int) -> List[float]:
    cmd = [str(binary), str(size.H), str(size.W), str(k)]
    times: List[float] = []
    for i in range(repeats):
        result = run(cmd, cwd=ROOT, capture=True)
        t = parse_cpu_time(result.stdout)
        times.append(t)
        print(f"[run] CPU {size.H}x{size.W} K={k} rep {i+1}/{repeats}: {t:.3f} ms")
    return times


def run_gpu(binary: Path, size: HWSize, k: int, repeats: int) -> Tuple[List[float], List[float]]:
    cmd = [str(binary), str(size.H), str(size.W), str(k)]
    naive_times: List[float] = []
    tiled_times: List[float] = []
    for i in range(repeats):
        result = run(cmd, cwd=ROOT, capture=True)
        naive, tiled = parse_gpu_times(result.stdout)
        print(
            f"[run] GPU {binary.name} {size.H}x{size.W} K={k} rep {i+1}/{repeats}: "
            f"naive={naive:.3f} ms, tiled={tiled:.3f} ms"
        )
        naive_times.append(naive)
        tiled_times.append(tiled)
    return naive_times, tiled_times


def aggregate(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals)


def save_results(
    rows: List[dict],
    output_stem: Path,
) -> None:
    csv_path = output_stem.with_suffix(".csv")
    json_path = output_stem.with_suffix(".json")

    fieldnames = [
        "H",
        "W",
        "K",
        "tile_x",
        "tile_y",
        "repeats",
        "cpu_ms",
        "gpu_naive_ms",
        "gpu_tiled_ms",
        "cpu_vs_gpu_tiled_speedup",
        "gpu_naive_vs_tiled_speedup",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    print(f"[report] wrote {csv_path}")
    print(f"[report] wrote {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=HWSize.parse,
        nargs="+",
        default=[HWSize(1024, 1024), HWSize(2048, 2048), HWSize(4096, 4096)],
        help="List of HxW shapes to benchmark.",
    )
    parser.add_argument(
        "--tiles",
        type=TileShape.parse,
        nargs="+",
        default=[TileShape(32, 16), TileShape(32, 32), TileShape(64, 8)],
        help="List of TILE_XxTILE_Y configurations to evaluate.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Kernel size (must be odd).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repetitions per measurement.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file stem for reports (without extension).",
    )
    args = parser.parse_args()

    if args.k % 2 == 0 or args.k < 1:
        parser.error("Kernel size must be odd and >= 1.")

    RESULTS_DIR.mkdir(exist_ok=True)
    BIN_DIR.mkdir(exist_ok=True)

    cpu_bin = BIN_DIR / "conv2d_serial"
    compile_cpu(cpu_bin)

    rows = []
    for tile in args.tiles:
        gpu_bin = BIN_DIR / f"conv2d_{tile.x}x{tile.y}"
        compile_gpu(tile, gpu_bin)

        for size in args.sizes:
            cpu_samples = run_cpu(cpu_bin, size, args.k, args.repeats)
            gpu_naive_samples, gpu_tiled_samples = run_gpu(gpu_bin, size, args.k, args.repeats)

            cpu_mean = aggregate(cpu_samples)
            gpu_naive_mean = aggregate(gpu_naive_samples)
            gpu_tiled_mean = aggregate(gpu_tiled_samples)
            rows.append(
                {
                    "H": size.H,
                    "W": size.W,
                    "K": args.k,
                    "tile_x": tile.x,
                    "tile_y": tile.y,
                    "repeats": args.repeats,
                    "cpu_ms": cpu_mean,
                    "gpu_naive_ms": gpu_naive_mean,
                    "gpu_tiled_ms": gpu_tiled_mean,
                    "cpu_vs_gpu_tiled_speedup": cpu_mean / gpu_tiled_mean if gpu_tiled_mean else None,
                    "gpu_naive_vs_tiled_speedup": gpu_naive_mean / gpu_tiled_mean if gpu_tiled_mean else None,
                }
            )

    save_results(rows, args.output)


if __name__ == "__main__":
    main()
