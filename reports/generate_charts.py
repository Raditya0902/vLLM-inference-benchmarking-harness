#!/usr/bin/env python3
"""Generate the Phase 4 report charts from results/*.json + *.gpu.csv.

Purely retrospective — reads Phase 1/2's already-saved benchmark output, no
GPU or live server needed. Writes PNGs into reports/images/, referenced from
reports/README.md.

Usage:
    python3 reports/generate_charts.py [--results-dir results] [--out-dir reports/images]
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]

# Fixed categorical order/colors (dataviz skill palette, slots 1-4: blue,
# aqua, yellow, green) — never reassigned or cycled per config.
CONFIGS = [
    ("fp16", "vLLM fp16", "#2a78d6"),
    ("awq", "vLLM AWQ (awq_marlin)", "#1baf7a"),
    ("gptq", "vLLM GPTQ", "#eda100"),
    ("baseline", "Naive HF baseline", "#008300"),
]

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
HOURLY_RATE = 0.27


def load_results(results_dir: Path) -> dict:
    data = {name: {} for name, _, _ in CONFIGS}
    for name in data:
        for c in CONCURRENCY_LEVELS:
            path = results_dir / f"{name}-c{c}.json"
            if path.exists():
                data[name][c] = json.loads(path.read_text())
    return data


def load_peak_gpu_mem_gib(results_dir: Path) -> dict:
    """Peak memory.used across each run's .gpu.csv, in GiB."""
    peaks = {name: {} for name, _, _ in CONFIGS}
    for name in peaks:
        for c in CONCURRENCY_LEVELS:
            path = results_dir / f"{name}-c{c}.gpu.csv"
            if not path.exists():
                continue
            with open(path) as f:
                reader = csv.DictReader(f)
                reader.fieldnames = [name.strip() for name in reader.fieldnames]
                rows = list(reader)
            used_mib = [
                float(row["memory.used [MiB]"].strip().split()[0]) for row in rows
            ]
            peaks[name][c] = max(used_mib) / 1024
    return peaks


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def plot_throughput_vs_concurrency(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    for name, label, color in CONFIGS:
        xs = [c for c in CONCURRENCY_LEVELS if c in results[name]]
        ys = [results[name][c]["output_throughput"] for c in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, label=label)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.set_xticklabels([str(c) for c in CONCURRENCY_LEVELS])
    ax.set_xlabel("Max concurrency")
    ax.set_ylabel("Output tokens/sec (log scale)")
    ax.set_title("Output Throughput vs. Concurrency")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_ttft_vs_concurrency(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=SURFACE)
    for name, label, color in CONFIGS:
        xs = [c for c in CONCURRENCY_LEVELS if c in results[name]]
        ys = [results[name][c]["median_ttft_ms"] for c in xs]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=8, label=label)
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCY_LEVELS)
    ax.set_xticklabels([str(c) for c in CONCURRENCY_LEVELS])
    ax.set_xlabel("Max concurrency")
    ax.set_ylabel("Median TTFT, ms (log scale)")
    ax.set_title("Median Time-to-First-Token vs. Concurrency")
    style_axes(ax)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ttft_vs_concurrency.png", dpi=150)
    plt.close(fig)


def plot_cost_per_1m_tokens(results: dict, out_dir: Path) -> None:
    concurrency = 32
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    labels, costs, colors = [], [], []
    for name, label, color in CONFIGS:
        if concurrency not in results[name]:
            continue
        throughput = results[name][concurrency]["output_throughput"]
        cost = (HOURLY_RATE / 3600) / throughput * 1_000_000
        labels.append(label)
        costs.append(cost)
        colors.append(color)
    bars = ax.bar(labels, costs, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("$ / 1M output tokens (log scale)")
    ax.set_title(f"Cost per 1M Output Tokens (concurrency={concurrency})")
    style_axes(ax)
    ax.tick_params(axis="x", labelsize=8)
    for bar, cost in zip(bars, costs):
        ax.annotate(
            f"${cost:.3f}" if cost < 1 else f"${cost:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_PRIMARY,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "cost_per_1m_tokens.png", dpi=150)
    plt.close(fig)


def plot_memory_tradeoff(peaks: dict, out_dir: Path) -> None:
    concurrency = 32
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=SURFACE)
    labels, mem, colors = [], [], []
    for name, label, color in CONFIGS:
        if concurrency not in peaks[name]:
            continue
        labels.append(label)
        mem.append(peaks[name][concurrency])
        colors.append(color)
    bars = ax.bar(labels, mem, color=colors, width=0.6)
    ax.set_ylabel("Peak GPU memory, GiB")
    ax.set_title(f"Peak GPU Memory (concurrency={concurrency})")
    style_axes(ax)
    ax.tick_params(axis="x", labelsize=8)
    for bar, m in zip(bars, mem):
        ax.annotate(
            f"{m:.1f} GiB",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_PRIMARY,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "memory_tradeoff.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/images"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    peaks = load_peak_gpu_mem_gib(args.results_dir)

    plot_throughput_vs_concurrency(results, args.out_dir)
    plot_ttft_vs_concurrency(results, args.out_dir)
    plot_cost_per_1m_tokens(results, args.out_dir)
    plot_memory_tradeoff(peaks, args.out_dir)

    print(f"Charts written to {args.out_dir}")


if __name__ == "__main__":
    main()
