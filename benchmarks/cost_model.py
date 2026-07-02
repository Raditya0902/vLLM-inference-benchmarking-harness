#!/usr/bin/env python3
"""Compute $/1M tokens for each benchmarked configuration.

Reads the vendored benchmark_serving.py result JSON files in results/ (one
per config x concurrency level, e.g. fp16-c32.json, awq-c8.json,
baseline-c1.json) and derives cost per 1M tokens from the RunPod hourly
rate divided by each run's measured throughput. Purely retrospective — no
GPU or live server needed, just the JSON files Phase 1/2 already produced.

Usage:
    python3 benchmarks/cost_model.py [--results-dir results] [--hourly-rate 0.27]
"""
import argparse
import json
import re
from pathlib import Path

RESULT_FILENAME_RE = re.compile(r"^(?P<config>[a-zA-Z0-9]+)-c(?P<concurrency>\d+)\.json$")


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*-c*.json")):
        match = RESULT_FILENAME_RE.match(path.name)
        if not match:
            continue
        data = json.loads(path.read_text())
        rows.append(
            {
                "config": match["config"],
                "concurrency": int(match["concurrency"]),
                "output_throughput": data["output_throughput"],
                "total_token_throughput": data["total_token_throughput"],
            }
        )
    return rows


def cost_per_1m_tokens(hourly_rate: float, tokens_per_second: float) -> float:
    cost_per_second = hourly_rate / 3600
    cost_per_token = cost_per_second / tokens_per_second
    return cost_per_token * 1_000_000


def build_table(rows: list[dict], hourly_rate: float) -> list[dict]:
    table = []
    for row in rows:
        table.append(
            {
                **row,
                "usd_per_1m_output_tokens": cost_per_1m_tokens(hourly_rate, row["output_throughput"]),
                "usd_per_1m_total_tokens": cost_per_1m_tokens(hourly_rate, row["total_token_throughput"]),
            }
        )
    return table


def print_table(table: list[dict], hourly_rate: float) -> None:
    print(f"GPU rental rate: ${hourly_rate:.2f}/hr\n")
    header = f"{'config':<10}{'concurrency':>12}{'output tok/s':>14}{'$/1M output tok':>18}{'$/1M total tok':>16}"
    print(header)
    print("-" * len(header))
    for row in table:
        print(
            f"{row['config']:<10}{row['concurrency']:>12}"
            f"{row['output_throughput']:>14.1f}"
            f"{row['usd_per_1m_output_tokens']:>18.4f}"
            f"{row['usd_per_1m_total_tokens']:>16.4f}"
        )

    max_concurrency = max(row["concurrency"] for row in table)
    print(f"\nHeadline (sustained production throughput, concurrency={max_concurrency}):")
    print(f"{'config':<10}{'$/1M output tok':>18}{'$/1M total tok':>16}")
    for row in table:
        if row["concurrency"] == max_concurrency:
            print(
                f"{row['config']:<10}"
                f"{row['usd_per_1m_output_tokens']:>18.4f}"
                f"{row['usd_per_1m_total_tokens']:>16.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--hourly-rate",
        type=float,
        default=0.27,
        help="GPU rental rate in USD/hr (default: 0.27, this project's RunPod A5000 rate)",
    )
    args = parser.parse_args()

    rows = load_results(args.results_dir)
    if not rows:
        raise SystemExit(f"No *-c*.json result files found in {args.results_dir}")

    table = build_table(rows, args.hourly_rate)
    table.sort(key=lambda r: (r["config"], r["concurrency"]))
    print_table(table, args.hourly_rate)


if __name__ == "__main__":
    main()
