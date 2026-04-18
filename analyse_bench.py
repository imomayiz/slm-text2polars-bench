"""
Lightweight benchmark analysis (no Polars).

Run:
    python analyze_bench.py bench.json
"""

import json
import sys
from collections import Counter, defaultdict
from statistics import mean
from pathlib import Path

import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return json.load(f)


def basic_stats(items):
    n = len(items)

    categories = Counter(it["category"] for it in items)
    difficulties = Counter(it["difficulty"] for it in items)

    num_tables = [
        len(it.get("schema", {}))
        for it in items
    ]

    num_rows = [
        sum(len(v) for v in it.get("data", {}).values())
        for it in items
    ]

    question_lengths = [
        len(it["question"])
        for it in items
    ]

    return {
        "num_items": n,
        "num_categories": len(categories),
        "num_difficulties": len(difficulties),
        "categories": categories,
        "difficulties": difficulties,
        "avg_tables": mean(num_tables),
        "avg_rows": mean(num_rows),
        "avg_question_length": mean(question_lengths),
    }


def cross_table(items):
    table = defaultdict(lambda: defaultdict(int))

    for it in items:
        table[it["category"]][it["difficulty"]] += 1

    return table


def print_summary(stats):
    print("\n=== BENCHMARK SUMMARY ===\n")
    print(f"Total items: {stats['num_items']}")
    print(f"Categories: {stats['num_categories']}")
    print(f"Difficulties: {stats['num_difficulties']}")
    print(f"Avg tables per item: {stats['avg_tables']:.2f}")
    print(f"Avg rows per item: {stats['avg_rows']:.2f}")
    print(f"Avg question length: {stats['avg_question_length']:.1f}")

    print("\n--- Categories ---")
    for k, v in stats["categories"].most_common():
        print(f"{k:15s} {v}")

    print("\n--- Difficulties ---")
    for k, v in stats["difficulties"].most_common():
        print(f"{k:10s} {v}")


def plot_bar(counter, title, out_path):
    keys = list(counter.keys())
    values = list(counter.values())

    plt.figure()
    plt.bar(keys, values)
    plt.title(title)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_heatmap(table, out_path):
    import numpy as np

    cats = sorted(table.keys())
    diffs = sorted({d for v in table.values() for d in v})

    matrix = []
    for c in cats:
        row = []
        for d in diffs:
            row.append(table[c].get(d, 0))
        matrix.append(row)

    plt.figure()
    plt.imshow(matrix)
    plt.colorbar()

    plt.xticks(range(len(diffs)), diffs)
    plt.yticks(range(len(cats)), cats)

    for i in range(len(cats)):
        for j in range(len(diffs)):
            plt.text(j, i, matrix[i][j], ha="center", va="center")

    plt.title("Category vs Difficulty")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_bench.py bench.json")
        return

    items = load(sys.argv[1])
    stats = basic_stats(items)
    table = cross_table(items)

    print_summary(stats)

    out_dir = Path("analysis_outputs")
    out_dir.mkdir(exist_ok=True)

    plot_bar(stats["categories"], "Items per Category", out_dir / "categories.png")
    plot_bar(stats["difficulties"], "Items per Difficulty", out_dir / "difficulties.png")
    plot_heatmap(table, out_dir / "heatmap.png")

    print(f"\nSaved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()