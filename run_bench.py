"""
End-to-end benchmark runner.

Loads your gold mini-benchmark, runs the generator, executes each output,
compares to gold, and computes the official score:

    Score = N / (T * VRAM^0.1 * RAM^0.01)

Usage:
    python run_bench.py --bench bench.json --model Qwen/Qwen2.5-Coder-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import polars as pl

from executor import run_generated_code, outputs_match
from generator import PolarsGenerator


def load_bench(path: str) -> list[dict[str, Any]]:
    """
    Bench file format (JSON list). Each item:
    {
      "id": "q001",
      "question": "...",
      "schema": {"df": {"col": "Utf8", ...}, ...},
      "data": {"df": [{"col": val, ...}, ...]},   # inline fixture data
      "reference_code": "result = df...",         # optional, for debugging
      "expected_output": {                        # computed from reference
          "kind": "dataframe" | "scalar" | "series",
          ...
      },
      "tolerance": {"order_matters": false, "float_atol": 1e-6}
    }
    """
    with open(path) as f:
        return json.load(f)


def materialize_frames(data: dict[str, list[dict]]) -> dict[str, pl.DataFrame]:
    """Turn the inline JSON data into Polars DataFrames."""
    return {name: pl.DataFrame(rows) for name, rows in data.items()}


def deserialize_expected(expected: dict) -> Any:
    """Re-hydrate the gold answer from JSON form."""
    kind = expected["kind"]
    if kind == "dataframe":
        df = pl.DataFrame(expected["value"])
        # Preserve dtypes if provided
        if "schema" in expected:
            df = df.cast({c: pl.datatypes.dtype_short_repr_to_dtype(t)  # type: ignore
                          if hasattr(pl.datatypes, "dtype_short_repr_to_dtype")
                          else t for c, t in expected["schema"].items()},
                         strict=False)
        return df
    if kind == "series":
        return pl.Series(name=expected.get("name", ""), values=expected["value"])
    if kind == "scalar":
        return expected["value"]
    raise ValueError(f"Unknown expected kind: {kind}")


def peak_vram_gb() -> float:
    """Peak GPU memory allocated this process has seen, in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except ImportError:
        pass
    return 0.0


def peak_ram_gb() -> float:
    """
    Peak resident memory in GB. `ru_maxrss` is in KB on Linux, bytes on macOS.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Heuristic: if rss looks like bytes (macOS), divide by 1e9; else by 1e6 (KB on Linux)
    if rss > 1e10:  # >10 GB in raw units => must be bytes
        return rss / 1e9
    return rss / 1e6


def compute_score(N: int, T: float, vram_gb: float, ram_gb: float) -> float:
    """Score = N / (T * VRAM^0.1 * RAM^0.01). Guard against zeros."""
    vram = max(vram_gb, 0.01)
    ram = max(ram_gb, 0.01)
    t = max(T, 1e-6)
    return N / (t * (vram ** 0.1) * (ram ** 0.01))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="Path to bench JSON")
    ap.add_argument("--model", required=True)
    ap.add_argument("--repair", action="store_true",
                    help="Enable execution-feedback repair loop (+T, +N).")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only first N items (fast dev loop)")
    args = ap.parse_args()

    bench = load_bench(args.bench)
    if args.limit:
        bench = bench[: args.limit]
    print(f"Loaded {len(bench)} benchmark items")

    # --- Load model
    print(f"Loading {args.model} ...")
    gen = PolarsGenerator(model=args.model)

    # --- Prepare dataframes
    frames_by_item = [materialize_frames(item["data"]) for item in bench]

    # --- Generate + execute
    t_start = time.perf_counter()

    if args.repair:
        per_item = gen.generate_with_repair(
            bench, frames_by_item, timeout=args.timeout,
        )
    else:
        gen_results = gen.generate(bench, max_tokens=args.max_tokens)
        per_item = []
        for item, gr, frames in zip(bench, gen_results, frames_by_item):
            exec_res = run_generated_code(gr.code, frames, timeout=args.timeout)
            per_item.append({
                "code": gr.code,
                "exec_result": exec_res,
                "repaired": False,
                "gen_time": gr.gen_time,
            })

    total_time = time.perf_counter() - t_start

    # --- Score each item
    correct = 0
    records = []
    for item, r in zip(bench, per_item):
        exec_res = r["exec_result"]
        gold = deserialize_expected(item["expected_output"])
        tol = item.get("tolerance", {})
        is_correct = (
            exec_res.ok and outputs_match(
                exec_res.value, gold,
                order_matters=tol.get("order_matters", False),
                float_atol=tol.get("float_atol", 1e-6),
                check_column_names=False,
            )
        )
        correct += int(is_correct)
        records.append({
            "id": item.get("id", ""),
            "question": item["question"],
            "code": r["code"],
            "repaired": r.get("repaired", False),
            "exec_ok": exec_res.ok,
            "correct": is_correct,
            "error": exec_res.error if not exec_res.ok else None,
            "exec_time": exec_res.wall_time,
            "gen_time": r.get("gen_time", 0.0),
        })

    vram = peak_vram_gb()
    ram = peak_ram_gb()
    score = compute_score(correct, total_time, vram, ram)

    summary = {
        "model": args.model,
        "n_items": len(bench),
        "N_correct": correct,
        "accuracy": correct / len(bench),
        "T_total_seconds": total_time,
        "VRAM_GB": vram,
        "RAM_GB": ram,
        "score": score,
        "repair_enabled": args.repair,
        "n_repaired": sum(1 for r in records if r["repaired"]),
        "records": records,
    }

    # --- Print + save
    print("\n" + "=" * 60)
    print(f"Model:       {args.model}")
    print(f"Correct:     {correct} / {len(bench)}  ({100*correct/len(bench):.1f}%)")
    print(f"Total time:  {total_time:.2f}s")
    print(f"VRAM peak:   {vram:.2f} GB")
    print(f"RAM peak:    {ram:.2f} GB")
    print(f"SCORE:       {score:.4f}")
    if args.repair:
        print(f"Repaired:    {summary['n_repaired']}")
    print("=" * 60)

    # --- Error breakdown
    wrong = [r for r in records if not r["correct"]]
    if wrong:
        print(f"\n{len(wrong)} failures. First 3:")
        for r in wrong[:3]:
            print(f"\n  [{r['id']}] {r['question']}")
            print(f"  code: {r['code'][:100]}")
            if r["error"]:
                print(f"  err:  {r['error'].splitlines()[-1][:120]}")

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
