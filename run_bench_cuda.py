"""
End-to-end benchmark runner for CUDA GPUs (Hugging Face Transformers backend).

Mirror of run_bench.py, but wired to PolarsGenerator in generator_cuda.py.

Scoring formula:
    Score = N / (T * VRAM^0.1 * RAM^0.01)

Usage:
    python run_bench_cuda.py --bench bench.json --model Qwen/Qwen2.5-Coder-7B-Instruct
    python run_bench_cuda.py --bench bench.json --model <model> --repair
    python run_bench_cuda.py --bench bench.json --model <model> \\
        --max-model-len 4096 --gpu-mem-util 0.85
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from typing import Any

import polars as pl

from executor import outputs_match, run_generated_code
from generator_cuda import PolarsGenerator


def load_bench(path: str) -> list[dict[str, Any]]:
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
        if "schema" in expected:
            df = df.cast(
                {
                    c: pl.datatypes.dtype_short_repr_to_dtype(t)  # type: ignore
                    if hasattr(pl.datatypes, "dtype_short_repr_to_dtype")
                    else t
                    for c, t in expected["schema"].items()
                },
                strict=False,
            )
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
            total = 0
            for i in range(torch.cuda.device_count()):
                total += torch.cuda.max_memory_allocated(i)
            return total / 1e9
    except ImportError:
        pass
    return 0.0


def peak_ram_gb() -> float:
    """
    Peak resident memory in GB. `ru_maxrss` is in KB on Linux, bytes on macOS.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > 1e10:
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
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="Per-item exec timeout (seconds).")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only first N items (fast dev loop).")

    ap.add_argument("--dtype", default="bfloat16",
                    choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85,
                    help="Reserved for config compatibility (HF uses device_map=auto).")
    ap.add_argument("--tensor-parallel", type=int, default=1,
                    help="Reserved for config compatibility (single-process HF).")
    ap.add_argument("--no-prefix-cache", action="store_true",
                    help="No-op (vLLM-only; kept for script compatibility).")
    args = ap.parse_args()

    bench = load_bench(args.bench)
    if args.limit:
        bench = bench[: args.limit]
    print(f"Loaded {len(bench)} benchmark items")

    print(f"Loading {args.model} (dtype={args.dtype}, Hugging Face Transformers) ...")
    t_load = time.perf_counter()
    gen = PolarsGenerator(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel,
        enable_prefix_caching=not args.no_prefix_cache,
    )
    print(f"Model loaded in {time.perf_counter() - t_load:.1f}s")

    frames_by_item = [materialize_frames(item["data"]) for item in bench]

    t_start = time.perf_counter()

    if args.repair:
        per_item = gen.generate_with_repair(
            bench, frames_by_item,
            timeout=args.timeout, max_tokens=args.max_tokens,
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
        "backend": "transformers-cuda",
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_mem_util": args.gpu_mem_util,
        "tensor_parallel": args.tensor_parallel,
        "n_items": len(bench),
        "N_correct": correct,
        "accuracy": correct / len(bench) if bench else 0.0,
        "T_total_seconds": total_time,
        "VRAM_GB": vram,
        "RAM_GB": ram,
        "score": score,
        "repair_enabled": args.repair,
        "n_repaired": sum(1 for r in records if r["repaired"]),
        "records": records,
    }

    print("\n" + "=" * 60)
    print(f"Model:       {args.model}")
    print(f"Backend:     Hugging Face Transformers ({args.dtype})")
    print(f"Correct:     {correct} / {len(bench)}  "
          f"({100 * correct / max(len(bench), 1):.1f}%)")
    print(f"Total time:  {total_time:.2f}s")
    print(f"VRAM peak:   {vram:.2f} GB")
    print(f"RAM peak:    {ram:.2f} GB")
    print(f"SCORE:       {score:.4f}")
    if args.repair:
        print(f"Repaired:    {summary['n_repaired']}")
    print("=" * 60)

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
