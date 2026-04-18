"""
Experiment runner for text2Polars.

Groups experiments by model so each loads once, then sweeps prompt/routing variants.

Usage:
    python run.py --bench bench.json
    python run.py --bench bench.json --grid samples/grid.json
    python run.py --bench bench.json --dry-run
    python run.py --bench bench.json --skip-done
    python run.py --bench bench.json --limit 5
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import resource
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from executor import outputs_match, run_generated_code
from inference import InferenceConfig, PolarsInferenceEngine


# ============================================================================
# Helpers
# ============================================================================

def load_bench(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def materialize_frames(data: dict[str, list[dict]]) -> dict[str, pl.DataFrame]:
    return {name: pl.DataFrame(rows) for name, rows in data.items()}


def deserialize_expected(expected: dict) -> Any:
    kind = expected["kind"]
    if kind == "dataframe":
        df = pl.DataFrame(expected["value"])
        if "schema" in expected:
            df = df.cast(
                {c: pl.datatypes.dtype_short_repr_to_dtype(t)
                 if hasattr(pl.datatypes, "dtype_short_repr_to_dtype") else t
                 for c, t in expected["schema"].items()},
                strict=False,
            )
        return df
    if kind == "series":
        return pl.Series(name=expected.get("name", ""), values=expected["value"])
    if kind == "scalar":
        return expected["value"]
    raise ValueError(f"Unknown expected kind: {kind}")


def peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return sum(torch.cuda.max_memory_allocated(i)
                       for i in range(torch.cuda.device_count())) / 1e9
    except ImportError:
        pass
    return 0.0


def reset_vram_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
    except ImportError:
        pass


def peak_ram_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if rss > 1e10 else rss / 1e6


def compute_score(N: int, T: float, vram_gb: float, ram_gb: float) -> float:
    return N / (max(T, 1e-6) * max(vram_gb, 0.01) ** 0.1 * max(ram_gb, 0.01) ** 0.01)


def _serialize_value(val: Any) -> str:
    """Best-effort short string representation of a result value."""
    if isinstance(val, pl.DataFrame):
        if val.height <= 5:
            return val.to_pandas().to_string(index=False)
        return f"DataFrame({val.height}x{val.width}): {val.head(3).to_pandas().to_string(index=False)} ..."
    if isinstance(val, pl.Series):
        return str(val.to_list()[:10])
    return repr(val)


def eval_items(engine: PolarsInferenceEngine, bench: list[dict], timeout: float) -> tuple[int, list[dict]]:
    """Run generate+execute+compare for each bench item. Returns (correct, records)."""
    correct = 0
    records = []
    for item in bench:
        frames = materialize_frames(item["data"])
        t0 = time.perf_counter()
        try:
            res = engine.generate_one(question=item["question"], schema=item["schema"])
            code = res.code
            raw = res.raw
            gen_error = None
        except Exception as e:
            code = ""
            raw = ""
            gen_error = f"{type(e).__name__}: {e}"
        gen_time = time.perf_counter() - t0

        exec_ok = False
        is_correct = False
        err = gen_error
        got = None
        expected = None

        if code:
            exec_res = run_generated_code(code, frames, timeout=timeout)
            exec_ok = exec_res.ok
            gold = deserialize_expected(item["expected_output"])
            tol = item.get("tolerance", {})
            if exec_ok:
                is_correct = outputs_match(
                    exec_res.value, gold,
                    order_matters=tol.get("order_matters", False),
                    float_atol=tol.get("float_atol", 1e-6),
                    check_column_names=False,
                )
                if not is_correct:
                    got = _serialize_value(exec_res.value)
                    expected = _serialize_value(gold)
            else:
                err = exec_res.error
                expected = _serialize_value(gold)

        correct += int(is_correct)
        record: dict[str, Any] = {
            "id": item.get("id", ""),
            "question": item["question"],
            "category": item.get("category", ""),
            "code": code,
            "raw": raw,
            "exec_ok": exec_ok,
            "correct": is_correct,
            "error": err,
            "got": got,
            "expected": expected,
            "gen_time": gen_time,
        }
        records.append(record)
    return correct, records


# ============================================================================
# Experiment config + default grid
# ============================================================================

@dataclass
class ExperimentConfig:
    model: str
    prompt: str = "baseline"
    use_routing: bool = True
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    max_tokens: int = 256
    timeout: float = 10.0
    limit: int | None = None
    tag: str = ""

    @property
    def group_key(self) -> tuple:
        return (self.model, self.dtype, self.max_model_len, self.max_tokens)

    @property
    def exp_id(self) -> str:
        slug = self.model.replace("/", "__")
        route = "route" if self.use_routing else "noroute"
        tag = f"_{self.tag}" if self.tag else ""
        h = hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:6]
        return f"{slug}__{self.prompt}__{route}{tag}__{h}"


_MODELS = [
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
]
_PROMPTS = ["baseline", "terse", "no_fewshot"]
_ROUTING = [True, False]

DEFAULT_GRID: list[ExperimentConfig] = [
    ExperimentConfig(model=m, prompt=p, use_routing=r)
    for m in _MODELS for p in _PROMPTS for r in _ROUTING
]


# ============================================================================
# Leaderboard
# ============================================================================

_LEADERBOARD_COLS = [
    "exp_id", "model", "prompt_variant", "use_routing",
    "n_items", "N_correct", "accuracy",
    "T_total_seconds", "VRAM_GB", "RAM_GB", "score",
    "dtype", "max_model_len", "max_tokens", "tag",
]


_FAILURES_COLS = [
    "exp_id", "model", "prompt_variant", "use_routing",
    "id", "category", "question", "code", "raw",
    "exec_ok", "error", "got", "expected",
]


def append_failures(csv_path: Path, summary: dict[str, Any]) -> None:
    """Append one row per failed question to a shared failures CSV."""
    failures = [r for r in summary.get("records", []) if not r.get("correct")]
    if not failures:
        return
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FAILURES_COLS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for r in failures:
            row = {
                "exp_id": summary.get("exp_id", ""),
                "model": summary.get("model", ""),
                "prompt_variant": summary.get("prompt_variant", ""),
                "use_routing": summary.get("use_routing", ""),
                **r,
            }
            w.writerow(row)


def append_leaderboard(csv_path: Path, summary: dict[str, Any], tag: str) -> None:
    row = {k: summary.get(k, "") for k in _LEADERBOARD_COLS}
    row["tag"] = tag
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LEADERBOARD_COLS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def print_leaderboard(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    rows.sort(key=lambda r: float(r["score"] or 0), reverse=True)
    print(f"\n{'=' * 100}\nLeaderboard ({csv_path}):\n{'=' * 100}")
    print(f"{'rank':>4}  {'score':>8}  {'acc':>6}  {'T(s)':>7}  "
          f"{'VRAM':>5}  {'prompt':<14} {'route':<5}  model")
    for i, r in enumerate(rows, 1):
        print(f"{i:>4}  {float(r['score'] or 0):>8.4f}  "
              f"{float(r['accuracy'] or 0):>6.1%}  "
              f"{float(r['T_total_seconds'] or 0):>7.1f}  "
              f"{float(r['VRAM_GB'] or 0):>5.1f}  "
              f"{r['prompt_variant']:<14} "
              f"{r['use_routing']:<5}  {r['model']}")


def load_grid_file(path: str) -> list[ExperimentConfig]:
    with open(path) as f:
        data = json.load(f)
    field_names = {f.name for f in dataclasses.fields(ExperimentConfig)}
    return [ExperimentConfig(**{k: v for k, v in e.items() if k in field_names}) for e in data]


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="text2Polars experiment runner")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--grid", default=None, help="JSON grid file; overrides default grid")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    grid = load_grid_file(args.grid) if args.grid else list(DEFAULT_GRID)
    if args.limit is not None:
        grid = [dataclasses.replace(c, limit=args.limit) for c in grid]

    if not grid:
        print("No experiments to run.")
        return

    out_dir = Path(args.out_dir or f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Planned experiments: {len(grid)}")
    for i, c in enumerate(grid, 1):
        print(f"  [{i:>2}] {c.exp_id}")

    if args.dry_run:
        return

    with open(out_dir / "grid.json", "w") as f:
        json.dump([asdict(c) for c in grid], f, indent=2)
    leaderboard = out_dir / "leaderboard.csv"

    groups: dict[tuple, list[ExperimentConfig]] = {}
    for c in grid:
        groups.setdefault(c.group_key, []).append(c)

    bench_full = load_bench(args.bench)
    print(f"\nLoaded bench: {len(bench_full)} items ({args.bench})")

    overall_t0 = time.perf_counter()
    for gi, (gkey, configs) in enumerate(groups.items(), 1):
        model, dtype, mlen, mtoks = gkey
        print(f"\n{'#' * 80}\n# Group {gi}/{len(groups)}: {model} "
              f"(dtype={dtype}, mlen={mlen}, mtoks={mtoks}) — {len(configs)} experiment(s)")
        print(f"{'#' * 80}")

        t_load = time.perf_counter()
        try:
            engine = PolarsInferenceEngine(InferenceConfig(
                model=model, dtype=dtype, max_model_len=mlen, max_tokens=mtoks,
                prompt_variant=configs[0].prompt, use_routing=configs[0].use_routing,
            ))
        except Exception:
            print(f"!! Failed to load {model}:\n{traceback.format_exc()}")
            continue
        print(f"  Loaded in {time.perf_counter() - t_load:.1f}s")

        for ci, cfg in enumerate(configs, 1):
            out_path = out_dir / f"{cfg.exp_id}.json"
            if args.skip_done and out_path.exists():
                print(f"\n  [{ci}/{len(configs)}] SKIP (exists): {cfg.exp_id}")
                continue

            bench = bench_full[:cfg.limit] if cfg.limit else bench_full
            print(f"\n  [{ci}/{len(configs)}] RUN: prompt={cfg.prompt} "
                  f"routing={cfg.use_routing} items={len(bench)}")

            engine.set_prompt(prompt_variant=cfg.prompt, use_routing=cfg.use_routing)
            reset_vram_peak()

            try:
                t_start = time.perf_counter()
                n_correct, records = eval_items(engine, bench, cfg.timeout)
                total_time = time.perf_counter() - t_start
                vram = peak_vram_gb()
                ram = peak_ram_gb()
                score = compute_score(n_correct, total_time, vram, ram)

                summary = {
                    "experiment": asdict(cfg), "exp_id": cfg.exp_id,
                    "model": cfg.model, "prompt_variant": cfg.prompt,
                    "use_routing": cfg.use_routing, "dtype": cfg.dtype,
                    "max_model_len": cfg.max_model_len, "max_tokens": cfg.max_tokens,
                    "n_items": len(bench), "N_correct": n_correct,
                    "accuracy": n_correct / len(bench) if bench else 0.0,
                    "T_total_seconds": total_time, "VRAM_GB": vram, "RAM_GB": ram,
                    "score": score, "records": records,
                }
            except Exception:
                err = traceback.format_exc()
                print(f"    !! experiment failed:\n{err}")
                summary = {
                    "experiment": asdict(cfg), "exp_id": cfg.exp_id,
                    "model": cfg.model, "prompt_variant": cfg.prompt,
                    "use_routing": cfg.use_routing, "dtype": cfg.dtype,
                    "max_model_len": cfg.max_model_len, "max_tokens": cfg.max_tokens,
                    "n_items": len(bench), "N_correct": 0, "accuracy": 0.0,
                    "T_total_seconds": 0.0, "VRAM_GB": 0.0, "RAM_GB": peak_ram_gb(),
                    "score": 0.0, "records": [], "fatal_error": err,
                }

            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            append_leaderboard(leaderboard, summary, cfg.tag)
            append_failures(out_dir / "failures.csv", summary)
            n_fail = sum(1 for r in summary.get("records", []) if not r.get("correct"))
            print(f"    -> {summary['N_correct']}/{summary['n_items']} "
                  f"(acc={summary['accuracy']:.1%}, T={summary['T_total_seconds']:.1f}s, "
                  f"VRAM={summary['VRAM_GB']:.1f}GB, score={summary['score']:.4f})")
            if n_fail:
                print(f"    failures: {n_fail} (see failures.csv)")

        del engine
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    print(f"\nAll experiments done in {time.perf_counter() - overall_t0:.1f}s")
    print_leaderboard(leaderboard)


if __name__ == "__main__":
    main()
