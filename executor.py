"""
Sandboxed executor for model-generated Polars code.

Why subprocess isolation:
- Generated code can infinite-loop, OOM, or segfault. A crash in the same
  process takes down your whole eval run. Subprocess = blast radius of 1.
- Enforces a hard wall-clock timeout.
- Guarantees a clean namespace per call (no state leaks between questions).

Usage:
    result = run_generated_code(
        code="result = df.filter(pl.col('x') > 5).select(pl.col('y').sum())",
        dataframes={"df": my_polars_df},
        timeout=10.0,
    )
    if result.ok:
        print(result.value)
    else:
        print(result.error)
"""

from __future__ import annotations

import base64
import io
import pickle
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass
class ExecResult:
    ok: bool
    value: Any = None            # the value bound to `result` in the generated code
    error: str | None = None     # stringified traceback or timeout/OOM reason
    wall_time: float = 0.0       # seconds, including subprocess spawn overhead
    exit_code: int | None = None


# Runner template. The child process:
#   1) unpickles dataframes from stdin
#   2) execs the generated code with `pl`, `df`, and any named dataframes in scope
#   3) pickles back whatever was bound to `result`
# We use base64+pickle over stdin/stdout to keep it dependency-free and binary-safe.
_RUNNER = textwrap.dedent("""
    import base64, pickle, sys, traceback, io
    import polars as pl

    def _main():
        raw = sys.stdin.buffer.read()
        payload = pickle.loads(base64.b64decode(raw))
        code = payload["code"]
        frames = payload["frames"]  # dict[str, bytes] -- parquet-serialized

        # Rehydrate dataframes in-process (fast, avoids repickling Polars objects
        # which can be fragile across versions).
        ns = {"pl": pl}
        for name, buf in frames.items():
            ns[name] = pl.read_parquet(io.BytesIO(buf))

        try:
            exec(code, ns)
        except Exception:
            sys.stdout.write("ERR::" + traceback.format_exc())
            sys.stdout.flush()
            sys.exit(2)

        result = ns.get("result", None)

        # Serialize result. DataFrames/Series go via parquet for fidelity;
        # everything else via pickle.
        if isinstance(result, pl.DataFrame):
            buf = io.BytesIO()
            result.write_parquet(buf)
            out = {"kind": "dataframe", "value": buf.getvalue()}
        elif isinstance(result, pl.Series):
            buf = io.BytesIO()
            result.to_frame().write_parquet(buf)
            out = {"kind": "series", "value": buf.getvalue(), "name": result.name}
        else:
            out = {"kind": "python", "value": result}

        sys.stdout.buffer.write(b"OK::" + base64.b64encode(pickle.dumps(out)))
        sys.stdout.flush()

    _main()
""")


def _serialize_frames(dataframes: dict[str, pl.DataFrame]) -> dict[str, bytes]:
    """Parquet-serialize each DataFrame. Parquet is fast + version-stable."""
    out = {}
    for name, df in dataframes.items():
        buf = io.BytesIO()
        df.write_parquet(buf)
        out[name] = buf.getvalue()
    return out


def run_generated_code(
    code: str,
    dataframes: dict[str, pl.DataFrame],
    timeout: float = 10.0,
    python_executable: str | None = None,
) -> ExecResult:
    """
    Execute `code` in a fresh subprocess with `dataframes` bound to their names.
    The code is expected to assign its answer to a variable named `result`.

    `pl` is always available. Dataframes are bound to the names given in the
    `dataframes` dict (e.g. pass {"df": ...} and the code uses `df`).
    """
    t0 = time.perf_counter()

    payload = {"code": code, "frames": _serialize_frames(dataframes)}
    blob = base64.b64encode(pickle.dumps(payload))

    try:
        proc = subprocess.run(
            [python_executable or sys.executable, "-c", _RUNNER],
            input=blob,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ExecResult(
            ok=False,
            error=f"Timeout after {timeout}s",
            wall_time=time.perf_counter() - t0,
            exit_code=None,
        )

    wall = time.perf_counter() - t0

    stdout = proc.stdout
    if proc.returncode == 0 and stdout.startswith(b"OK::"):
        try:
            out = pickle.loads(base64.b64decode(stdout[4:]))
        except Exception as e:
            return ExecResult(ok=False, error=f"Deserialize failed: {e}",
                              wall_time=wall, exit_code=proc.returncode)

        if out["kind"] == "dataframe":
            val = pl.read_parquet(io.BytesIO(out["value"]))
        elif out["kind"] == "series":
            val = pl.read_parquet(io.BytesIO(out["value"])).to_series()
            val = val.rename(out.get("name", ""))
        else:
            val = out["value"]
        return ExecResult(ok=True, value=val, wall_time=wall, exit_code=0)

    # Error path
    err_text = stdout.decode("utf-8", errors="replace")
    if err_text.startswith("ERR::"):
        err_text = err_text[5:]
    else:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        err_text = err_text + "\n" + stderr

    return ExecResult(
        ok=False,
        error=err_text.strip() or f"Process exited with code {proc.returncode}",
        wall_time=wall,
        exit_code=proc.returncode,
    )


# ---------- Output comparison ----------

def outputs_match(
    pred: Any,
    gold: Any,
    *,
    order_matters: bool = False,
    float_atol: float = 1e-6,
    check_column_names: bool = True,
) -> bool:
    """
    Tolerant equality for Polars results.

    Handles:
      - DataFrame vs DataFrame (row order optional, column order ignored,
        compatible int/float dtypes auto-cast)
      - Series vs Series (and Series vs single-column DataFrame)
      - Scalar vs scalar (with float tolerance)
      - None
    """
    # Normalize: Series -> single-column DataFrame
    if isinstance(pred, pl.Series):
        pred = pred.to_frame()
    if isinstance(gold, pl.Series):
        gold = gold.to_frame()

    if isinstance(gold, pl.DataFrame):
        if not isinstance(pred, pl.DataFrame):
            return False
        return _df_match(pred, gold, order_matters, float_atol, check_column_names)

    return _scalar_match(pred, gold, float_atol)


def _df_match(
    pred: pl.DataFrame,
    gold: pl.DataFrame,
    order_matters: bool,
    float_atol: float,
    check_column_names: bool,
) -> bool:
    if pred.shape != gold.shape:
        return False

    if check_column_names:
        if set(pred.columns) != set(gold.columns):
            return False
        pred = pred.select(gold.columns)
    else:
        pred = pred.rename({p: g for p, g in zip(pred.columns, gold.columns)})

    # Align numeric dtypes to avoid Int32 vs Int64 / Float32 vs Float64 false negatives
    for col in gold.columns:
        gdt, pdt = gold.schema[col], pred.schema[col]
        if gdt != pdt:
            if gdt.is_numeric() and pdt.is_numeric():
                try:
                    pred = pred.with_columns(pl.col(col).cast(gdt, strict=False))
                except Exception:
                    return False
            else:
                return False

    if not order_matters:
        try:
            pred = pred.sort(pred.columns, nulls_last=True)
            gold = gold.sort(gold.columns, nulls_last=True)
        except Exception:
            pass  # unsortable types; fall through

    for p_row, g_row in zip(pred.iter_rows(), gold.iter_rows()):
        for p, g in zip(p_row, g_row):
            if not _scalar_match(p, g, float_atol):
                return False
    return True


def _scalar_match(a: Any, b: Any, atol: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= atol
        except (TypeError, ValueError):
            return False
    try:
        return bool(a == b)
    except Exception:
        return False


# ---------- Quick self-test ----------

if __name__ == "__main__":
    df = pl.DataFrame({
        "dept": ["Eng", "Eng", "Sales", "Sales", "Ops"],
        "salary": [100.0, 120.0, 90.0, 95.0, 85.0],
    })

    # Happy path
    r = run_generated_code(
        code="result = df.group_by('dept').agg(pl.col('salary').mean()).sort('dept')",
        dataframes={"df": df},
    )
    print("OK case:", r.ok, "time=", round(r.wall_time, 3))
    print(r.value)

    gold = df.group_by("dept").agg(pl.col("salary").mean()).sort("dept")
    print("matches gold:", outputs_match(r.value, gold))

    # Wrong row order but same content — should still match
    gold_shuffled = gold.sample(fraction=1.0, shuffle=True, seed=0)
    print("matches shuffled gold:", outputs_match(r.value, gold_shuffled))

    # Runtime error
    r2 = run_generated_code(
        code="result = df.filter(pl.col('nonexistent') > 0)",
        dataframes={"df": df},
    )
    print("\nerror case: ok=", r2.ok)
    print("error head:", (r2.error or "")[:140])

    # Infinite loop -> timeout
    r3 = run_generated_code(
        code="import time\nwhile True: time.sleep(0.1)\nresult = 1",
        dataframes={"df": df},
        timeout=1.5,
    )
    print("\ntimeout case: ok=", r3.ok, "error=", r3.error)

    # Scalar result
    r4 = run_generated_code(
        code="result = df.select(pl.col('salary').mean()).item()",
        dataframes={"df": df},
    )
    print("\nscalar case:", r4.value, "matches 98.0:", outputs_match(r4.value, 98.0))
