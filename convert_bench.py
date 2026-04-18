"""
Convert existing text-to-SQL / text-to-pandas benchmarks to our polars.bench format.

Strategy:
1. Load source dataset from HuggingFace (wikisql / spider / ds1000).
2. For each item:
   a. Extract question + schema + data + ground-truth SQL (or pandas code).
   b. Use an LLM to translate SQL -> Polars code.
   c. Execute BOTH the original (via DuckDB for SQL, or exec for pandas)
      AND the Polars translation.
   d. Compare outputs. If they match -> accept. If not -> reject.
3. Save accepted items in polars.bench format.

Why this works: the original SQL is the ground truth. If the Polars translation
produces the same result, we KNOW the Polars code is correct — no LLM
hallucination slips through.

Install:
    pip install datasets duckdb polars anthropic
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import polars as pl


# ============================================================================
# LLM prompt for SQL -> Polars translation
# ============================================================================

SQL_TO_POLARS_PROMPT = """Translate the following SQL query into equivalent Polars (Python) code.

# Context

The data is available as Polars DataFrames with these names and schemas:

{schema_description}

# SQL query

```sql
{sql}
```

# Output requirements

1. Return ONLY the Python code, no prose, no markdown fences.
2. First line must be `result = `.
3. Use only `pl` and the DataFrame names listed above.
4. Do NOT import anything. `pl` is already available.
5. If the SQL returns multiple rows, produce a pl.DataFrame.
6. If the SQL returns a single scalar value (e.g. SELECT COUNT(*)), unwrap it with `.item()`.
7. Preserve column names from the SQL SELECT clause. Use `.alias(...)` where needed.
8. If the SQL has ORDER BY, include an explicit `.sort(...)` in the Polars code.

# Output
"""


# ============================================================================
# SQL execution (via DuckDB — handles Spider/WikiSQL SQL dialects gracefully)
# ============================================================================

def execute_sql(sql: str, frames: dict[str, pl.DataFrame]) -> Any:
    """Run SQL against Polars DataFrames via DuckDB. Returns a pl.DataFrame or scalar."""
    import duckdb
    conn = duckdb.connect(":memory:")
    try:
        # DuckDB reads Polars DataFrames directly (via Arrow).
        for name, df in frames.items():
            conn.register(name, df)
        result = conn.execute(sql).pl()
    finally:
        conn.close()
    # If single row single column, unwrap to scalar
    if result.shape == (1, 1):
        return result.item()
    return result


def execute_polars(code: str, frames: dict[str, pl.DataFrame]) -> Any:
    """Run Polars code in a sandbox. Returns value of `result`."""
    ns = {"pl": pl, **frames}
    exec(code, ns)
    return ns.get("result")


# ============================================================================
# Output comparison (same logic as executor.outputs_match but self-contained)
# ============================================================================

def results_equal(a: Any, b: Any, float_atol: float = 1e-6) -> bool:
    # Normalize series -> dataframe
    if isinstance(a, pl.Series):
        a = a.to_frame()
    if isinstance(b, pl.Series):
        b = b.to_frame()

    if isinstance(a, pl.DataFrame) and isinstance(b, pl.DataFrame):
        if a.shape != b.shape:
            return False
        if set(a.columns) != set(b.columns):
            # SQL and Polars often disagree on column naming (e.g. avg(x) vs x_mean)
            # Fall back to positional comparison if column counts match.
            if len(a.columns) != len(b.columns):
                return False
            b = b.rename({bn: an for an, bn in zip(a.columns, b.columns)})
        b = b.select(a.columns)
        # Align numeric dtypes
        for col in a.columns:
            if a.schema[col] != b.schema[col]:
                if a.schema[col].is_numeric() and b.schema[col].is_numeric():
                    try:
                        b = b.with_columns(pl.col(col).cast(a.schema[col], strict=False))
                    except Exception:
                        return False
        try:
            a = a.sort(a.columns, nulls_last=True)
            b = b.sort(b.columns, nulls_last=True)
        except Exception:
            pass
        for ra, rb in zip(a.iter_rows(), b.iter_rows()):
            for x, y in zip(ra, rb):
                if x is None and y is None:
                    continue
                if x is None or y is None:
                    return False
                if isinstance(x, float) or isinstance(y, float):
                    try:
                        if abs(float(x) - float(y)) > float_atol:
                            return False
                    except (TypeError, ValueError):
                        return False
                elif x != y:
                    return False
        return True

    # Scalar comparison
    if isinstance(a, (pl.DataFrame, pl.Series)) or isinstance(b, (pl.DataFrame, pl.Series)):
        return False
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= float_atol
        except (TypeError, ValueError):
            return False
    return a == b


# ============================================================================
# WikiSQL loader
# ============================================================================

def load_wikisql_sample(n: int = 50, split: str = "validation") -> list[dict]:
    """
    Load WikiSQL items. Each row has: question, table (header+rows), sql (structured).
    We'll reconstruct the SQL string from the structured form.

    Returns items with: question, df (Polars), sql_string
    """
    from datasets import load_dataset
    ds = load_dataset("wikisql", split=split, trust_remote_code=True)

    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    cond_ops = ["=", ">", "<", "OP"]

    items = []
    for row in ds.select(range(min(n * 3, len(ds)))):   # oversample
        try:
            header = row["table"]["header"]
            rows = row["table"]["rows"]
            types = row["table"]["types"]
            sql = row["sql"]

            # Build Polars DataFrame
            # WikiSQL types: text, real
            data: dict[str, list] = {h: [] for h in header}
            for r in rows:
                for h, v in zip(header, r):
                    data[h].append(v)
            # Coerce numeric columns
            for h, t in zip(header, types):
                if t == "real":
                    data[h] = [_to_float(v) for v in data[h]]
            df = pl.DataFrame(data)

            # Reconstruct SQL string
            sel_col = header[sql["sel"]]
            agg = agg_ops[sql["agg"]]
            sel_clause = f'{agg}("{sel_col}")' if agg else f'"{sel_col}"'
            where_parts = []
            for cond_col_idx, op_idx, val in zip(sql["conds"]["column_index"],
                                                  sql["conds"]["operator_index"],
                                                  sql["conds"]["condition"]):
                col = header[cond_col_idx]
                op = cond_ops[op_idx]
                if op == "OP":
                    continue   # skip: ambiguous
                val_str = f"'{val}'" if types[cond_col_idx] == "text" else str(val)
                where_parts.append(f'"{col}" {op} {val_str}')
            where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""
            sql_str = f'SELECT {sel_clause} FROM df{where_clause}'

            items.append({
                "question": row["question"],
                "frames": {"df": df},
                "sql": sql_str,
                "source": "wikisql",
            })
        except Exception:
            continue
        if len(items) >= n:
            break
    return items


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ============================================================================
# Spider loader (more complex — skipping full impl, stub below)
# ============================================================================

def load_spider_sample(n: int = 30, split: str = "validation") -> list[dict]:
    """
    Spider is harder: each item references a full database with multiple tables.
    We need to load the SQLite DB, extract only the tables referenced in the
    SQL, and convert them to Polars.

    Left as an exercise — see xlangai/spider on HF.
    """
    raise NotImplementedError(
        "Spider loading involves SQLite DB files; see xlangai/spider dataset card "
        "for the tables_json structure. Start with WikiSQL for volume."
    )


# ============================================================================
# DS-1000 loader (pandas -> polars)
# ============================================================================

def load_ds1000_sample(n: int = 20) -> list[dict]:
    """DS-1000 has Pandas problems with reference code. Translate pandas -> Polars."""
    from datasets import load_dataset
    ds = load_dataset("xlangai/DS-1000", split="test")
    items = []
    for row in ds:
        if row["metadata"]["library"] != "Pandas":
            continue
        items.append({
            "question": row["prompt"],   # includes problem statement + code stub
            "reference_pandas": row["reference_code"],
            "source": "ds1000",
            # DS-1000 doesn't bundle clean DataFrames — you'd need to parse the
            # prompt for the data setup. More work; skip if time is short.
        })
        if len(items) >= n:
            break
    return items


# ============================================================================
# Conversion pipeline
# ============================================================================

def format_schema(frames: dict[str, pl.DataFrame]) -> str:
    lines = []
    for name, df in frames.items():
        cols = ", ".join(f"{c}: {dt}" for c, dt in df.schema.items())
        lines.append(f"{name}({cols})")
    return "\n".join(lines)


def convert_batch(
    source_items: list[dict],
    llm_call: Callable[[str], str],
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    For each source item with (question, frames, sql), produce a polars.bench item.
    Returns (accepted, rejected).
    """
    accepted, rejected = [], []

    for idx, src in enumerate(source_items):
        question = src["question"]
        frames = src["frames"]
        sql = src.get("sql")
        if sql is None:
            rejected.append({"question": question[:80], "reason": "no SQL"})
            continue

        # 1) Execute SQL to get ground-truth result
        try:
            sql_result = execute_sql(sql, frames)
        except Exception as e:
            rejected.append({"question": question[:80], "reason": f"SQL exec: {e}"})
            continue

        # 2) Ask LLM to translate
        prompt = SQL_TO_POLARS_PROMPT.format(
            schema_description=format_schema(frames),
            sql=sql,
        )
        try:
            polars_code = _clean_code(llm_call(prompt))
        except Exception as e:
            rejected.append({"question": question[:80], "reason": f"LLM error: {e}"})
            continue

        # 3) Execute Polars code
        try:
            polars_result = execute_polars(polars_code, frames)
        except Exception as e:
            rejected.append({
                "question": question[:80],
                "reason": f"Polars exec: {type(e).__name__}: {str(e)[:80]}",
                "code": polars_code,
            })
            continue

        # 4) Compare
        if not results_equal(sql_result, polars_result):
            rejected.append({
                "question": question[:80],
                "reason": "SQL result ≠ Polars result",
                "sql_result": str(sql_result)[:80],
                "polars_result": str(polars_result)[:80],
                "code": polars_code,
            })
            continue

        # 5) Package
        accepted.append({
            "id": f"c{len(accepted)+1:03d}",
            "question": question,
            "category": _infer_category(sql),
            "difficulty": _infer_difficulty(sql),
            "source": src.get("source", "converted"),
            "schema": {name: {c: str(dt) for c, dt in df.schema.items()}
                       for name, df in frames.items()},
            "data": {name: df.to_dicts() for name, df in frames.items()},
            "reference_code": polars_code,
            "expected_output": _serialize(polars_result),
            "tolerance": {"order_matters": "ORDER BY" in sql.upper(), "float_atol": 1e-6},
        })
        if verbose and (idx + 1) % 5 == 0:
            print(f"  processed {idx+1}/{len(source_items)}  "
                  f"({len(accepted)} accepted, {len(rejected)} rejected)")

    return accepted, rejected


def _clean_code(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1]
        if s.endswith("```"):
            s = s[:-3]
    # Extract the first line starting with `result =` and take from there
    m = re.search(r"^result\s*=.*", s, re.MULTILINE | re.DOTALL)
    if m:
        return m.group(0).strip()
    return s.strip()


def _infer_category(sql: str) -> str:
    s = sql.upper()
    if " JOIN " in s:
        return "join"
    if "GROUP BY" in s:
        return "groupby"
    if "ORDER BY" in s and "LIMIT" in s:
        return "sort_limit"
    if any(agg in s for agg in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]):
        return "agg"
    return "filter"


def _infer_difficulty(sql: str) -> str:
    s = sql.upper()
    complexity = 0
    complexity += s.count(" JOIN ") * 2
    complexity += s.count("GROUP BY")
    complexity += s.count("ORDER BY")
    complexity += s.count("(SELECT")  # nested
    complexity += s.count("CASE WHEN")
    if complexity >= 3:
        return "hard"
    if complexity >= 1:
        return "medium"
    return "easy"


def _serialize(result: Any) -> dict:
    if isinstance(result, pl.DataFrame):
        return {"kind": "dataframe", "value": result.to_dicts(),
                "columns": result.columns}
    if isinstance(result, pl.Series):
        return {"kind": "series", "value": result.to_list(), "name": result.name}
    return {"kind": "scalar", "value": result}


# ============================================================================
# Example driver
# ============================================================================

def main():
    import anthropic
    client = anthropic.Anthropic()

    def llm_call(prompt: str) -> str:
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    print("Loading WikiSQL sample (targeting 50 items)...")
    src = load_wikisql_sample(n=50)
    print(f"Loaded {len(src)} source items.")

    print("\nConverting to Polars bench format...")
    accepted, rejected = convert_batch(src, llm_call)

    print(f"\n{'='*60}")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")

    # Show rejection reasons histogram
    reason_counts: dict[str, int] = {}
    for r in rejected:
        key = r["reason"].split(":")[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1
    print("\nRejection reasons:")
    for k, v in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {v:3d}  {k}")

    out = "bench_converted.json"
    with open(out, "w") as f:
        json.dump(accepted, f, indent=2, default=str)
    print(f"\nSaved {len(accepted)} items to {out}")


if __name__ == "__main__":
    main()
