"""
Prompt variants registry for text2Polars experiments.

Each PromptVariant is a (system_prompt, few_shot) pair with a unique name.
`inference.py` looks up the variant by name via `PROMPT_VARIANTS[name]`.

To add a new variant, append to PROMPT_VARIANTS below and reference it from
run.py or the --prompt CLI flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptVariant:
    name: str
    system_prompt: str
    few_shot: list[dict] = field(default_factory=list)
    description: str = ""


# ---------- System prompts ----------

_BASELINE_SYSTEM = """You are an expert Python programmer specializing in the Polars DataFrame library.

Your job: given a natural-language question and a DataFrame schema, write Python code \
using Polars that answers the question.

Rules — follow them exactly:
1. Output ONLY Python code. No explanations, no prose, no markdown fences.
2. Assign the final answer to a variable named `result`.
3. Use ONLY `pl` (Polars, already imported) and the DataFrames named in the schema. \
Use exactly the DataFrame variable names given in the schema — do not guess or rename them.
4. Do NOT import anything. Do NOT use pandas, .collect(), .to_pandas(), or .lazy().
5. Use the Polars eager DataFrame API: pl.col("x"), .filter(), .group_by(), .agg(), \
.with_columns(), .select(), .join(), .sort(), .head().
6. Use .item() ONLY when the result is a single scalar value (one row, one column). \
Never call .item() on a DataFrame with multiple rows or columns.
7. If the question asks for "top N" or "largest/smallest", use .sort() then .head(N) — \
keep all columns needed in the result before sorting.
8. Use only column names that appear in the schema. Do not guess data values — \
use the string values exactly as they would appear in the data based on the question context.
9. For computed columns, use .with_columns() before .group_by().
"""

_TERSE_SYSTEM = """Write Polars (eager DataFrame) code that answers the question.
Rules:
- Output ONLY code. No prose, no markdown.
- Assign the answer to `result`.
- Use `pl` and the DataFrame names from the schema. No imports. No .lazy()/.collect()/pandas.
- Use .item() only for single-scalar answers.
"""


# ---------- Few-shot examples ----------
# Each example teaches one distinct Polars API pattern.
# Ordered from simple to complex. Keep each code snippet on one logical line.

_BASELINE_FEW_SHOT = [
    # 1. Filter + scalar with .item()
    {
        "schema": {"df": {"name": "Utf8", "department": "Utf8", "salary": "Float64"}},
        "question": "What is the average salary in the Eng department?",
        "code": 'result = df.filter(pl.col("department") == "Eng").select(pl.col("salary").mean()).item()',
    },
    # 2. Group-by + single agg (returns DataFrame, no .item())
    {
        "schema": {"df": {"name": "Utf8", "department": "Utf8", "salary": "Float64"}},
        "question": "Average salary per department.",
        "code": (
            'result = df.group_by("department")'
            '.agg(pl.col("salary").mean().alias("avg_salary"))'
            '.sort("department")'
        ),
    },
    # 3. Computed column with .with_columns() before group_by
    {
        "schema": {"df": {"product": "Utf8", "region": "Utf8", "quantity": "Int64", "price": "Float64"}},
        "question": "Total revenue per region, where revenue = quantity * price.",
        "code": (
            'result = (df.with_columns((pl.col("quantity") * pl.col("price")).alias("revenue"))'
            '.group_by("region")'
            '.agg(pl.col("revenue").sum().alias("total_revenue"))'
            '.sort("region"))'
        ),
    },
    # 4. Inner join + sort + head + .item()
    {
        "schema": {
            "orders": {"order_id": "Int64", "customer_id": "Int64", "amount": "Float64"},
            "customers": {"customer_id": "Int64", "name": "Utf8", "country": "Utf8"},
        },
        "question": "Name of the customer with the largest single order.",
        "code": (
            'result = (orders.join(customers, on="customer_id")'
            '.sort("amount", descending=True)'
            '.select("name").head(1).item())'
        ),
    },
    # 5. Anti-join ("who never ...")
    {
        "schema": {
            "customers": {"customer_id": "Int64", "name": "Utf8"},
            "orders": {"order_id": "Int64", "customer_id": "Int64", "amount": "Float64"},
        },
        "question": "Names of customers who never placed an order.",
        "code": (
            'result = customers.join(orders, on="customer_id", how="anti")'
            '.select("name")'
        ),
    },
    # 6. String .str namespace
    {
        "schema": {"df": {"city": "Utf8", "population": "Int64"}},
        "question": "Cities whose name starts with 'San'.",
        "code": 'result = df.filter(pl.col("city").str.starts_with("San"))',
    },
    # 7. Date .dt namespace
    {
        "schema": {"df": {"event": "Utf8", "ts": "Datetime", "value": "Float64"}},
        "question": "Total value per year.",
        "code": (
            'result = (df.with_columns(pl.col("ts").dt.year().alias("year"))'
            '.group_by("year")'
            '.agg(pl.col("value").sum().alias("total"))'
            '.sort("year"))'
        ),
    },
    # 8. Conditional with pl.when / .then / .otherwise
    {
        "schema": {"df": {"name": "Utf8", "score": "Float64"}},
        "question": "Add a 'grade' column: 'pass' if score >= 60, else 'fail'.",
        "code": (
            'result = df.with_columns('
            'pl.when(pl.col("score") >= 60).then(pl.lit("pass"))'
            '.otherwise(pl.lit("fail")).alias("grade"))'
        ),
    },
    # 9. Window function with .over()
    {
        "schema": {"df": {"department": "Utf8", "name": "Utf8", "salary": "Float64"}},
        "question": "Rank each employee by salary within their department (highest first).",
        "code": (
            'result = df.with_columns('
            'pl.col("salary").rank(descending=True).over("department").alias("rank"))'
        ),
    },
    # 10. Group-by with multiple aggregations (list syntax)
    {
        "schema": {"df": {"category": "Utf8", "price": "Float64", "quantity": "Int64"}},
        "question": "For each category, show the average price and total quantity.",
        "code": (
            'result = df.group_by("category").agg(['
            'pl.col("price").mean().alias("avg_price"), '
            'pl.col("quantity").sum().alias("total_qty")'
            ']).sort("category")'
        ),
    },
]


# ---------- Variants ----------

PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "baseline": PromptVariant(
        name="baseline",
        system_prompt=_BASELINE_SYSTEM,
        few_shot=_BASELINE_FEW_SHOT,
        description="Production prompt (9 rules + 10 few-shot).",
    ),
    "terse": PromptVariant(
        name="terse",
        system_prompt=_TERSE_SYSTEM,
        few_shot=_BASELINE_FEW_SHOT,
        description="Minimal system prompt + same few-shot.",
    ),
    "no_fewshot": PromptVariant(
        name="no_fewshot",
        system_prompt=_BASELINE_SYSTEM,
        few_shot=[],
        description="Baseline system prompt with zero few-shot examples (ablation).",
    ),
}


def get_variant(name: str) -> PromptVariant:
    if name not in PROMPT_VARIANTS:
        available = ", ".join(sorted(PROMPT_VARIANTS))
        raise KeyError(f"Unknown prompt variant {name!r}. Available: {available}")
    return PROMPT_VARIANTS[name]
