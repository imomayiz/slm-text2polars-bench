"""
Prompt variants registry for text2Polars experiments.

Each PromptVariant is a (system_prompt, few_shot) pair with a unique name.
`inference.py` looks up the variant by name via `PROMPT_VARIANTS[name]`.

To add a new variant, append to PROMPT_VARIANTS below and reference it from
`run_experiments.py` or the --prompt CLI flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptVariant:
    name: str
    system_prompt: str
    few_shot: list[dict] = field(default_factory=list)
    description: str = ""


# ---------- baseline: current production prompt ----------

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

_BASELINE_FEW_SHOT = [
    {
        "schema": {"df": {"name": "Utf8", "department": "Utf8", "salary": "Float64"}},
        "question": "What is the average salary in the Eng department?",
        "code": 'result = df.filter(pl.col("department") == "Eng").select(pl.col("salary").mean()).item()',
    },
    {
        "schema": {"df": {"name": "Utf8", "department": "Utf8", "salary": "Float64"}},
        "question": "Average salary per department.",
        "code": (
            'result = df.group_by("department")'
            '.agg(pl.col("salary").mean().alias("avg_salary"))'
            '.sort("department")'
        ),
    },
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
    {
        "schema": {"df": {"user_id": "Int64", "event": "Utf8", "ts": "Datetime"}},
        "question": "Number of distinct users who had a 'purchase' event.",
        "code": 'result = df.filter(pl.col("event") == "purchase").select(pl.col("user_id").n_unique()).item()',
    },
]


# ---------- terse: minimal system prompt, same few-shot ----------

_TERSE_SYSTEM = """Write Polars (eager DataFrame) code that answers the question.
Rules:
- Output ONLY code. No prose, no markdown.
- Assign the answer to `result`.
- Use `pl` and the DataFrame names from the schema. No imports. No .lazy()/.collect()/pandas.
- Use .item() only for single-scalar answers.
"""


# ---------- no_fewshot: system-only, useful as an ablation ----------

_NO_FEWSHOT_SYSTEM = _BASELINE_SYSTEM


# ---------- strict_columns: extra emphasis on literal column names ----------

_STRICT_COLUMNS_SYSTEM = _BASELINE_SYSTEM + """
10. Every column referenced via pl.col("...") must appear verbatim in the schema. \
Do not invent columns and do not rename them.
11. When joining, always pass on="<key>" explicitly; do not rely on implicit join keys.
"""


PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "baseline": PromptVariant(
        name="baseline",
        system_prompt=_BASELINE_SYSTEM,
        few_shot=_BASELINE_FEW_SHOT,
        description="Production prompt (9 rules + 5 few-shot).",
    ),
    "terse": PromptVariant(
        name="terse",
        system_prompt=_TERSE_SYSTEM,
        few_shot=_BASELINE_FEW_SHOT,
        description="Minimal 5-bullet system prompt + same few-shot.",
    ),
    "no_fewshot": PromptVariant(
        name="no_fewshot",
        system_prompt=_NO_FEWSHOT_SYSTEM,
        few_shot=[],
        description="Baseline system prompt with zero few-shot examples (ablation).",
    ),
    "strict_columns": PromptVariant(
        name="strict_columns",
        system_prompt=_STRICT_COLUMNS_SYSTEM,
        few_shot=_BASELINE_FEW_SHOT,
        description="Baseline + extra rules on literal column names and join keys.",
    ),
}


def get_variant(name: str) -> PromptVariant:
    if name not in PROMPT_VARIANTS:
        available = ", ".join(sorted(PROMPT_VARIANTS))
        raise KeyError(f"Unknown prompt variant {name!r}. Available: {available}")
    return PROMPT_VARIANTS[name]
