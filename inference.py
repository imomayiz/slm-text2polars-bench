"""
vLLM inference engine 
    python -c "from inference import *; e = PolarsInferenceEngine(InferenceConfig()); \
        print(e.generate_one(question='sum of x', schema={'df': {'x': 'Int64'}}).code)"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------- Prompt template ----------

SYSTEM_PROMPT = """You are an expert Python programmer specializing in the Polars DataFrame library.

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

FEW_SHOT = [
    # Scalar result with .item()
    {
        "schema": {
            "df": {"name": "Utf8", "department": "Utf8", "salary": "Float64"}
        },
        "question": "What is the average salary in the Eng department?",
        "code": 'result = df.filter(pl.col("department") == "Eng").select(pl.col("salary").mean()).item()',
    },
    # Group-by returning a DataFrame (no .item())
    {
        "schema": {
            "df": {"name": "Utf8", "department": "Utf8", "salary": "Float64"}
        },
        "question": "Average salary per department.",
        "code": (
            'result = df.group_by("department")'
            '.agg(pl.col("salary").mean().alias("avg_salary"))'
            '.sort("department")'
        ),
    },
    # Computed column with .with_columns() before group_by
    {
        "schema": {
            "df": {"product": "Utf8", "region": "Utf8", "quantity": "Int64", "price": "Float64"}
        },
        "question": "Total revenue per region, where revenue = quantity * price.",
        "code": (
            'result = (df.with_columns((pl.col("quantity") * pl.col("price")).alias("revenue"))'
            '.group_by("region")'
            '.agg(pl.col("revenue").sum().alias("total_revenue"))'
            '.sort("region"))'
        ),
    },
    # Join + sort + head (keep columns before sorting)
    {
        "schema": {
            "orders":    {"order_id": "Int64", "customer_id": "Int64", "amount": "Float64"},
            "customers": {"customer_id": "Int64", "name": "Utf8", "country": "Utf8"},
        },
        "question": "Name of the customer with the largest single order.",
        "code": (
            'result = (orders.join(customers, on="customer_id")'
            '.sort("amount", descending=True)'
            '.select("name").head(1).item())'
        ),
    },
    # Scalar count with .n_unique()
    {
        "schema": {
            "df": {"user_id": "Int64", "event": "Utf8", "ts": "Datetime"}
        },
        "question": "Number of distinct users who had a 'purchase' event.",
        "code": 'result = df.filter(pl.col("event") == "purchase").select(pl.col("user_id").n_unique()).item()',
    },
]


def format_schema(schema: dict[str, dict[str, str]]) -> str:
    return "\n".join(
        f"{name}(" + ", ".join(f"{c}: {t}" for c, t in cols.items()) + ")"
        for name, cols in schema.items()
    )


from prompt_routing import API_SNIPPETS, classify_question


def build_messages(
    question: str,
    schema: dict[str, dict[str, str]],
    prior_error: str | None = None,
    prior_code: str | None = None,
) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT:
        msgs.append({
            "role": "user",
            "content": f"Schema:\n{format_schema(ex['schema'])}\n\nQuestion: {ex['question']}",
        })
        msgs.append({"role": "assistant", "content": ex["code"]})

    # Inject targeted API snippets based on question type
    categories = classify_question(question, schema)
    snippets = "\n".join(API_SNIPPETS[cat] for cat in categories if cat in API_SNIPPETS)

    user_msg = f"Schema:\n{format_schema(schema)}\n\nQuestion: {question}"
    if snippets:
        user_msg += f"\n\nRelevant Polars API reference:\n{snippets}"
    if prior_error is not None:
        user_msg += (
            f"\n\nYour previous attempt raised an error. Fix it.\n"
            f"Previous code:\n{prior_code}\n"
            f"Error:\n{prior_error[:500]}"
        )
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def clean_code(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Strip chat template stop tokens that leak into output
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"):
        s = s.replace(tok, "")
    s = s.strip()
    for prefix in ("Here is the code:", "Here's the code:", "Answer:", "Code:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].lstrip("\n ").strip()
    return s


# ---------- Engine ----------

@dataclass
class InferenceConfig:
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.85
    max_tokens: int = 256
    temperature: float = 0.0
    stop: tuple[str, ...] = ("```", "\n\n\n", "Question:", "# End")
    enable_prefix_caching: bool = True
    tensor_parallel_size: int = 1


@dataclass
class InferenceResult:
    code: str
    raw: str
    tokens: int


class PolarsInferenceEngine:
    """One-time model load. Thread-safe for vLLM's internal batching."""

    def __init__(self, cfg: InferenceConfig):
        from vllm import LLM
        self.cfg = cfg
        self.llm = LLM(
            model=cfg.model,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enable_prefix_caching=cfg.enable_prefix_caching,
            tensor_parallel_size=cfg.tensor_parallel_size,
            trust_remote_code=True,
        )

    def _params(self, max_tokens: int | None, temperature: float | None):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=self.cfg.temperature if temperature is None else temperature,
            top_p=1.0,
            max_tokens=max_tokens or self.cfg.max_tokens,
            stop=list(self.cfg.stop),
        )

    def generate_one(
        self,
        question: str | None = None,
        schema: dict[str, dict[str, str]] | None = None,
        prompt_override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> InferenceResult:
        params = self._params(max_tokens, temperature)

        if prompt_override is not None:
            outputs = self.llm.generate([prompt_override], params, use_tqdm=False)
        else:
            if question is None or schema is None:
                raise ValueError("Need either prompt_override or (question + schema)")
            msgs = build_messages(question, schema)
            outputs = self.llm.chat([msgs], params, use_tqdm=False)

        out = outputs[0].outputs[0]
        return InferenceResult(
            code=clean_code(out.text),
            raw=out.text,
            tokens=len(out.token_ids),
        )

    def generate_batch(self, items: list[dict[str, Any]]) -> list[InferenceResult]:
        """Batch over a list of dicts. Each dict: question+schema or prompt_override."""
        if not items:
            return []

        # Split into prompt-override vs structured; vLLM can't mix chat+completion
        # in one call, so we handle them separately.
        structured_idx = []
        structured_msgs = []
        raw_idx = []
        raw_prompts = []
        for i, it in enumerate(items):
            if it.get("prompt_override"):
                raw_idx.append(i)
                raw_prompts.append(it["prompt_override"])
            else:
                structured_idx.append(i)
                structured_msgs.append(build_messages(it["question"], it["schema"]))

        # Use per-call sampling params (items might override)
        results: list[InferenceResult | None] = [None] * len(items)

        if structured_msgs:
            params = self._params(
                items[structured_idx[0]].get("max_tokens"),
                items[structured_idx[0]].get("temperature"),
            )
            outs = self.llm.chat(structured_msgs, params, use_tqdm=False)
            for i, o in zip(structured_idx, outs):
                t = o.outputs[0]
                results[i] = InferenceResult(
                    code=clean_code(t.text), raw=t.text, tokens=len(t.token_ids),
                )

        if raw_prompts:
            params = self._params(
                items[raw_idx[0]].get("max_tokens"),
                items[raw_idx[0]].get("temperature"),
            )
            outs = self.llm.generate(raw_prompts, params, use_tqdm=False)
            for i, o in zip(raw_idx, outs):
                t = o.outputs[0]
                results[i] = InferenceResult(
                    code=clean_code(t.text), raw=t.text, tokens=len(t.token_ids),
                )

        return [r for r in results if r is not None]
