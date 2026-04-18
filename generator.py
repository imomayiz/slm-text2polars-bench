"""
MLX-based Polars code generator for Apple Silicon.

Two modes:
  1) `PolarsGenerator.generate(questions)` — one-shot greedy, fastest.
  2) `PolarsGenerator.generate_with_repair(...)` — runs, catches errors,
     retries once with error feedback. Trades T for N.

Usage:
    gen = PolarsGenerator("mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit")
    out = gen.generate([
        {
            "question": "Average salary per department",
            "schema": {"df": {"dept": "Utf8", "salary": "Float64"}},
        },
    ])
    print(out[0].code)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prompt_routing import API_SNIPPETS, classify_question

# ---------- Prompt construction ----------

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

# A few high-signal examples. Keep it short — more examples = more tokens = slower.
# Pick examples that cover the common failure modes: filter, groupby+agg, sort+limit,
# join, and date handling.
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
    """Render schema as compact, model-friendly text."""
    lines = []
    for name, cols in schema.items():
        col_str = ", ".join(f"{c}: {t}" for c, t in cols.items())
        lines.append(f"{name}({col_str})")
    return "\n".join(lines)


def build_prompt(question: str, schema: dict[str, dict[str, str]],
                 prior_error: str | None = None,
                 prior_code: str | None = None) -> list[dict[str, str]]:
    """
    Build a chat-format prompt. vLLM's `chat()` API handles the model's
    specific chat template for us.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Few-shots as synthetic turns — chat models follow this pattern reliably
    for ex in FEW_SHOT:
        user = f"Schema:\n{format_schema(ex['schema'])}\n\nQuestion: {ex['question']}"
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": ex["code"]})

    # Inject targeted API snippets based on question type
    categories = classify_question(question, schema)
    snippets = "\n".join(API_SNIPPETS[cat] for cat in categories if cat in API_SNIPPETS)

    # The real question
    user_msg = f"Schema:\n{format_schema(schema)}\n\nQuestion: {question}"
    if snippets:
        user_msg += f"\n\nRelevant Polars API reference:\n{snippets}"
    if prior_error is not None:
        user_msg += (
            f"\n\nYour previous attempt raised an error. Fix it.\n"
            f"Previous code:\n{prior_code}\n"
            f"Error:\n{prior_error[:500]}"
        )
    messages.append({"role": "user", "content": user_msg})
    return messages


# ---------- Output cleaning ----------

def clean_code(raw: str) -> str:
    """
    Strip markdown fences and common preambles that small models sometimes
    emit despite instructions.
    """
    s = raw.strip()
    # Strip ```python ... ``` or ``` ... ```
    if s.startswith("```"):
        lines = s.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Strip chat template stop tokens that leak into output
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"):
        s = s.replace(tok, "")
    s = s.strip()
    # Strip "Here's the code:" style preambles (rare with strict system prompt)
    for prefix in ("Here is the code:", "Here's the code:", "Answer:", "Code:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].lstrip("\n ").strip()
    return s


# ---------- Generator ----------

@dataclass
class GenResult:
    code: str
    raw: str
    gen_time: float


class PolarsGenerator:
    def __init__(
        self,
        model: str = "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        backend: str = "auto",
    ):
        """
        Load the model once; reuse for all questions.
        backend: "auto" (detect from model name), "mlx", or "transformers".
        """
        self.model_name = model

        if backend == "auto":
            backend = "mlx" if "mlx-community" in model else "transformers"
        self.backend = backend

        if backend == "mlx":
            from mlx_lm import load
            self.model, self.tokenizer = load(model)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()

    def _generate_one_prompt(
        self,
        prompt: str,
        max_tokens: int = 256,
    ) -> str:
        if self.backend == "mlx":
            from mlx_lm import generate as mlx_generate
            return mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
        else:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            return self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

    def generate(
        self,
        items: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[GenResult]:
        """
        Generate code for a list of items. Each item:
            {"question": str, "schema": {df_name: {col: dtype, ...}, ...}}
        """
        import time

        messages_list = [build_prompt(it["question"], it["schema"]) for it in items]

        results = []
        for messages in messages_list:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            t0 = time.perf_counter()
            raw = self._generate_one_prompt(prompt, max_tokens)
            elapsed = time.perf_counter() - t0
            results.append(GenResult(
                code=clean_code(raw),
                raw=raw,
                gen_time=elapsed,
            ))

        return results

    def generate_with_repair(
        self,
        items: list[dict[str, Any]],
        dataframes_by_item: list[dict[str, "pl.DataFrame"]],  # noqa: F821
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """
        Generate → execute → if error, one repair round with the error fed back.
        Returns list of dicts with keys: code, exec_result, repaired (bool).
        """
        from executor import run_generated_code

        first = self.generate(items)

        out = []
        retry_items = []
        retry_indices = []
        retry_codes = []
        retry_errors = []

        for i, (item, gen) in enumerate(zip(items, first)):
            exec_res = run_generated_code(
                gen.code, dataframes_by_item[i], timeout=timeout,
            )
            if exec_res.ok:
                out.append({
                    "code": gen.code, "exec_result": exec_res, "repaired": False,
                    "gen_time": gen.gen_time,
                })
            else:
                out.append(None)
                retry_items.append(item)
                retry_indices.append(i)
                retry_codes.append(gen.code)
                retry_errors.append(exec_res.error or "")

        if retry_items:
            import time

            for j, (item, err, code) in enumerate(zip(retry_items, retry_errors, retry_codes)):
                idx = retry_indices[j]
                messages = build_prompt(
                    item["question"], item["schema"],
                    prior_error=err, prior_code=code,
                )
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                t0 = time.perf_counter()
                raw = self._generate_one_prompt(prompt, max_tokens=256)
                elapsed = time.perf_counter() - t0

                repaired_code = clean_code(raw)
                exec_res = run_generated_code(
                    repaired_code, dataframes_by_item[idx], timeout=timeout,
                )
                out[idx] = {
                    "code": repaired_code,
                    "exec_result": exec_res,
                    "repaired": True,
                    "gen_time": first[idx].gen_time + elapsed,
                }

        return out


# ---------- Smoke test (no vLLM needed) ----------

if __name__ == "__main__":
    # Test just the prompt-building path — actual model load requires GPU.
    prompt = build_prompt(
        question="Average revenue per region.",
        schema={"df": {"region": "Utf8", "revenue": "Float64"}},
    )
    print(f"Prompt has {len(prompt)} messages")
    print("--- last user message ---")
    print(prompt[-1]["content"])
    print()

    # Test code cleaning
    messy = '```python\nresult = df.select(pl.col("x").sum())\n```'
    print("cleaned:", repr(clean_code(messy)))

    messy2 = "Here's the code:\nresult = df.head()"
    print("cleaned:", repr(clean_code(messy2)))
