"""
Hugging Face Transformers inference engine (CUDA/CPU).

    python -c "from inference import *; e = PolarsInferenceEngine(InferenceConfig()); \\
        print(e.generate_one(question='sum of x', schema={'df': {'x': 'Int64'}}).code)"
"""
from __future__ import annotations

from dataclasses import dataclass
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

def _torch_dtype_from_str(s: str):
    import torch
    if s == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(s, torch.bfloat16)


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
    trust_remote_code: bool = True


@dataclass
class InferenceResult:
    code: str
    raw: str
    tokens: int


class PolarsInferenceEngine:
    """One-time HF model load; greedy / temperature sampling via `model.generate`."""

    def __init__(self, cfg: InferenceConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        td = _torch_dtype_from_str(cfg.dtype)
        load_kw: dict[str, Any] = {"trust_remote_code": cfg.trust_remote_code}
        if torch.cuda.is_available():
            load_kw["device_map"] = "auto"
            load_kw["torch_dtype"] = td if td != "auto" else "auto"
        else:
            load_kw["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model, **load_kw)
        self.model.eval()

    @property
    def device(self):
        import torch
        return next(self.model.parameters()).device

    def _max_prompt_length(self, max_new_tokens: int) -> int:
        cap = self.cfg.max_model_len
        try:
            ml = self.tokenizer.model_max_length
            if ml is not None and ml < 10_000_000:
                cap = min(cap, ml)
        except Exception:
            pass
        return max(128, cap - max_new_tokens)

    def _apply_chat_template(self, messages: list[dict]) -> str:
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        parts: list[str] = []
        for m in messages:
            parts.append(f"### {m['role']}\n{m['content']}")
        return "\n\n".join(parts) + "\n### assistant\n"

    def _generate_text(
        self,
        prompt: str,
        max_tokens: int | None,
        temperature: float | None,
    ) -> InferenceResult:
        import torch

        max_tokens = max_tokens if max_tokens is not None else self.cfg.max_tokens
        temperature = self.cfg.temperature if temperature is None else temperature

        max_len = self._max_prompt_length(max_tokens)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature is not None and temperature > 1e-6:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
        else:
            gen_kw["do_sample"] = False

        with torch.inference_mode():
            out_ids = self.model.generate(**enc, **gen_kw)

        in_len = enc["input_ids"].shape[1]
        new_ids = out_ids[0, in_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        return InferenceResult(
            code=clean_code(text),
            raw=text,
            tokens=int(new_ids.numel()),
        )

    def generate_one(
        self,
        question: str | None = None,
        schema: dict[str, dict[str, str]] | None = None,
        prompt_override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> InferenceResult:
        if prompt_override is not None:
            return self._generate_text(prompt_override, max_tokens, temperature)
        if question is None or schema is None:
            raise ValueError("Need either prompt_override or (question + schema)")
        msgs = build_messages(question, schema)
        prompt = self._apply_chat_template(msgs)
        return self._generate_text(prompt, max_tokens, temperature)

    def generate_from_messages_list(
        self,
        messages_list: list[list[dict]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[InferenceResult]:
        """One forward per item (same semantics as batched vLLM chat, slower on HF)."""
        return [
            self._generate_text(self._apply_chat_template(msgs), max_tokens, temperature)
            for msgs in messages_list
        ]

    def generate_batch(self, items: list[dict[str, Any]]) -> list[InferenceResult]:
        """Each dict: question+schema, prompt_override, or messages (+ optional max_tokens, temperature)."""
        if not items:
            return []

        results: list[InferenceResult] = []
        for it in items:
            mt = it.get("max_tokens") or self.cfg.max_tokens
            temp = it.get("temperature")
            if temp is None:
                temp = self.cfg.temperature

            if it.get("prompt_override") is not None:
                results.append(self._generate_text(it["prompt_override"], mt, temp))
            elif it.get("messages") is not None:
                prompt = self._apply_chat_template(it["messages"])
                results.append(self._generate_text(prompt, mt, temp))
            else:
                msgs = build_messages(it["question"], it["schema"])
                prompt = self._apply_chat_template(msgs)
                results.append(self._generate_text(prompt, mt, temp))

        return results
