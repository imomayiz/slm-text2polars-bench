"""
Hugging Face Transformers–based Polars code generator for CUDA GPUs.

Mirrors the interface of `generator.py` (MLX/Apple Silicon) so the bench
runner can swap backends by import only.

Two modes:
  1) `PolarsGenerator.generate(items)` — sequential greedy decode (one HF
     `generate` call per item).
  2) `PolarsGenerator.generate_with_repair(...)` — runs, catches errors,
     retries failed items once with error feedback.

Usage:
    gen = PolarsGenerator("Qwen/Qwen2.5-Coder-7B-Instruct")
    out = gen.generate([
        {
            "question": "Average salary per department",
            "schema": {"df": {"dept": "Utf8", "salary": "Float64"}},
        },
    ])
    print(out[0].code)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from inference import InferenceConfig, PolarsInferenceEngine, build_messages


# ---------- Generator ----------

@dataclass
class GenResult:
    code: str
    raw: str
    gen_time: float


class PolarsGenerator:
    """
    Hugging Face Transformers wrapper for batch evaluation on CUDA.

    Loads the model once; reuse across the whole bench.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        *,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        trust_remote_code: bool = True,
    ):
        self.cfg = InferenceConfig(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=trust_remote_code,
        )
        self.engine = PolarsInferenceEngine(self.cfg)

    def generate(
        self,
        items: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[GenResult]:
        """
        Generate code for each item (sequential HF generate).
        Each item: {"question": str, "schema": {df_name: {col: dtype, ...}}}
        """
        if not items:
            return []

        batch_in = [
            {
                "question": it["question"],
                "schema": it["schema"],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            for it in items
        ]
        t0 = time.perf_counter()
        outs = self.engine.generate_batch(batch_in)
        elapsed = time.perf_counter() - t0
        per_item_time = elapsed / max(len(items), 1)

        return [
            GenResult(code=o.code, raw=o.raw, gen_time=per_item_time)
            for o in outs
        ]

    def generate_with_repair(
        self,
        items: list[dict[str, Any]],
        dataframes_by_item: list[dict[str, "pl.DataFrame"]],  # noqa: F821
        timeout: float = 10.0,
        max_tokens: int = 256,
    ) -> list[dict[str, Any]]:
        """
        Generate → execute → if error, one repair round with error feedback.

        Returns list of dicts with keys: code, exec_result, repaired, gen_time.
        """
        from executor import run_generated_code

        first = self.generate(items, max_tokens=max_tokens)

        out: list[dict[str, Any] | None] = [None] * len(items)
        retry_indices: list[int] = []
        retry_messages: list[list[dict[str, str]]] = []

        for i, (item, gen) in enumerate(zip(items, first)):
            exec_res = run_generated_code(
                gen.code, dataframes_by_item[i], timeout=timeout,
            )
            if exec_res.ok:
                out[i] = {
                    "code": gen.code,
                    "exec_result": exec_res,
                    "repaired": False,
                    "gen_time": gen.gen_time,
                }
            else:
                retry_indices.append(i)
                retry_messages.append(build_messages(
                    item["question"], item["schema"],
                    prior_error=exec_res.error or "",
                    prior_code=gen.code,
                ))

        if retry_indices:
            t0 = time.perf_counter()
            repair_outs = self.engine.generate_from_messages_list(
                retry_messages, max_tokens=max_tokens, temperature=0.0,
            )
            elapsed = time.perf_counter() - t0
            per_item_time = elapsed / max(len(retry_indices), 1)

            for j, idx in enumerate(retry_indices):
                repaired_code = repair_outs[j].code
                exec_res = run_generated_code(
                    repaired_code, dataframes_by_item[idx], timeout=timeout,
                )
                out[idx] = {
                    "code": repaired_code,
                    "exec_result": exec_res,
                    "repaired": True,
                    "gen_time": first[idx].gen_time + per_item_time,
                }

        return out  # type: ignore[return-value]


# ---------- Smoke test (prompt building only; real load needs a GPU) ----------

if __name__ == "__main__":
    msgs = build_messages(
        question="Average revenue per region.",
        schema={"df": {"region": "Utf8", "revenue": "Float64"}},
    )
    print(f"Prompt has {len(msgs)} messages")
    print("--- last user message ---")
    print(msgs[-1]["content"])
