"""
vLLM-based Polars code generator for CUDA GPUs.

Mirrors the interface of `generator.py` (MLX/Apple Silicon) so the bench
runner can swap backends by import only.

Two modes:
  1) `PolarsGenerator.generate(items)` — batched greedy decode.
     vLLM batches the whole list in one forward pass, which is the main
     speed win vs. the MLX one-shot loop.
  2) `PolarsGenerator.generate_with_repair(...)` — runs, catches errors,
     retries failed items once with error feedback. Repair pass is also
     batched across all failing items.

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

# Shared prompt/cleaning logic lives in inference.py (the vLLM path).
# Reusing avoids drift between the API server and the bench runner.
from inference import build_messages, clean_code


# ---------- Generator ----------

@dataclass
class GenResult:
    code: str
    raw: str
    gen_time: float


class PolarsGenerator:
    """
    vLLM wrapper for batch evaluation on CUDA.

    Loads the model once; reuse across the whole bench. vLLM internally
    handles continuous batching of the chat calls below.
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
        from vllm import LLM

        self.model_name = model
        self.llm = LLM(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=trust_remote_code,
        )

    def _sampling_params(self, max_tokens: int, temperature: float):
        from vllm import SamplingParams
        return SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            stop=["```", "\n\n\n", "Question:", "# End"],
        )

    def generate(
        self,
        items: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> list[GenResult]:
        """
        Generate code for a list of items in a single batched vLLM call.
        Each item: {"question": str, "schema": {df_name: {col: dtype, ...}}}
        """
        if not items:
            return []

        messages_list = [build_messages(it["question"], it["schema"]) for it in items]
        params = self._sampling_params(max_tokens, temperature)

        t0 = time.perf_counter()
        outputs = self.llm.chat(messages_list, params, use_tqdm=False)
        elapsed = time.perf_counter() - t0

        # vLLM returns outputs in input order. Amortize batch time across items
        # so the per-item "gen_time" field stays meaningful for reporting.
        per_item_time = elapsed / max(len(items), 1)

        results: list[GenResult] = []
        for o in outputs:
            raw = o.outputs[0].text
            results.append(GenResult(
                code=clean_code(raw),
                raw=raw,
                gen_time=per_item_time,
            ))
        return results

    def generate_with_repair(
        self,
        items: list[dict[str, Any]],
        dataframes_by_item: list[dict[str, "pl.DataFrame"]],  # noqa: F821
        timeout: float = 10.0,
        max_tokens: int = 256,
    ) -> list[dict[str, Any]]:
        """
        Generate → execute → if error, one batched repair round with each
        failing item's error fed back into the prompt.

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
            from vllm import SamplingParams
            params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=max_tokens,
                stop=["```", "\n\n\n", "Question:", "# End"],
            )
            t0 = time.perf_counter()
            outputs = self.llm.chat(retry_messages, params, use_tqdm=False)
            elapsed = time.perf_counter() - t0
            per_item_time = elapsed / max(len(retry_indices), 1)

            for j, idx in enumerate(retry_indices):
                raw = outputs[j].outputs[0].text
                repaired_code = clean_code(raw)
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
