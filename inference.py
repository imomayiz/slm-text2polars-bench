"""
vLLM inference engine (CUDA).

Used by main.py (FastAPI server) and run_bench_cuda.py (offline benchmark).

The prompt template is configurable via a named PromptVariant from prompts.py,
and API-snippet routing (classify_question) can be toggled off for ablations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_routing import API_SNIPPETS, classify_question
from prompts import PROMPT_VARIANTS, PromptVariant, get_variant


# Back-compat aliases: some callers may import these directly.
SYSTEM_PROMPT = PROMPT_VARIANTS["baseline"].system_prompt
FEW_SHOT = PROMPT_VARIANTS["baseline"].few_shot


def format_schema(schema: dict[str, dict[str, str]]) -> str:
    return "\n".join(
        f"{name}(" + ", ".join(f"{c}: {t}" for c, t in cols.items()) + ")"
        for name, cols in schema.items()
    )


def build_messages(
    question: str,
    schema: dict[str, dict[str, str]],
    variant: PromptVariant | str = "baseline",
    use_routing: bool = True,
) -> list[dict]:
    """Build chat messages for a question+schema using the given prompt variant.

    - `variant`: either a PromptVariant instance or its registered name.
    - `use_routing`: if True, inject API snippets selected by classify_question().
    """
    pv = variant if isinstance(variant, PromptVariant) else get_variant(variant)

    msgs: list[dict] = [{"role": "system", "content": pv.system_prompt}]
    for ex in pv.few_shot:
        msgs.append({
            "role": "user",
            "content": f"Schema:\n{format_schema(ex['schema'])}\n\nQuestion: {ex['question']}",
        })
        msgs.append({"role": "assistant", "content": ex["code"]})

    user_msg = f"Schema:\n{format_schema(schema)}\n\nQuestion: {question}"
    if use_routing:
        categories = classify_question(question, schema)
        snippets = "\n".join(API_SNIPPETS[cat] for cat in categories if cat in API_SNIPPETS)
        if snippets:
            user_msg += f"\n\nRelevant Polars API reference:\n{snippets}"
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def clean_code(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    for tok in ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>"):
        s = s.replace(tok, "")
    s = s.strip()
    for prefix in ("Here is the code:", "Here's the code:", "Answer:", "Code:"):
        if s.lower().startswith(prefix.lower()):
            s = s[len(prefix):].lstrip("\n ").strip()
    # Strip import lines that small models sometimes emit
    lines = s.split("\n")
    lines = [l for l in lines if not l.strip().startswith(("import ", "from "))]
    s = "\n".join(lines).strip()
    # Strip pandas/pyarrow tail calls: .collect(), .to_pandas(), .to_list(), .to_numpy()
    import re
    s = re.sub(r'\.collect\(\)', '', s)
    s = re.sub(r'\.to_pandas\(\)[^\n]*', '', s)
    s = re.sub(r'\.to_numpy\(\)[^\n]*', '', s)
    # Fix deprecated / pandas-style method names → modern Polars equivalents
    s = s.replace('.groupby(', '.group_by(')
    s = s.replace('.isin(', '.is_in(')
    s = s.replace('.startswith(', '.starts_with(')
    s = s.replace('.endswith(', '.ends_with(')
    s = s.replace('.distinct()', '.unique()')
    s = s.replace('.cumsum()', '.cum_sum()')
    s = s.replace('.cummax()', '.cum_max()')
    s = s.replace('.cummin()', '.cum_min()')
    s = s.replace('.cumprod()', '.cum_prod()')
    s = s.replace('.drop_duplicates(', '.unique(')
    # Fix .sort(reverse=...) → .sort(descending=...)
    s = re.sub(r'\.sort\(([^)]*)\breverse\b\s*=', r'.sort(\1descending=', s)
    # Fix .sort(ascending=...) → .sort(descending=...) with inverted bool
    s = re.sub(r'\.sort\(([^)]*)\bascending\b\s*=\s*True', r'.sort(\1descending=False', s)
    s = re.sub(r'\.sort\(([^)]*)\bascending\b\s*=\s*False', r'.sort(\1descending=True', s)
    s = re.sub(r'\.sort\(([^)]*)\bascending\b\s*=\s*\[', r'.sort(\1descending=[', s)
    # Fix .then("string") → .then(pl.lit("string")) — bare strings in when/then
    s = re.sub(r'\.then\("([^"]+)"\)', r'.then(pl.lit("\1"))', s)
    s = re.sub(r"\.then\('([^']+)'\)", r".then(pl.lit('\1'))", s)
    s = re.sub(r'\.otherwise\("([^"]+)"\)', r'.otherwise(pl.lit("\1"))', s)
    s = re.sub(r"\.otherwise\('([^']+)'\)", r".otherwise(pl.lit('\1'))", s)
    return s.strip()


# ---------- Engine ----------

@dataclass
class InferenceConfig:
    model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    max_tokens: int = 256
    temperature: float = 0.0
    trust_remote_code: bool = True
    prompt_variant: str = "baseline"
    use_routing: bool = True
    # vLLM-specific knobs
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    extra_vllm_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    code: str
    raw: str
    tokens: int


class PolarsInferenceEngine:
    """vLLM-backed engine; greedy generation via LLM.generate on rendered prompts."""

    def __init__(self, cfg: InferenceConfig):
        from vllm import LLM
        from transformers import AutoTokenizer

        self.cfg = cfg
        # Validate prompt variant early to fail fast.
        self.variant = get_variant(cfg.prompt_variant)

        # Tokenizer is used only to render chat templates; vLLM owns its own copy
        # internally for generation, so this is a cheap read-only instance.
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model, trust_remote_code=cfg.trust_remote_code,
        )

        llm_kwargs: dict[str, Any] = {
            "model": cfg.model,
            "dtype": cfg.dtype,
            "max_model_len": cfg.max_model_len,
            "trust_remote_code": cfg.trust_remote_code,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "enforce_eager": cfg.enforce_eager,
        }
        llm_kwargs.update(cfg.extra_vllm_kwargs)
        self.llm = LLM(**llm_kwargs)

    def set_prompt(self, prompt_variant: str | None = None, use_routing: bool | None = None) -> None:
        """Swap prompt variant / routing without reloading the model."""
        if prompt_variant is not None:
            self.variant = get_variant(prompt_variant)
            self.cfg.prompt_variant = prompt_variant
        if use_routing is not None:
            self.cfg.use_routing = use_routing

    def _render_prompt(self, msgs: list[dict]) -> str:
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        return "\n\n".join(f"### {m['role']}\n{m['content']}" for m in msgs) + "\n### assistant\n"

    def _sampling_params(self, max_tokens: int, temperature: float):
        from vllm import SamplingParams
        if temperature and temperature > 1e-6:
            return SamplingParams(temperature=temperature, max_tokens=max_tokens)
        # Greedy decoding
        return SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)

    def generate_one(
        self,
        question: str,
        schema: dict[str, dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> InferenceResult:
        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = self.cfg.temperature if temperature is None else temperature

        msgs = build_messages(
            question, schema,
            variant=self.variant,
            use_routing=self.cfg.use_routing,
        )
        prompt = self._render_prompt(msgs)
        sp = self._sampling_params(max_tokens, temperature)

        outputs = self.llm.generate([prompt], sp, use_tqdm=False)
        out = outputs[0].outputs[0]
        text = out.text
        n_tokens = len(out.token_ids) if getattr(out, "token_ids", None) is not None else 0
        return InferenceResult(code=clean_code(text), raw=text, tokens=int(n_tokens))

    def generate_batch(
        self,
        items: list[tuple[str, dict[str, dict[str, str]]]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[InferenceResult]:
        """Batched generation — the main reason to use vLLM.

        `items` is a list of (question, schema) tuples. Returns results in order.
        """
        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = self.cfg.temperature if temperature is None else temperature

        prompts = [
            self._render_prompt(
                build_messages(q, s, variant=self.variant, use_routing=self.cfg.use_routing)
            )
            for q, s in items
        ]
        sp = self._sampling_params(max_tokens, temperature)
        outputs = self.llm.generate(prompts, sp, use_tqdm=False)

        results: list[InferenceResult] = []
        for o in outputs:
            out = o.outputs[0]
            text = out.text
            n_tokens = len(out.token_ids) if getattr(out, "token_ids", None) is not None else 0
            results.append(InferenceResult(code=clean_code(text), raw=text, tokens=int(n_tokens)))
        return results
