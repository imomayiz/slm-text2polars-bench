"""
Hugging Face Transformers inference engine (CUDA/CPU).

Used by main.py (FastAPI server) and run_bench_cuda.py (offline benchmark).

The prompt template is configurable via a named PromptVariant from prompts.py,
and API-snippet routing (classify_question) can be toggled off for ablations.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    return s


# ---------- Engine ----------

def _torch_dtype_from_str(s: str):
    import torch
    if s == "auto":
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(s, torch.bfloat16)


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


@dataclass
class InferenceResult:
    code: str
    raw: str
    tokens: int


class PolarsInferenceEngine:
    """One-time HF model load; greedy generation via model.generate."""

    def __init__(self, cfg: InferenceConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        # Validate prompt variant early to fail fast.
        self.variant = get_variant(cfg.prompt_variant)

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
        return next(self.model.parameters()).device

    def set_prompt(self, prompt_variant: str | None = None, use_routing: bool | None = None) -> None:
        """Swap prompt variant / routing without reloading the model."""
        if prompt_variant is not None:
            self.variant = get_variant(prompt_variant)
            self.cfg.prompt_variant = prompt_variant
        if use_routing is not None:
            self.cfg.use_routing = use_routing

    def generate_one(
        self,
        question: str,
        schema: dict[str, dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> InferenceResult:
        import torch

        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = self.cfg.temperature if temperature is None else temperature

        msgs = build_messages(
            question, schema,
            variant=self.variant,
            use_routing=self.cfg.use_routing,
        )
        tok = self.tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,)
        else:
            prompt = "\n\n".join(f"### {m['role']}\n{m['content']}" for m in msgs) + "\n### assistant\n"

        max_prompt_len = max(128, self.cfg.max_model_len - max_tokens)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        gen_kw: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tok.pad_token_id,
        }
        if temperature and temperature > 1e-6:
            gen_kw["do_sample"] = True
            gen_kw["temperature"] = temperature
        else:
            gen_kw["do_sample"] = False

        with torch.inference_mode():
            out_ids = self.model.generate(**enc, **gen_kw)

        in_len = enc["input_ids"].shape[1]
        new_ids = out_ids[0, in_len:]
        text = tok.decode(new_ids, skip_special_tokens=True)

        return InferenceResult(code=clean_code(text), raw=text, tokens=int(new_ids.numel()))
