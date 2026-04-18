# vLLM crash course (everything you need for the hackathon)

## What vLLM is

A high-throughput inference engine for LLMs. Two things matter for you:

1. **Continuous batching** — keeps the GPU saturated by streaming new requests
   into the same batch as old ones finish. Translation: 5–10× faster than HF
   `transformers.generate()` for batched workloads.
2. **PagedAttention** — KV cache is paged like OS memory, not monolithic.
   Translation: fits bigger batches in the same VRAM.

Both directly help your score (lower T, lower VRAM).

## Install

```bash
pip install vllm
# That's it. vLLM pulls in the right torch/cuda versions.
```

If you hit CUDA version mismatches, install torch first from the correct
channel, then vLLM.

## The two APIs

### 1. Offline batch (what your sweep uses)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-Coder-3B-Instruct")
params = SamplingParams(temperature=0.0, max_tokens=256)

prompts = ["Write a polars query for ...", "..."]
outputs = llm.generate(prompts, params)
for o in outputs:
    print(o.outputs[0].text)
```

Use `llm.chat(messages, params)` for chat-formatted models — it applies the
right chat template automatically.

### 2. OpenAI-compatible server (what you'll likely submit)

```bash
vllm serve Qwen/Qwen2.5-Coder-3B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096
```

Then hit it like OpenAI:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-3B-Instruct",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=256,
    temperature=0,
)
```

## Flags that matter for your score

| Flag | What it does | Impact |
|---|---|---|
| `--dtype bfloat16` | bf16 weights on Ampere+ GPUs | Halves VRAM vs fp32, faster |
| `--gpu-memory-utilization 0.85` | Fraction of VRAM vLLM can use for KV cache | Lower = less VRAM in score, but fewer concurrent reqs |
| `--max-model-len 4096` | Max context length | Lower = less KV cache = less VRAM |
| `--enforce-eager` | Skip CUDA graph compilation | Faster startup, ~5% slower per-step. Skip this flag in production |
| `--quantization awq` | Load AWQ-quantized weights | 4× smaller weights, small accuracy hit |
| `--kv-cache-dtype fp8` | KV cache in fp8 | Halves KV cache memory |
| `--enable-prefix-caching` | Cache system prompt across requests | Huge speedup if prompts share a prefix (few-shots!) |

For your task: `bfloat16 + max_model_len=2048 + enable_prefix_caching` is probably optimal.
`max_model_len=2048` is fine because: system prompt (~200 tok) + 4 few-shots (~400
tok) + schema (~50 tok) + question (~30 tok) + response (~150 tok) ≈ 830 tokens.

## SamplingParams cheat sheet

```python
SamplingParams(
    temperature=0.0,           # greedy. Best for code. Full stop.
    top_p=1.0,                 # irrelevant at temp=0
    max_tokens=256,            # cap response length
    stop=["```", "\n\n\n"],    # early termination — saves real wall time
    n=1,                       # samples per prompt. n=3 + temp=0.2 if you do self-consistency
    skip_special_tokens=True,
)
```

## Gotchas you'll hit

1. **Model doesn't fit in VRAM** → lower `gpu_memory_utilization`, lower
   `max_model_len`, or switch to AWQ version.
2. **"Model architecture X not supported"** → vLLM doesn't support every
   arch. Qwen2.5, Phi-3, Llama, Mistral all work. Check the
   [supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)
   before committing.
3. **First-run is slow** → CUDA graph compilation + weight download. Cache
   models and warm up before timing anything.
4. **Reloading models in same Python process leaks VRAM** → always use a
   fresh process per model. (This is why `sweep.py` uses subprocess.)
5. **Chat template not applied** → use `llm.chat(messages, ...)` not
   `llm.generate(prompt_string, ...)` for instruction-tuned models.

## Measuring VRAM correctly

```python
import torch
torch.cuda.reset_peak_memory_stats()
# ... load model, run generation ...
peak_gb = torch.cuda.max_memory_allocated() / 1e9
```

Call `reset_peak_memory_stats()` BEFORE loading the model to capture the full
footprint (weights + KV cache + activations). After loading mid-run tells
you only the delta.

## Fast way to check a model "works" before adding to sweep

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-Coder-1.5B-Instruct', max_model_len=2048)
out = llm.chat(
    [{'role': 'user', 'content': 'Write result = df.head() in Polars.'}],
    SamplingParams(temperature=0, max_tokens=50),
)
print(out[0].outputs[0].text)
"
```

If this errors, fix before running the sweep. 2 min of sanity-checking saves
30 min of debugging in a loop.
