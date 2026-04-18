# text2Polars

Natural-language to Polars code generation. A small LLM generates executable Polars DataFrame code from plain English questions, served via FastAPI.

## Setup

```bash
pip install transformers accelerate polars torch fastapi uvicorn requests

# Serve the model
uvicorn main:app --host 0.0.0.0 --port 8000
```

Hardware: NVIDIA GeForce RTX 5090 (RunPod VM).

## Synthetic Benchmark

30 question-code pairs generated with ChatGPT, covering 11 Polars operation categories across 3 difficulty levels.

![Items per Category](analysis_outputs/categories.png)
![Items per Difficulty](analysis_outputs/difficulties.png)
![Category vs Difficulty](analysis_outputs/heatmap.png)

```bash
python make_benchmark.py   # generates bench.json
```

**Next**: convert existing Text2SQL datasets to Text2Polars — SQL-to-Polars translation is more reliable than generating instruction-code pairs from scratch.

## Prompt Engineering

Three prompt variants tested:

| Variant | System Prompt | Few-Shot Examples |
|---|---|---|
| `baseline` | 9-rule detailed prompt | 10 examples |
| `terse` | Minimal 4-rule prompt | 10 examples |
| `no_fewshot` | 9-rule detailed prompt | 0 (ablation) |

10 few-shot examples cover: filter+scalar, group-by, computed columns, inner join, anti-join, string `.str`, date `.dt`, `pl.when/then`, window `.over()`, and multi-aggregation.

## Pseudo-RAG (API Snippet Routing)

Instead of retrieving from a vector store, `prompt_routing.py` classifies each question using regex patterns on the question text and schema column dtypes, then injects targeted Polars API snippets into the prompt.

Categories: `string`, `join`, `window`, `date`, `when_then`, `null_nan`, `concat`, `groupby_multi`.

Schema-aware signals:
- Multi-table schema automatically triggers the join snippet
- Date/Datetime columns trigger the date snippet even without date keywords
- String columns + name-like filters trigger the string snippet

## Experiments & Results

`run.py` is an eval framework that mirrors the platform's evaluation backend:
- Calls the model to generate Polars code
- Executes generated code in a sandboxed Python subprocess
- Compares output against gold answers (tolerant: row order, float precision, dtype casting)
- Tracks VRAM, wall time, and computes accuracy
- Scores using: `Score = N / (T * VRAM^0.1 * RAM^0.01)`

```bash
# Run the full grid (5 models x 3 prompts x 2 routing = 30 experiments)
python run.py --bench bench.json

# Run a custom grid
python run.py --bench bench.json --grid samples/grid.json

# Quick test
python run.py --bench bench.json --limit 5 --dry-run
```

Models evaluated:

| Model | Parameters |
|---|---|
| `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1.5B |
| `Qwen/Qwen3-4B-Instruct-2507` | 4B |
| `google/gemma-4-E2B-it` | 2B |
| `google/gemma-4-E4B-it` | 4B |
| `Qwen/Qwen2.5-Coder-7B-Instruct` | 7B |

### Leaderboard (30-item bench, RTX 5090, bfloat16)

| Rank | Score  | Acc   | T (s) | VRAM (GB) | Prompt      | Routing | Model                               |
|-----:|-------:|------:|------:|----------:|-------------|---------|-------------------------------------|
|    1 | 0.3661 | 76.7% |  49.6 |       8.4 | baseline    | True    | Qwen/Qwen3-4B-Instruct-2507         |
|    2 | 0.3444 | 70.0% |  48.2 |       8.3 | terse       | True    | Qwen/Qwen3-4B-Instruct-2507         |
|    3 | 0.3167 | 70.0% |  52.4 |       8.4 | baseline    | False   | Qwen/Qwen3-4B-Instruct-2507         |
|    4 | 0.3158 | 50.0% |  41.7 |       3.2 | baseline    | False   | Qwen/Qwen2.5-Coder-1.5B-Instruct    |
|    5 | 0.3051 | 46.7% |  40.3 |       3.2 | terse       | True    | Qwen/Qwen2.5-Coder-1.5B-Instruct    |
|    6 | 0.2950 | 50.0% |  44.6 |       3.2 | baseline    | True    | Qwen/Qwen2.5-Coder-1.5B-Instruct    |
|    7 | 0.2643 | 56.7% |  50.9 |       8.2 | no_fewshot  | True    | Qwen/Qwen3-4B-Instruct-2507         |
|    8 | 0.2012 | 33.3% |  43.7 |       3.2 | no_fewshot  | True    | Qwen/Qwen2.5-Coder-1.5B-Instruct    |
|    9 | 0.1719 | 60.0% |  80.7 |      10.5 | terse       | True    | google/gemma-4-E2B-it               |
|   10 | 0.1581 | 73.3% | 102.2 |      16.4 | baseline    | False   | google/gemma-4-E4B-it               |
|   11 | 0.1552 | 53.3% |  79.4 |      10.6 | baseline    | False   | google/gemma-4-E2B-it               |
|   12 | 0.1524 | 70.0% | 101.3 |      16.3 | terse       | True    | google/gemma-4-E4B-it               |
|   13 | 0.1479 | 70.0% | 104.3 |      16.5 | baseline    | True    | google/gemma-4-E4B-it               |
|   14 | 0.1363 | 50.0% |  84.7 |      10.7 | baseline    | True    | google/gemma-4-E2B-it               |
|   15 | 0.1296 | 50.0% |  89.3 |      10.4 | no_fewshot  | True    | google/gemma-4-E2B-it               |
|   16 | 0.1168 | 53.3% | 100.8 |      16.1 | no_fewshot  | True    | google/gemma-4-E4B-it               |

**Best result**: `Qwen/Qwen3-4B-Instruct-2507` + baseline prompt + routing enabled — **76.7% accuracy, score 0.3661**.

### Analysis

- **Qwen3-4B is the clear winner.** It sweeps the top 3 slots and takes 4 of the top 7. It hits the best accuracy/latency balance on a 5090: ~50 s for 30 items at 8 GB VRAM.
- **Routing helps Qwen but hurts Gemma.** On Qwen3-4B, baseline + routing gains **+6.7 pp accuracy** vs no-routing (70.0% -> 76.7%) and **+16% score**. On Gemma-4-E4B the effect inverts: routing *loses* 3.3 pp (73.3% -> 70.0%). Hypothesis: Gemma's 4096-token context gets crowded by our 10 few-shots + extra snippets, pushing the actual question away from the generation head. A Gemma-specific prompt variant with fewer snippets (or fewer few-shots when routing is on) is the obvious next step.
- **Few-shots are doing most of the work.** `no_fewshot` is the worst variant for every model, by a wide margin on the smaller ones: Qwen-1.5B drops **-16.7 pp** (50.0% -> 33.3%), Qwen3-4B drops **-20 pp** (76.7% -> 56.7%). The demos are more valuable than the system prompt.
- **Score is dominated by `T`, not `VRAM`.** VRAM enters as `VRAM^0.1`, so doubling VRAM only costs ~7% score; doubling `T` halves it. That's why Qwen-1.5B lands in the top 6 despite only 50% accuracy (it's fast + light), and why Gemma gets punished — its 80-100 s runs kill the score even at 70%+ accuracy.
- **Terse prompt is a near-free win when routing is on.** On Qwen3-4B it's ~3% faster but loses 6.7 pp accuracy vs baseline — not worth it. On Gemma-4-E2B it's actually the best variant (60% vs 50% baseline), suggesting Gemma prefers shorter system prompts.
- **Qwen2.5-Coder-7B** is missing from this run (too slow in the session). Adding vLLM with prefix caching would likely bring it in under the wall-time cap and is the most promising single upgrade for pushing past 0.37.

Takeaway: on this hardware and formula, the winning recipe is **Qwen3-4B + rich few-shots + routing**. The next lever is inference throughput (vLLM) more than model size or prompt engineering.

## Next Steps

- **Bigger benchmark**: convert Text2SQL datasets (Spider, WikiSQL) to Text2Polars for a larger and more diverse eval set
- **Prompt iteration**: analyze `failures.csv` to find systematic error patterns and add targeted few-shot examples
- **Model finetuning**: finetune on a synthetic dataset thant can be converted from text2SQL datasets