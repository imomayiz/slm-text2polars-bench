# Sweep runbook

## Day-before setup (do tonight)

### 1. Install

```bash
pip install vllm polars torch
```

### 2. Pre-download model weights

Saves 15–30 min on hackathon day. Run this overnight if your home internet is slow:

```bash
export HF_HOME=~/hf_cache    # somewhere with plenty of disk

# 24GB profile
for M in \
    Qwen/Qwen2.5-Coder-1.5B-Instruct \
    Qwen/Qwen2.5-Coder-3B-Instruct \
    Qwen/Qwen2.5-Coder-7B-Instruct \
    microsoft/Phi-3.5-mini-instruct \
    HuggingFaceTB/SmolLM3-3B
do
    huggingface-cli download "$M"
done

# 16GB profile: swap 7B for 7B-AWQ
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct-AWQ
```

### 3. Smoke-test vLLM

```bash
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='Qwen/Qwen2.5-Coder-1.5B-Instruct', max_model_len=2048)
out = llm.chat(
    [{'role': 'user', 'content': 'Write: result = df.filter(pl.col(\"x\") > 5)'}],
    SamplingParams(temperature=0, max_tokens=50),
)
print(out[0].outputs[0].text)
"
```

If this works, vLLM is good. If not, fix now.

### 4. Smoke-test the bench harness

```bash
python make_starter_bench.py    # writes bench.json
python smoke_test.py            # should print 10/10 using reference code
```

## Hackathon day — the sweep

### Phase 1: baseline sweep (90 min budget)

```bash
# 24GB box
python sweep.py --bench bench.json --profile 24gb

# 16GB box
python sweep.py --bench bench.json --profile 16gb
```

This runs each model with default prompt + greedy decoding. Outputs
`sweep_results/leaderboard.csv` sorted by score, plus per-category accuracy.

Look at:
1. **Which model wins on score?** That's your primary candidate.
2. **Which model wins on accuracy alone?** That's your "if I can make it
   faster, it wins" candidate.
3. **Category breakdown:** is any model weak on a specific category (e.g.
   joins)? That's a prompt-engineering fix, not a model switch.

### Phase 2: prompt tuning (60 min budget)

Lock the best model. Now vary prompt components on that single model:

```bash
# Edit generator.py FEW_SHOT list, then:
python run_bench.py --bench bench.json --model <winner> --out v1_4shot.json
# Reduce to 2 few-shots:
python run_bench.py --bench bench.json --model <winner> --out v2_2shot.json
# Add a Polars cheat-sheet snippet to system prompt:
python run_bench.py --bench bench.json --model <winner> --out v3_cheatsheet.json
```

Compare JSON summaries. One change at a time.

### Phase 3: decoding tricks (30 min)

- Try `max_tokens=128` vs 256 vs 512 — lower if responses are short
- Tighten `stop` tokens
- Try `--enable-prefix-caching` in vLLM (hardcode in `PolarsGenerator.__init__`)

### Phase 4: stretch experiments (if time)

- Self-consistency: `n=3, temp=0.2`, pick the output that runs without error
- Repair loop: already built — toggle `--repair` in `run_bench.py`
- AWQ quantization of the winning model (may hurt accuracy — measure)

## Reading the leaderboard

```
model                                           N/tot   T(s)   VRAM    RAM    score
Qwen2.5-Coder-1.5B-Instruct                     8/10     9.2   3.10   3.82   0.8052
Qwen2.5-Coder-3B-Instruct                       9/10    18.5   6.80   4.11   0.4552
Qwen2.5-Coder-7B-Instruct                       9/10    42.1  15.20   4.20   0.1988
Phi-3.5-mini-instruct                           7/10    21.0   7.50   4.05   0.3154
SmolLM3-3B                                      7/10    16.4   6.20   3.95   0.3950
```

Interpretation:
- **1.5B wins on score** even at 80% accuracy because T is 5× lower than 7B.
- **3B is a close second** and might win after prompt tuning (if you can push
  9/10 → 10/10 without increasing T much).
- **7B is a trap** here: highest accuracy but T penalty crushes score.
- **Phi-3.5 underperforms** on code gen vs Qwen-Coder — expected, Qwen-Coder
  is code-specialized.

Your decision tree:
- If 1.5B and 3B both hit ≥80% accuracy → tune the 1.5B. Speed wins ties.
- If 1.5B <60% but 3B ≥80% → 3B is the winner. Below 60% accuracy, prompt
  tuning won't save you.
- If nothing breaks 70% → the questions are harder than you think. Revisit
  your gold set — are questions ambiguous?

## Red flags during the sweep

- **VRAM > expected**: `--gpu-memory-utilization` defaults to 0.9. Lower it
  to 0.7 for a more honest footprint.
- **One model's T >> others**: probably no batching. vLLM batches inside
  `llm.chat()` if you pass all prompts at once — check `generator.py` does this.
- **All models score ~same accuracy**: either your bench is too easy (add
  hard items) or there's a prompt-template bug affecting all models equally.

## Submission time (16:30-ish)

Whichever model + config wins your sweep → wire it into the organizer's
sample repo. Their API interface will dictate the wrapping; your job is
just to swap in:

```python
MODEL = "Qwen/Qwen2.5-Coder-<winner>-Instruct"
PROMPT_TEMPLATE = ...  # from your tuned generator.py
SAMPLING = SamplingParams(temperature=0, max_tokens=256, stop=[...])
```

Submit one thing that works before 16:45. Then try variants in the remaining time.
