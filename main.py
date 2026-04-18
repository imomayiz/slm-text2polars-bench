"""
FastAPI app
Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference import PolarsInferenceEngine, InferenceConfig

log = logging.getLogger("main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Read config from env so the same image can serve different models 
CFG = InferenceConfig(
    model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    dtype=os.getenv("DTYPE", "bfloat16"),
    max_model_len=int(os.getenv("MAX_MODEL_LEN", "2048")),
    gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL", "0.85")),
    max_tokens=int(os.getenv("MAX_TOKENS", "256")),
)


engine: PolarsInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; release on shutdown."""
    global engine
    log.info(f"Loading model: {CFG.model}")
    t0 = time.perf_counter()
    engine = PolarsInferenceEngine(CFG)
    log.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")
    yield
    log.info("Shutting down")
    engine = None


app = FastAPI(title="muon-submission", lifespan=lifespan)


# ---------- Schemas (adjust to organizer's spec!) ----------

# Likely request shape — two common variants:
#   (a) question + schema (what our local harness uses)
#   (b) a full pre-built prompt string (simpler, more common in benchmark runners)
# We accept BOTH — if `prompt` is given, it takes priority.

class SchemaSpec(BaseModel):
    """Per-dataframe column map, e.g. {'df': {'col1': 'Int64', ...}}"""


class GenerateRequest(BaseModel):
    # Variant A: structured
    question: str | None = Field(None, description="Natural language question")
    schema_: dict[str, dict[str, str]] | None = Field(
        None,
        alias="schema",
        description="{df_name: {col: dtype, ...}, ...}",
    )

    # Variant B: raw prompt (if the runner pre-builds it)
    prompt: str | None = Field(None, description="Pre-built prompt, overrides question+schema")

    # Optional overrides
    max_tokens: int | None = None
    temperature: float | None = None

    model_config = {"populate_by_name": True}  # allow both 'schema' and 'schema_'


class GenerateResponse(BaseModel):
    code: str
    generation_time_seconds: float
    tokens_generated: int


# ---------- Routes ----------

@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe. Return 200 only once the model has loaded."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "model": CFG.model}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if req.prompt is None and (req.question is None or req.schema_ is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either `prompt` OR both `question` and `schema`.",
        )

    t0 = time.perf_counter()
    try:
        result = engine.generate_one(
            question=req.question,
            schema=req.schema_,
            prompt_override=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    except Exception as e:
        log.exception("generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        code=result.code,
        generation_time_seconds=time.perf_counter() - t0,
        tokens_generated=result.tokens,
    )


# ---------- Optional: batched endpoint for higher throughput ----------
# if supported

class BatchGenerateRequest(BaseModel):
    items: list[GenerateRequest]


class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]


@app.post("/generate_batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest) -> BatchGenerateResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    results = engine.generate_batch([
        {
            "question": it.question,
            "schema": it.schema_,
            "prompt_override": it.prompt,
            "max_tokens": it.max_tokens,
            "temperature": it.temperature,
        }
        for it in req.items
    ])
    elapsed = time.perf_counter() - t0
    per_item = elapsed / max(len(results), 1)

    return BatchGenerateResponse(results=[
        GenerateResponse(
            code=r.code,
            generation_time_seconds=per_item,
            tokens_generated=r.tokens,
        )
        for r in results
    ])
