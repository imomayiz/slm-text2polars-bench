"""
FastAPI server 
Endpoints:
  GET  /      -> {"status": "ok"}
  POST /chat  -> {"response": "<polars code>"}

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from inference import PolarsInferenceEngine, InferenceConfig

log = logging.getLogger("main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

CFG = InferenceConfig(
    model=os.getenv("MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"),
    dtype=os.getenv("DTYPE", "bfloat16"),
    max_model_len=int(os.getenv("MAX_MODEL_LEN", "2048")),
    gpu_memory_utilization=float(os.getenv("GPU_MEM_UTIL", "0.85")),
    max_tokens=int(os.getenv("MAX_TOKENS", "256")),
)

# Load model at module level (matches organizer's pattern)
log.info(f"Loading model: {CFG.model}")
t0 = time.perf_counter()
engine = PolarsInferenceEngine(CFG)
log.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")

app = FastAPI()


# ---------- Schemas ----------

class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


# ---------- Routes ----------

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    # tables is a dict like {"df": {"col1": "Int64", ...}, ...}
    schema = {}
    for name, cols in payload.tables.items():
        if isinstance(cols, dict):
            schema[name] = {c: str(t) for c, t in cols.items()}
        else:
            # Fallback: if tables has unexpected shape, pass as-is
            schema[name] = cols

    result = engine.generate_one(
        question=payload.message,
        schema=schema,
    )

    return ChatResponse(response=result.code)
