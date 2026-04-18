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
    model=os.getenv("MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
    dtype=os.getenv("DTYPE", "bfloat16"),
    max_model_len=int(os.getenv("MAX_MODEL_LEN", "4096")),
    max_tokens=int(os.getenv("MAX_TOKENS", "256")),
    use_routing=os.getenv("USE_ROUTING", "true") == "true",
    prompt_variant=os.getenv("PROMPT_VARIANT", "baseline"),
)

log.info(f"Loading model: {CFG.model}")
t0 = time.perf_counter()
engine = PolarsInferenceEngine(CFG)
log.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    schema = {}
    for name, cols in payload.tables.items():
        if isinstance(cols, dict):
            schema[name] = {c: str(t) for c, t in cols.items()}
        else:
            schema[name] = cols

    result = engine.generate_one(
        question=payload.message,
        schema=schema,
    )

    return ChatResponse(response=result.code)
