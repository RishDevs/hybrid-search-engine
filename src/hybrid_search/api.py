import math
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .config import Config


class HitOut(BaseModel):
    doc_id: str
    score: float
    snippet: str


class SearchOut(BaseModel):
    query: str
    reranked: bool
    predicted_gain: float
    timings_ms: dict[str, float]
    hits: list[HitOut]


def create_app(searcher=None, dataset=None):
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        if searcher is not None:
            state["searcher"] = searcher
        else:
            from .pipeline import HybridSearcher

            name = dataset or os.environ.get("HSE_DATASET", "scifact")
            state["searcher"] = HybridSearcher.open(name, Config(device=os.environ.get("HSE_DEVICE")))
        yield
        state.clear()

    app = FastAPI(title="Hybrid Search Engine", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        s = state.get("searcher")
        if s is None:
            raise HTTPException(503, "index not loaded")
        return {
            "status": "ok",
            "dataset": s.ds.name,
            "docs": len(s.ds),
            "dense_model": s.cfg.dense_model,
            "reranker": s.reranker.name if s.reranker else None,
            "router_threshold": s.router.threshold if math.isfinite(s.router.threshold) else None,
        }

    @app.get("/search", response_model=SearchOut)
    def search(
        q: str = Query(..., min_length=1, max_length=1000),
        k: int = Query(10, ge=1, le=100),
        rerank: Literal["auto", "always", "never"] = "auto",
    ):
        s = state["searcher"]
        res = s.search(q, k=k, rerank=rerank)
        return SearchOut(
            query=q,
            reranked=res.reranked,
            predicted_gain=res.predicted_gain,
            timings_ms=res.timings_ms,
            hits=[HitOut(doc_id=h.doc_id, score=h.score, snippet=h.text[:300]) for h in res.hits],
        )

    return app
