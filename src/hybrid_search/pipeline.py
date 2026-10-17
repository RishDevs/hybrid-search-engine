import json
import time
from dataclasses import dataclass, field

import numpy as np

from . import data
from .bm25 import BM25
from .config import ARTIFACTS_DIR, ROUTER_FILE, Config, artifact_dir, slug
from .dense import DenseIndex
from .embed import Encoder, cached_doc_vectors
from .fusion import rrf
from .rerank import Reranker
from .router import Router, query_features


@dataclass
class Hit:
    doc_id: str
    score: float
    text: str


@dataclass
class SearchResult:
    hits: list
    reranked: bool
    predicted_gain: float
    timings_ms: dict = field(default_factory=dict)


def bm25_path(dataset, cfg):
    return ARTIFACTS_DIR / dataset / f"bm25-k1_{cfg.k1}-b_{cfg.b}.npz"


def build_artifacts(ds, cfg, encoder=None):
    folder = artifact_dir(ds.name, cfg)
    folder.mkdir(parents=True, exist_ok=True)

    bm_file = bm25_path(ds.name, cfg)
    if bm_file.exists():
        bm = BM25.load(bm_file)
    else:
        bm = BM25(cfg.k1, cfg.b).fit(ds.texts)
        bm.save(bm_file)

    encoder = encoder or Encoder(cfg.dense_model, cfg.query_prefix, cfg.device)
    vectors = cached_doc_vectors(encoder, ds, folder)

    index_file = folder / f"{slug(cfg.index_factory)}.faiss"
    if index_file.exists():
        index = DenseIndex.load(index_file, cfg.index_factory)
    else:
        index = DenseIndex.build(vectors, cfg.index_factory)
        index.save(index_file)

    (folder / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    return bm, index, encoder


class HybridSearcher:
    def __init__(self, cfg, ds, bm25, index, encoder, reranker=None, router=None):
        self.cfg = cfg
        self.ds = ds
        self.bm25 = bm25
        self.index = index
        self.encoder = encoder
        self.reranker = reranker
        self.router = router or Router.always()

    @classmethod
    def open(cls, dataset, cfg=None, with_reranker=True, cache_scores=False):
        cfg = cfg or Config()
        ds = data.load(dataset)
        bm, index, encoder = build_artifacts(ds, cfg)
        reranker = Reranker(cfg.reranker_model, cfg.device, cache=cache_scores) if with_reranker else None
        router_file = ROUTER_FILE
        router = Router.load(router_file) if router_file.exists() else None
        return cls(cfg, ds, bm, index, encoder, reranker, router)

    def _fuse(self, bm_ids, dense_ids, ce_ids=None):
        cfg = self.cfg
        if ce_ids is None:
            return rrf([bm_ids, dense_ids], cfg.rrf_k, [cfg.w_bm25, cfg.w_dense])
        return rrf([ce_ids, bm_ids, dense_ids], cfg.rrf_k, [cfg.w_rerank, cfg.w_bm25, cfg.w_dense])

    def features(self, query, bm, dense, fused_scores):
        return query_features(len(self.bm25.analyzer(query)), *bm, *dense, fused_scores)

    def search(self, query, k=10, rerank="auto"):
        clock = {}
        t0 = time.perf_counter()
        qvec = self.encoder.encode_queries([query])
        t1 = time.perf_counter()
        dense_scores, dense_ids = self.index.search(qvec, self.cfg.depth)
        keep = dense_ids[0] >= 0
        dense = (dense_ids[0][keep], dense_scores[0][keep])
        t2 = time.perf_counter()
        bm = self.bm25.search(query, self.cfg.depth)
        t3 = time.perf_counter()
        ids, scores = self._fuse(bm[0], dense[0])
        feats = self.features(query, bm, dense, scores)
        predicted = float(self.router.predict(feats)[0])
        t4 = time.perf_counter()
        clock.update(encode=t1 - t0, dense=t2 - t1, bm25=t3 - t2, fuse=t4 - t3)

        use_reranker = self.reranker is not None and (
            rerank == "always" or (rerank == "auto" and predicted >= self.router.threshold)
        )
        if use_reranker:
            ce_ids, _ = self.reranker.rerank(query, ids[: self.cfg.rerank_depth], self.ds.texts)
            ids, scores = self._fuse(bm[0], dense[0], ce_ids)
            clock["rerank"] = time.perf_counter() - t4

        hits = [Hit(self.ds.doc_ids[i], float(s), self.ds.texts[i]) for i, s in zip(ids[:k], scores[:k])]
        timings = {name: round(sec * 1000, 3) for name, sec in clock.items()}
        timings["total"] = round((time.perf_counter() - t0) * 1000, 3)
        return SearchResult(hits, use_reranker, predicted, timings)

    def run_batch(self, queries, rerank=True, progress=False):
        qvecs = self.encoder.encode_queries(queries)
        dense_scores, dense_ids = self.index.search(qvecs, self.cfg.depth)
        dense = [(i[i >= 0], s[i >= 0]) for i, s in zip(dense_ids, dense_scores)]
        bm = self.bm25.search_many(queries, self.cfg.depth)
        fused = [self._fuse(b[0], d[0]) for b, d in zip(bm, dense)]
        out = {
            "bm25": bm,
            "dense": dense,
            "hybrid": fused,
            "features": np.array([self.features(q, b, d, f[1]) for q, b, d, f in zip(queries, bm, dense, fused)]),
        }
        if rerank and self.reranker is not None:
            depth = self.cfg.rerank_depth
            ce = self.reranker.rerank_many(queries, [ids[:depth] for ids, _ in fused], self.ds.texts, progress)
            out["hybrid+ce-only"] = [
                (np.concatenate([c[0], f[0][depth:]]), None) for c, f in zip(ce, fused)
            ]
            out["hybrid+rerank"] = [self._fuse(b[0], d[0], c[0]) for b, d, c in zip(bm, dense, ce)]
        return out
