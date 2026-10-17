import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

HOME = Path(os.environ.get("HSE_HOME", Path.cwd()))
DATA_DIR = HOME / "data"
ARTIFACTS_DIR = HOME / "artifacts"
RESULTS_DIR = HOME / "results"
CACHE_DIR = HOME / "cache"
ROUTER_FILE = RESULTS_DIR / "router.json"

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Config:
    dense_model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = BGE_PREFIX
    reranker_model: str = "Alibaba-NLP/gte-reranker-modernbert-base"
    k1: float = 0.9
    b: float = 0.4
    depth: int = 100
    rrf_k: int = 60
    w_bm25: float = 0.3
    w_dense: float = 1.0
    w_rerank: float = 4.0
    rerank_depth: int = 30
    index_factory: str = "Flat"
    device: str | None = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, raw):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


def artifact_dir(dataset, cfg):
    return ARTIFACTS_DIR / dataset / slug(cfg.dense_model)
