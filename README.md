# Hybrid Search Engine

Keyword search misses meaning; vector search misses exact terms like IDs and names. This project runs BM25 and dense embedding search side by side, fuses them with weighted reciprocal rank fusion, lets a cross-encoder vote on the top candidates when a learned router predicts it will help, and measures the tradeoff between search quality, speed, and memory.

## Status

**Complete and reproducible.** Every component below is implemented, tested (37 unit tests) and covered by CI, and all experiments have been run end to end. Measured results, with the exact commands that produce them, are in [`results/RESULTS.md`](results/RESULTS.md).

| Component | State |
|---|---|
| Hybrid retrieval (BM25 + dense + weighted RRF) | Done |
| Cross-encoder reranking + learned router | Done |
| Evaluation on SciFact, NFCorpus, FiQA (dev-only tuning) | Done |
| FAISS benchmark (12 configs, 200K docs) | Done |
| CI quality gate | Done; runs on every PR |
| FastAPI service | Done; runs locally (not hosted) |

**Possible next steps:** faster CPU reranking (ONNX/int8), incremental indexing, and evaluation on a held-out fourth dataset.

## What's in here

- **Hybrid retrieval** — a from-scratch BM25 (sparse matrix, Lucene-style IDF) and `bge-small-en-v1.5` embeddings in FAISS run on every query; their rankings are combined with weighted RRF.
- **Cross-encoder as a voter** — `gte-reranker-modernbert-base` scores the top 30 fused candidates and its ranking is fused back in, instead of replacing the first-stage order. The reranker was picked from four candidates on dev splits.
- **Learned rerank router** — a ridge-regression model over eight cheap query features predicts how much reranking will help each query; only queries above a threshold pay for the cross-encoder.
- **Reranker score cache** — cross-encoder scores are cached in SQLite by (model, query, passage), so experiments and CI only score new pairs.
- **FAISS benchmark** — 12 index configurations (flat, scalar/product/binary quantization, HNSW, IVF) on a 200K-document corpus: recall vs. exact search, single-query latency, memory, build time, and nDCG.
- **CI quality gate** — GitHub Actions evaluates retrieval quality on SciFact for every pull request and fails if nDCG@10 or recall@100 drops more than 0.5 points below `results/baseline.json`.
- **FastAPI service** — `/search` and `/health` endpoints over a built index.

## How it works

```
query ─┬─▶ BM25 (top 100) ─────────────────┐
       │                                   ├─▶ weighted RRF ─▶ router: predicted gain ≥ threshold? ─┬─ no ─▶ top-k
       └─▶ bge-small ─▶ FAISS (top 100) ───┘                                                      │
                                                                                                  └─ yes ─▶ cross-encoder on top 30
                                                                                                             └─▶ RRF(cross-encoder, BM25, dense) ─▶ top-k
```

The reranker, rerank depth and fusion weights are grid-searched on the BEIR **train/dev** splits (`hse tune`), and the router is trained on the same dev queries (`hse route`). Test queries are only used for the final evaluation.

## Getting started

Requirements: Python 3.10–3.12, [uv](https://docs.astral.sh/uv/), ~3 GB of disk for datasets, models and indexes, 8 GB RAM (16 GB for the 200K benchmark). No GPU needed; Apple Silicon (MPS) and CUDA are used automatically when present.

```bash
git clone https://github.com/RishDevs/hybrid-search-engine.git
```

```bash
cd hybrid-search-engine
```

```bash
make setup
```

`make setup` installs the locked dependencies and, on macOS, runs `hse fix-openmp`. FAISS and PyTorch each bundle their own OpenMP runtime on macOS, and loading both crashes FAISS mid-search; the fix points FAISS at PyTorch's copy. It's a no-op on Linux.

Without `make`:

```bash
uv sync --extra dev --python 3.11 && uv run hse fix-openmp
```

### Try a query

```bash
uv run hse search "vitamin D deficiency and bone fractures" --dataset scifact
```

The first run downloads SciFact (~3 MB) and both models (~700 MB), embeds the corpus, and caches everything under `data/` and `artifacts/`. Later runs start in seconds. On CPU, a query that goes to the reranker takes about 1.6 s; one that skips it takes under 10 ms.

### Run the API

```bash
uv run hse serve --dataset scifact --port 8000
```

```bash
curl "http://127.0.0.1:8000/search?q=statins+and+muscle+pain&k=5&rerank=auto"
```

`rerank` is `auto` (use the router), `always`, or `never`. Interactive docs are at `http://127.0.0.1:8000/docs`.

### Reproduce the experiments

| step | command | what it does |
|---|---|---|
| 1 | `make download` | fetch SciFact, NFCorpus, FiQA and Quora from BEIR |
| 2 | `make tune` | pick reranker, depth and fusion weights on train/dev splits → `results/tune.json` |
| 3 | `make eval` | BM25 / dense / hybrid / rerank on the three test sets → `results/eval/` |
| 4 | `make bench` | 12 FAISS configs on a 200K-doc Quora sample → `results/bench/` |
| 5 | `make route` | train the router on dev, measure it on test (CPU latency) → `results/router.json`, `results/routing/` |
| 6 | `make report` | collect everything into `results/RESULTS.md` |

On an M4 Pro, `make tune` takes about 45 minutes the first time (three rerankers on ~1,600 dev queries; cached after that), `make eval` about 20 minutes, `make bench` about 20 and `make route` about 30.

### Tests and the quality gate

```bash
make test
```

```bash
make gate
```

`make gate` runs the SciFact evaluation on CPU and compares it with `results/baseline.json`. After an intentional change, regenerate the baseline with `make baseline` and commit it with the change so the reviewer sees the new numbers.

## Project layout

```
src/hybrid_search/
  bm25.py         BM25 over a scipy sparse matrix
  embed.py        sentence-transformers wrapper + on-disk vector cache
  dense.py        FAISS index wrapper (float and binary), rescoring
  fusion.py       weighted reciprocal rank fusion, top-1 margin
  rerank.py       cross-encoder scoring + SQLite score cache
  router.py       query features + ridge-regression rerank router
  pipeline.py     index building and the query path
  metrics.py      nDCG, recall, MRR, paired t-test
  experiments.py  tune / eval / bench / routing runs
  report.py       results/RESULTS.md
  gate.py         CI regression check
  api.py          FastAPI app
  cli.py          `hse` command
  openmp.py       macOS libomp conflict check and fix
tests/            unit tests
docs/             design and walkthrough documents
```

## Stack

All free and open source: Python, NumPy/SciPy, PyStemmer, FAISS, PyTorch, sentence-transformers, FastAPI, GitHub Actions. Data comes from the [BEIR](https://github.com/beir-cellar/beir) benchmark; check each dataset's license before commercial use (several are non-commercial).

## License

[MIT](LICENSE)
