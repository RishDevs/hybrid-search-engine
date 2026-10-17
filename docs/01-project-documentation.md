# Hybrid Search Engine — Project Documentation

**Status:** Built and evaluated (October 2026). All components below exist in `src/hybrid_search/` and run with the commands in §8. Measured results live in `results/RESULTS.md`.

**Cost constraint:** Everything runs at zero cost: open-source libraries, open-weight models, public datasets, local CPU/Apple-GPU compute, free GitHub Actions on a public repo.

Related documents: [02 — significance](02-significance.md) · [03 — code walkthrough](03-code-walkthrough.md) · [04 — defense](04-defense.md)

---

## Contents

1. [What the project is](#1-what-the-project-is)
2. [Architecture](#2-architecture)
3. [Components](#3-components)
4. [Evaluation protocol](#4-evaluation-protocol)
5. [FAISS benchmark](#5-faiss-benchmark)
6. [CI quality gate](#6-ci-quality-gate)
7. [Tech stack and zero-cost choices](#7-tech-stack-and-zero-cost-choices)
8. [Setup and running](#8-setup-and-running)
9. [Decisions made](#9-decisions-made)
10. [Known limitations](#10-known-limitations)

---

## 1. What the project is

A search engine that combines BM25 keyword retrieval and dense embedding retrieval, fuses them with weighted reciprocal rank fusion, lets a cross-encoder vote on the top candidates when a learned router predicts it will help, and measures the quality / speed / memory tradeoffs.

| Deliverable | Where |
|---|---|
| Hybrid retrieval pipeline | `pipeline.py`, `bm25.py`, `dense.py`, `fusion.py`, `rerank.py`, `router.py` |
| Evaluation on 3 BEIR datasets with dev-only tuning | `experiments.evaluate`, `experiments.tune`, `metrics.py` |
| FAISS benchmark (12 configs, 200K docs) | `experiments.bench` |
| Reranker selection and weight tuning | `experiments.tune` |
| Router training and routing experiment | `router.py`, `experiments.routing` |
| CI quality gate | `gate.py`, `.github/workflows/ci.yml`, `results/baseline.json` |
| HTTP API | `api.py` (FastAPI) |
| CLI | `cli.py` (`hse ...`) |

---

## 2. Architecture

```
query ─┬─▶ BM25 (top 100) ──────────────────┐
       │                                    ├─▶ weighted RRF (w_bm25 0.3, w_dense 1.0) ─▶ 8 query features ─▶ router
       └─▶ bge-small ─▶ FAISS Flat (top 100)┘                                                                │
                                         predicted gain < threshold ◀───────────────────────────────────────┤
                                            └─▶ return top-k                                                 │
                                         predicted gain ≥ threshold                                          │
                                            └─▶ gte cross-encoder on top 30 ◀────────────────────────────────┘
                                                 └─▶ RRF(ce 4.0, bm25 0.3, dense 1.0) ─▶ top-k
```

Indexing (offline): corpus → BM25 sparse weight matrix (`.npz`) and corpus → bge-small → float32 vectors (`.npy`, memory-mapped) → FAISS index (`.faiss`). All three are cached under `artifacts/<dataset>/`. Reranker scores used by experiments are cached in `cache/ce/<model>.sqlite`. The trained router is `results/router.json`.

---

## 3. Components

| Component | Implementation | Key parameters |
|---|---|---|
| Data | BEIR zips from UKP; `Dataset(doc_ids, texts, queries, qrels)`; title prepended to body | splits: test (eval), train/dev (tuning) |
| BM25 | Own implementation: precomputed BM25 weights in a CSC sparse matrix; Lucene stopwords + Snowball stemmer; Lucene IDF | k1 = 0.9, b = 0.4 |
| Dense encoder | `BAAI/bge-small-en-v1.5` via sentence-transformers, L2-normalized, query instruction prefix | 384 dims, max 512 tokens |
| Vector index | FAISS, inner product; any `index_factory` string; binary (`B…`) indexes supported; optional rescoring | serving default `Flat` |
| Fusion | Weighted reciprocal rank fusion | k = 60, w_bm25 = 0.3, w_dense = 1.0 |
| Reranker | `Alibaba-NLP/gte-reranker-modernbert-base` (150M, Apache-2.0), used as a third RRF vote; SQLite score cache for experiments | depth 30, w_rerank = 4.0 |
| Router | Ridge regression on 8 query features predicting nDCG gain; threshold set on cross-validated dev predictions to recover 80% of the gain | one global router, `results/router.json` |
| Metrics | nDCG@10 (linear gain, trec_eval style), recall@100, MRR@10, paired t-test; query-ID self-matches dropped (BEIR convention) | — |

---

## 4. Evaluation protocol

| Item | Choice |
|---|---|
| Datasets | SciFact (5,183 docs / 300 test queries), NFCorpus (3,633 / 323), FiQA-2018 (57,638 / 648) |
| Tuning data | SciFact train (809 queries), NFCorpus dev (324), FiQA dev (500) |
| Tuning | `hse tune`: rerankers {MiniLM-L6, bge-reranker-base, gte-reranker-modernbert} × depth {10, 20, 30} × w_bm25 {0.1, 0.2, 0.3, 0.5} × w_rerank {0.5, 1, 2, 3, 4, 6, 10}; best mean dev nDCG@10 → gte, 30, 0.3, 4.0 |
| Systems compared | bm25, dense, hybrid, hybrid+ce-only (cross-encoder replaces the order), hybrid+rerank (cross-encoder votes) |
| Significance | Paired t-test on per-query nDCG@10 |
| Router | Trained and calibrated on the dev/train queries only; applied frozen to test; compared with always, never and random routing at the same share |
| Reproducibility | Every result JSON stores config, library versions, CPU and git commit; `uv.lock` pins dependencies |

---

## 5. FAISS benchmark

- **Corpus:** Quora subsampled to 200,000 docs while keeping all 15,675 docs judged in the test qrels; 10,000 test queries; d = 384.
- **Configurations (12):** Flat, SQfp16, SQ8, PQ48, OPQ48+PQ48, HNSW32+Flat, HNSW32+SQ8, IVF1024+Flat, IVF4096+Flat, IVF1024+SQ8, IVF1024+PQ48, binary Flat.
- **Per setting:** recall@10 vs exact, nDCG@10, mean single-query latency (1 thread, 1,000 queries, 20 warm-up), build time, real serialized size.
- **Operating points:** fastest with recall@10 ≥ 0.95, and fastest within 0.5 nDCG points of exact.
- **Rescoring** (PQ48, OPQ, binary): top-100 from the compressed index re-scored with float32 vectors from a memory-mapped file.

---

## 6. CI quality gate

- Workflow `.github/workflows/ci.yml`, on every pull request and every push to `main`.
- Job 1 runs the unit tests (37 tests, no models or downloads).
- Job 2 runs `hse gate --dataset scifact --tolerance 0.5 --device cpu`, which re-evaluates SciFact and compares dense, hybrid and hybrid+rerank (nDCG@10 and recall@100) with `results/baseline.json`.
- A drop of more than 0.5 points exits with code 1, which fails the check. The comparison table is written to the job summary.
- Two caches: model weights + reranker scores (key: `config.py`, with fallback to the latest), and data + indexes (key: `uv.lock` plus every source file that affects stored artifacts). The cold first run takes longest because the reranker runs on CPU.
- To update the baseline, run `make baseline` and commit `results/baseline.json` in the same PR.
- **Repository setting to finish (manual):** in GitHub → Settings → Branches, add a protection rule on `main` that requires the `quality-gate` check to pass before merging.

---

## 7. Tech stack and zero-cost choices

| Layer | Choice | Free-alternative tradeoff |
|---|---|---|
| Language / env | Python 3.11, uv + `uv.lock` | — |
| Keyword search | Own BM25 on NumPy/SciPy + PyStemmer | Instead of Elasticsearch/OpenSearch: no server, but no incremental indexing or sharding |
| Embeddings | bge-small-en-v1.5 (MIT) | Instead of paid embedding APIs: weaker than the best APIs, no per-call cost, data stays local |
| Vector search | faiss-cpu (MIT) | Instead of a managed vector DB: no persistence layer or filtering, full control of index type |
| Reranker | gte-reranker-modernbert-base (Apache-2.0) | Instead of paid rerank APIs: ~1.6 s per reranked query on CPU, which the router cuts to a 678 ms mean |
| API | FastAPI + uvicorn | — |
| CI | GitHub Actions (public repo, unlimited minutes) | CPU-only runners, so the gate uses one small dataset |
| Compute | Local; MPS on Apple Silicon, CUDA if present, otherwise CPU | — |

---

## 8. Setup and running

Requirements: Python 3.10–3.12, uv, ~3 GB disk, 8 GB RAM (16 GB for the 200K benchmark). No GPU required.

```bash
git clone https://github.com/RishDevs/hybrid-search-engine.git
```

```bash
cd hybrid-search-engine
```

```bash
make setup
```

| Command | What it does |
|---|---|
| `uv run hse search "query" --dataset scifact` | one query, with per-stage timings |
| `uv run hse serve --dataset scifact` | API at `http://127.0.0.1:8000` (`/search`, `/health`, `/docs`) |
| `make test` | unit tests |
| `make download` | SciFact, NFCorpus, FiQA, Quora |
| `make tune` | fusion weight grid search on train/dev |
| `make eval` | test-set evaluation of all systems |
| `make bench` | FAISS benchmark on quora-200k |
| `make route` | router training on dev + test evaluation + CPU latency |
| `make report` | writes `results/RESULTS.md` |
| `make gate` / `make baseline` | run the CI check locally / regenerate the baseline |

**macOS:** `make setup` runs `hse fix-openmp`. FAISS and PyTorch ship separate `libomp` copies, and loading both segfaults inside FAISS. The command points FAISS at PyTorch's copy. `dense.py` refuses to import FAISS with a clear message if the fix is missing. Re-run it after recreating `.venv`.

---

## 9. Decisions made

| Was open | Decision | Basis |
|---|---|---|
| BM25 library | Own implementation | Matches Anserini BEIR numbers within 0.2 points; fully explainable |
| Embedding model | bge-small-en-v1.5 with query prefix | Prefix kept after dev-split comparison |
| Quality datasets | SciFact, NFCorpus, FiQA | Small enough to run on CPU; mix of domains |
| Speed-benchmark corpus | Quora 200K, qrels-preserving sample | Real embeddings and labels; fits in RAM |
| Fusion | Weighted RRF | Equal-weight RRF fell below dense on FiQA |
| Reranker | gte-reranker-modernbert-base as a fusion vote | Best of four on dev (54.87 vs ≤ 51.24 for the others; dense 49.49) |
| Router | Learned ridge regression on 8 features | A single-margin router was no better than random |
| Evaluation library | Own metrics (trec_eval-style nDCG) | No extra dependency; unit-tested |
| CI tolerance | Absolute 0.5 points | Deterministic locally; catches the depth-cut regression |
| Interface | CLI + FastAPI | No hosted demo yet |
| Compression measurement | Real serialized bytes; raw and rescored nDCG both reported | Honest size and quality claims |
| Rescoring | Reported as a separate row | RAM saving is 32×; disk still holds float32 vectors |

---

## 10. Known limitations

- The cross-encoder takes about 1.6 s per reranked query on CPU; the router brings the mean to 678 ms, but reranked queries still pay the full cost.
- On test, the router recovered 69% of the reranker's gain against an 80% target set on dev.
- The reranker was selected on dev, but after the first version's test results were known (see the defense document, question 1).
- BM25 rebuilds fully when documents change; all indexes are in memory on one machine.
- English only; documents are truncated at 512 tokens by both neural models.
- The CI gate evaluates one dataset.
