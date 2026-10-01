# Hybrid Search Engine — Project Documentation

**Status:** Design stage. No code exists yet (repository empty as of 2026-10-01). Everything below describes the system as it will be built. Items marked **OPEN** are decisions for the project owner; nothing marked OPEN has been decided.

**Cost constraint:** Every component runs at zero cost — open-source libraries, open-weight models, public datasets, free-tier compute and CI. Where a paid option would be the usual choice, the free substitute and its tradeoff are noted.

---

## Contents

1. [What the project is](#1-what-the-project-is)
2. [Targets and how they are measured](#2-targets-and-how-they-are-measured)
3. [System architecture](#3-system-architecture)
4. [Components](#4-components)
5. [FAISS index benchmark](#5-faiss-index-benchmark)
6. [Evaluation methodology](#6-evaluation-methodology)
7. [CI quality gate](#7-ci-quality-gate)
8. [Tech stack and zero-cost choices](#8-tech-stack-and-zero-cost-choices)
9. [Repository layout](#9-repository-layout)
10. [Setup and running](#10-setup-and-running)
11. [Open decisions](#11-open-decisions)
12. [Build plan](#12-build-plan)
13. [Known risks and limitations](#13-known-risks-and-limitations)
14. [Glossary](#14-glossary)

---

## 1. What the project is

A search engine that combines two retrieval methods and measures the tradeoffs between quality, speed and memory.

| Problem | Cause | This project's answer |
|---|---|---|
| Keyword search misses meaning | BM25 only matches exact terms; "car" does not match "automobile" | Add dense (embedding) retrieval |
| Vector search misses exact terms | Embeddings blur rare tokens such as IDs, codes and names | Keep BM25 alongside it |
| Combining both is not free | Two retrievers + a reranker cost latency and memory | Benchmark index types and compression to find the best tradeoff |
| Quality silently regresses | Changes to models, parameters or code shift results | CI check blocks any change that lowers quality |

**Deliverables**

1. A hybrid retrieval pipeline: BM25 and dense retrieval run on each query, their result lists are fused, and the top candidates are reranked by a cross-encoder.
2. A FAISS benchmark comparing index types on recall, latency, memory and build time, showing which index is best under which constraint.
3. A CI quality gate that fails any pull request that lowers retrieval quality beyond a set tolerance.
4. Reproducible results: configs, pinned dependencies, result files and plots.

---

## 2. Targets and how they are measured

These are goals, not guaranteed outcomes. Results depend on the dataset and models; they will be reported as measured, including where a target is missed.

| # | Target | Exact definition | Comparison |
|---|---|---|---|
| T1 | ~+4 nDCG@10 points over vector-only search (larger over BM25) | nDCG@10 × 100 on the BEIR test split, averaged per dataset | Full pipeline (hybrid + rerank) vs dense-only and vs BM25-only, same dataset, same queries |
| T2 | 32× smaller index, < 1 point quality loss | Bytes per vector of the compressed index vs float32 flat; nDCG@10 drop of dense retrieval | e.g. 384-dim float32 = 1,536 B/vector → 48 B/vector |
| T3 | ~20× faster than exact search at 95% recall | Single-query latency of an approximate index at recall@10 ≥ 0.95, where recall is measured against exact (flat) search results | Approximate index vs `IndexFlatIP` on the same vectors, same hardware, same thread count |

Notes on each target:

- **T1** — gains vary by dataset. On some BEIR datasets BM25 beats dense retrieval; on others the reverse. The target is evaluated on at least two datasets so a result on one dataset is not presented as general.
- **T2** — two standard routes reach exactly 32×: product quantization with 48 one-byte codes (PQ48×8) or binary quantization (1 bit per dimension). Reaching < 1 point loss usually needs **rescoring**: retrieve extra candidates with the compressed index, then re-score them with full-precision vectors kept on disk. If rescoring is used, the result will state that RAM shrinks 32× but disk does not. IVF-based indexes also store an 8-byte ID per vector, so IVF-PQ48 is 56 B/vector (~27×), not 32×; reported sizes will include this.
- **T3** — speedups only appear at scale. On a 5,000-document corpus exact search already takes well under a millisecond, so T3 is measured on a corpus of ≥ ~500k vectors (see [OPEN-4](#11-open-decisions)).

---

## 3. System architecture

### Query path (online)

```
                  ┌──────────────┐   top-k₁ (e.g. 100)
             ┌──▶ │ BM25 index   │ ────────────────┐
             │    └──────────────┘                 ▼
  query ─────┤                               ┌───────────┐  top-N (e.g. 50–100)  ┌───────────────┐  top-10
             │    ┌──────────────┐           │  Fusion   │ ────────────────────▶ │ Cross-encoder │ ───────▶ results
             └──▶ │ Encoder →    │ ────────▶ │ (RRF or   │                       │ reranker      │
                  │ FAISS index  │  top-k₂   │ weighted) │                       └───────────────┘
                  └──────────────┘           └───────────┘
```

1. The query goes to BM25 and to the dense retriever (encoder → FAISS search). The two can run concurrently.
2. Each returns a ranked list of document IDs with scores.
3. Fusion merges the two lists into one ranking.
4. The cross-encoder scores each (query, document) pair among the top-N fused candidates and re-sorts them.
5. The top 10 are returned.

### Indexing path (offline)

```
corpus ──▶ tokenize ──▶ BM25 index (saved to disk)
   │
   └────▶ encoder (batch) ──▶ float32 embeddings (.npy) ──▶ FAISS index (saved to disk)
```

Embeddings are computed once and cached as `.npy` files keyed by (model name, dataset, hash of the encoding code). All FAISS variants are built from the same cached embeddings, so index comparisons are not confounded by re-encoding.

---

## 4. Components

### 4.1 Data

Public BEIR datasets, which ship with queries and human relevance labels (qrels) — required to compute nDCG@10.

| Dataset | Documents | Test queries | Role |
|---|---|---|---|
| SciFact | 5,183 | 300 | CI gate (small, fast), development |
| NFCorpus | 3,633 | 323 | Second small quality dataset |
| FiQA-2018 | 57,638 | 648 | Mid-size quality dataset |
| ArguAna | 8,674 | 1,406 | Quality dataset where BM25 is competitive |
| Quora | 522,931 | 10,000 | Candidate for the T3 speed benchmark |

Source: Hugging Face Hub (`BeIR/<name>` + `BeIR/<name>-qrels`) or the `beir` package. Which datasets are used is **OPEN-3**. Several BEIR datasets carry non-commercial licenses (e.g. SciFact is CC BY-NC); this is fine for a portfolio/research project but must be rechecked for any commercial use.

### 4.2 BM25 retriever

Scores documents by term frequency, inverse document frequency and length normalization (parameters `k1`, `b`). Library choice is **OPEN-1**:

| Option | Strengths | Weaknesses |
|---|---|---|
| `bm25s` | Pure Python + NumPy/SciPy, fast, no Java, simple install | Fewer tokenizer/analyzer options than Lucene |
| Pyserini (Lucene) | Reference-grade BEIR BM25 numbers; strong analyzers | Requires a Java JDK; heavier install; more complex |
| `rank_bm25` | Simplest code, easy to read | Slow at scale (scores every document per query in Python) |
| `tantivy` (Python bindings) | Rust engine, fast, persistent index | Less common in IR research; scoring less configurable |

### 4.3 Dense retriever

An open-weight bi-encoder (via `sentence-transformers`) maps queries and documents to vectors. Vectors are L2-normalized so inner product equals cosine similarity. Model choice is **OPEN-2**:

| Model | Dim | License | Notes |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Apache-2.0 | Very fast; weaker on BEIR; widely known baseline |
| `BAAI/bge-small-en-v1.5` | 384 | MIT | Stronger retrieval quality at the same size; uses a query instruction prefix |
| `intfloat/e5-small-v2` | 384 | MIT | Strong; requires `query: ` / `passage: ` prefixes |
| `BAAI/bge-base-en-v1.5` | 768 | MIT | Higher quality; ~2× vectors size and slower encoding |

Tradeoff to note: a stronger dense model makes T1 (+4 over dense-only) harder, because the dense baseline itself improves. A 384-dim model also makes binary quantization (T2) lossier than a higher-dimensional model would.

### 4.4 Fusion

Merges the BM25 and dense rankings. Method is **OPEN-5**:

| Method | Formula | Strengths | Weaknesses |
|---|---|---|---|
| Reciprocal Rank Fusion (RRF) | score(d) = Σ 1 / (k + rank(d)), k = 60 | No tuning; ignores incompatible score scales | Discards score magnitudes |
| Weighted score fusion | score(d) = α·norm(dense) + (1−α)·norm(bm25), min-max normalized | Often higher quality when α is tuned | α must be tuned on held-out queries, not the test set |

Both can be implemented and compared; the pipeline config selects one. Tuning α on test queries would inflate results and is not allowed — a dev split or cross-validation is used.

### 4.5 Reranker

A cross-encoder reads the query and a document together and outputs a relevance score. It is more accurate than the bi-encoder but too slow to run over the whole corpus, so it only scores the top-N fused candidates. Model choice is **OPEN-6**:

| Model | Size | Notes |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~22M params | Fast on CPU; standard baseline |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | ~33M params | Somewhat better, ~2× slower |
| `BAAI/bge-reranker-base` | ~278M params | Higher quality; much slower on CPU |

N (candidates reranked) trades quality for latency: larger N can recover more relevant documents but costs one cross-encoder forward pass per candidate. N will be swept (e.g. 20, 50, 100) and the curve reported.

---

## 5. FAISS index benchmark

### 5.1 Index types compared

All indexes are built from the same normalized float32 embeddings. Sizes assume d = 384.

| Index | Type | Bytes/vector (d=384) | Key parameters | Expected role |
|---|---|---|---|---|
| `IndexFlatIP` | Exact | 1,536 | — | Ground truth and speed baseline |
| `IndexHNSWFlat` | Graph | 1,536 + ~2·M·4 (links) | `M`, `efConstruction`, `efSearch` | Best speed/recall when memory is available |
| `IndexIVFFlat` | Clustered | 1,536 + 8 (id) | `nlist`, `nprobe` | Speed with tunable recall; needs training |
| `IndexScalarQuantizer` (fp16 / SQ8) | Quantized flat | 768 / 384 | `qtype` | 2× / 4× compression, small loss |
| `IndexPQ` | Product quantization | 48 (m=48, 8 bits) | `m`, `nbits` | 32× compression |
| `IndexIVFPQ` (optionally with OPQ) | Clustered + PQ | 48 + 8 (id) | `nlist`, `nprobe`, `m` | Memory-constrained + fast |
| `IndexBinaryFlat` / `IndexBinaryHNSW` | Binary (Hamming) | 48 | sign threshold | 32× compression, fastest distance |

### 5.2 What is measured per index and parameter setting

| Metric | Definition |
|---|---|
| ANN recall@10 | Fraction of the exact top-10 (from `IndexFlatIP`) that the index also returns in its top-10 |
| Latency p50 / p95 | Per-query wall time, single thread (`faiss.omp_set_num_threads(1)`), after warm-up |
| Throughput (QPS) | Batched queries per second at the default thread count |
| Index size | Bytes from `faiss.serialize_index` (or `serialize_index_binary`), plus bytes/vector |
| Build time | Training + adding vectors |
| Downstream nDCG@10 | Dense-only and full-pipeline quality with this index in place of exact search |

### 5.3 Method

- Parameter sweeps: `efSearch` ∈ {16 … 512}; `nprobe` ∈ {1 … 256}; `nlist` ≈ 4√N to 16√N (FAISS guideline); `M` ∈ {16, 32, 48}.
- IVF/PQ training uses a sample of corpus vectors; FAISS warns below 39 × `nlist` training points, so `nlist` is bounded by corpus size.
- Each timing is repeated (e.g. 5 runs) and the median reported. Hardware (CPU model, cores, RAM, OS) is written into every result file.
- Output: Pareto-frontier plots — recall@10 vs latency, recall@10 vs bytes/vector — and a "which index for which constraint" table, e.g. *lowest latency at ≥ 0.95 recall*, *smallest memory at < 1 point nDCG loss*, *fastest build*.

---

## 6. Evaluation methodology

| Item | Choice |
|---|---|
| Primary metric | nDCG@10 (reported × 100, as "points") |
| Secondary metrics | Recall@100 (does the right document reach the reranker at all?), MRR@10, latency |
| Systems compared | BM25 · Dense · Hybrid (fusion) · BM25 + rerank · Dense + rerank · Hybrid + rerank |
| Significance | Paired test over queries (e.g. paired t-test or randomization test) between the full pipeline and each baseline |
| Evaluation library | **OPEN-7**: `ranx` (metrics + fusion + significance tests in one package) or `pytrec_eval` (reference TREC implementation; via `pytrec-eval-terrier` wheels) |
| Reproducibility | Config files define every run; results saved as JSON with config, git commit, library versions and hardware |

Rules that keep results honest:

- Hyperparameters (fusion weight, `k1`/`b`, N) are tuned only on a dev split, never on test queries.
- Every system in a comparison uses identical queries, qrels and corpus.
- Results that miss a target are reported as missed.

---

## 7. CI quality gate

A GitHub Actions workflow that blocks quality regressions.

**Trigger:** every pull request and every push to `main`.

**Steps:**

1. Install pinned dependencies (pip/uv cache restored).
2. Restore the Hugging Face model cache and the embedding cache (`actions/cache`, keyed on model name + dataset + hash of encoding code). A cache miss re-encodes; a hit skips it.
3. Run the evaluation on the gate dataset (SciFact test, 300 queries) for each gated system (at minimum: dense, hybrid, hybrid + rerank).
4. Compare against `results/baseline.json`, which is committed to the repo.
5. **Fail** the job if any gated metric drops by more than the tolerance. Write a comparison table to the job summary (`$GITHUB_STEP_SUMMARY`).
6. Mark the job as a required status check on `main` (repository branch-protection setting) so a failing PR cannot be merged.

**Updating the baseline:** a deliberate change to `results/baseline.json` in the same PR, visible in review. Improvements are not auto-written to the baseline.

**Tolerance** is **OPEN-8**:

| Option | Strengths | Weaknesses |
|---|---|---|
| Fixed absolute (e.g. 0.5 nDCG points) | Simple, predictable | May flag noise or miss real small drops |
| Statistical (paired test, fail only if significantly worse) | Principled | Small query sets have low power; more complex |
| Both (fail if drop > threshold **and** significant) | Fewer false alarms | Most complex to explain |

**Cost:** GitHub Actions is free with unlimited minutes for public repositories on standard runners; private repositories on the Free plan get 2,000 minutes/month. Keeping the repo public removes the limit. The gate runs on CPU only; the target runtime is under ~15 minutes and will be measured once built. If reranking makes CI too slow, the gate reranks a smaller N than production — documented if done.

---

## 8. Tech stack and zero-cost choices

| Layer | Choice | Cost | Note / tradeoff vs paid option |
|---|---|---|---|
| Language | Python 3.11 | Free | — |
| Environment | `uv` (or `venv` + `pip`) with a lockfile | Free | — |
| BM25 | See OPEN-1 | Free | Replaces hosted search (Elasticsearch Cloud, Algolia); no managed scaling/HA |
| Embeddings | `sentence-transformers` + open model (OPEN-2) | Free | Replaces paid embedding APIs (OpenAI, Cohere); open small models are somewhat weaker than top paid APIs but run locally with no per-call cost |
| Vector index | `faiss-cpu` | Free | Replaces managed vector DBs (Pinecone etc.); no server, persistence is files on disk |
| Reranker | Open cross-encoder (OPEN-6) | Free | Replaces paid rerank APIs (Cohere Rerank); CPU latency is the main cost |
| Evaluation | `ranx` or `pytrec_eval` (OPEN-7) | Free | — |
| Data | BEIR via Hugging Face Hub | Free | Check per-dataset licenses |
| Compute (local) | Your Mac; PyTorch MPS backend for encoding on Apple Silicon | Free | FAISS runs on CPU (no FAISS GPU build on macOS) |
| Compute (extra) | Google Colab free / Kaggle Notebooks free GPU | Free | Quotas and GPU availability vary and change; sessions time out. Only needed if local encoding of the large corpus is too slow |
| CI | GitHub Actions | Free (public repo) | See §7 |
| Artifact storage | Local disk; optional Hugging Face Hub dataset repo for embeddings/indexes | Free | Do not commit large binaries to git; Git LFS free quota is small |
| Plots | `matplotlib` | Free | — |
| Interface | See OPEN-9 | Free | — |

---

## 9. Repository layout

Proposed; final names are set during scaffolding.

```
hybrid-search-engine/
├── pyproject.toml              # dependencies, tool config
├── uv.lock                     # pinned versions
├── configs/
│   ├── pipeline.yaml           # models, k₁, k₂, N, fusion method
│   └── bench_faiss.yaml        # index types and parameter sweeps
├── src/hybridsearch/
│   ├── data.py                 # load corpus, queries, qrels
│   ├── bm25.py                 # BM25 index build/search
│   ├── encoder.py              # embedding + on-disk cache
│   ├── dense.py                # FAISS index build/search wrapper
│   ├── fusion.py               # RRF and weighted fusion
│   ├── rerank.py               # cross-encoder reranking
│   ├── pipeline.py             # end-to-end query path
│   ├── evaluate.py             # metrics, significance tests
│   └── bench/faiss_bench.py    # index benchmark harness
├── scripts/                    # CLI entry points (index, search, eval, bench, gate)
├── tests/                      # unit tests (fusion math, metrics on toy data, index wrappers)
├── results/
│   ├── baseline.json           # committed CI baseline
│   └── runs/                   # per-run JSON + plots (large files git-ignored)
├── docs/
│   └── 01-project-documentation.md
└── .github/workflows/quality-gate.yml
```

---

## 10. Setup and running

Commands below are the **planned** interface. They will be confirmed or corrected as each part is implemented.

### 10.1 Requirements

- macOS, Linux or Windows (WSL) with Python 3.11
- ~4 GB free disk for small datasets and models; ~5 GB more if the large speed-benchmark corpus is used
- 8 GB RAM minimum; 16 GB recommended for the large-corpus benchmark (≈ 800 MB of float32 vectors for 523k × 384, plus index overhead)
- No GPU required

### 10.2 Install

```bash
git clone <repo-url> hybrid-search-engine
```

```bash
cd hybrid-search-engine
```

```bash
uv sync
```

Without `uv`:

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

Core dependencies: `faiss-cpu`, `sentence-transformers`, `torch`, the chosen BM25 library, the chosen evaluation library, `numpy`, `pyyaml`, `matplotlib`, `pytest`.

**macOS note:** FAISS and PyTorch each bundle an OpenMP runtime; loading both can abort with `OMP: Error #15`. The common workaround is `export KMP_DUPLICATE_LIB_OK=TRUE`, which Intel documents as unsafe in general. The project will instead try to avoid the conflict (e.g. compatible wheel versions, or encoding and FAISS search in separate processes) and document what works.

### 10.3 Planned workflow

```bash
python scripts/download.py --dataset scifact
```

```bash
python scripts/build_index.py --dataset scifact --config configs/pipeline.yaml
```

```bash
python scripts/search.py --dataset scifact --query "vitamin D and bone density"
```

```bash
python scripts/evaluate.py --dataset scifact --systems bm25,dense,hybrid,hybrid_rerank
```

```bash
python scripts/bench_faiss.py --dataset quora --config configs/bench_faiss.yaml
```

```bash
python scripts/quality_gate.py --baseline results/baseline.json
```

```bash
pytest
```

---

## 11. Open decisions

Each needs a choice from the project owner. Options and tradeoffs are in the sections referenced.

| ID | Decision | Options | Key tradeoff | Details |
|---|---|---|---|---|
| OPEN-1 | BM25 library | `bm25s` · Pyserini · `rank_bm25` · `tantivy` | Simplicity and no Java vs reference-grade Lucene numbers | §4.2 |
| OPEN-2 | Embedding model | MiniLM-L6 · bge-small · e5-small · bge-base | Speed and an easier T1 vs absolute quality; dim affects T2 | §4.3 |
| OPEN-3 | Quality datasets | SciFact, NFCorpus, FiQA, ArguAna (choose ≥ 2) | More datasets = stronger claims, longer runs | §4.1 |
| OPEN-4 | Speed-benchmark corpus | Quora (523k, real BEIR text) · MS MARCO 1M-passage subset · ANN-benchmark vectors (e.g. SIFT1M) | Project's own embeddings and qrels vs no encoding cost but unrelated vectors | §2, §5 |
| OPEN-5 | Fusion method | RRF · weighted · both | No tuning vs possibly higher quality with tuning | §4.4 |
| OPEN-6 | Reranker | MiniLM-L6 · MiniLM-L12 · bge-reranker-base | CPU latency vs quality | §4.5 |
| OPEN-7 | Evaluation library | `ranx` · `pytrec_eval` | One package with fusion + tests vs the reference implementation | §6 |
| OPEN-8 | CI tolerance rule | Absolute · statistical · both | Simplicity vs fewer false alarms | §7 |
| OPEN-9 | User interface | See below | Effort vs demo-ability | below |
| OPEN-10 | T2 measurement | Dense-only nDCG (strict) · full pipeline (reranker masks loss) · report both | Honesty of the claim vs a better-looking number | §2 |
| OPEN-11 | Rescoring for T2 | No rescoring (pure 32×) · rescoring from on-disk float32 vectors | Simpler, stricter claim vs much smaller quality loss | §2 |

**OPEN-9 — interface options**

| Option | What it is | Free hosting | Tradeoff |
|---|---|---|---|
| Library + CLI | Python package and scripts | Not hosted | Least effort; nothing to click for a reviewer |
| REST API | FastAPI + uvicorn, `/search` endpoint | Local only (free hosts with enough RAM for models are scarce) | Shows API design; must be run locally to try |
| Web demo | Gradio or Streamlit UI | Hugging Face Spaces free CPU tier (sleeps when idle) or Streamlit Community Cloud (lower memory limits) | Easiest for others to try; cold starts and CPU-only latency; needs a small corpus to fit free limits |

---

## 12. Build plan

Ordered so the quality gate exists before the parts it protects.

| Phase | Work | Done when |
|---|---|---|
| 0 | Scaffold repo, `pyproject.toml`, lockfile, data loader | SciFact corpus/queries/qrels load in tests |
| 1 | BM25 + dense (flat) retrieval; evaluation harness | nDCG@10 for BM25 and dense on SciFact match published ballparks |
| 2 | CI quality gate with initial baseline | A deliberately degraded PR fails the check |
| 3 | Fusion + reranker; ablations; N sweep | Six-system comparison table with significance tests (T1) |
| 4 | FAISS benchmark on the large corpus | Pareto plots and "which index wins" table (T3) |
| 5 | Compression study | Bytes/vector vs nDCG@10 table (T2) |
| 6 | Interface (OPEN-9) and final README | Project runnable by a new user from the README alone |

Document 2 (the detailed understanding guide) is written after phase 6, from the final code.

---

## 13. Known risks and limitations

| Risk | Effect | Mitigation |
|---|---|---|
| Targets are dataset-dependent | T1/T2/T3 may be missed on some datasets | Report per-dataset results as measured |
| Strong dense model narrows hybrid gains | T1 harder to reach | Report gains for more than one dense model if time allows |
| Binary quantization on 384-dim vectors | Larger quality loss than on 768–1024-dim models | Compare with PQ; use rescoring (OPEN-11) |
| CPU-only reranking latency | Full pipeline slower than retrieval-only | Report latency per stage; sweep N |
| Free compute quotas change | Colab/Kaggle may be unavailable | Design everything to run locally on CPU; cloud GPU is optional |
| macOS OpenMP conflict | Crash when FAISS and PyTorch load together | See §10.2 |
| CI runtime on free runners | Slow PR feedback | Cache models and embeddings; small gate dataset |
| Dataset licenses | Some BEIR sets are non-commercial | Non-commercial portfolio use only; note in README |

---

## 14. Glossary

| Term | Meaning |
|---|---|
| BM25 | Keyword ranking function based on term frequency, rarity across documents, and document length |
| Dense retrieval | Search by comparing embedding vectors of query and documents |
| Bi-encoder | Model that embeds query and document separately; fast, used for first-stage retrieval |
| Cross-encoder | Model that reads query and document together; accurate, used for reranking |
| FAISS | Open-source library from Meta for vector similarity search |
| ANN | Approximate nearest neighbor search; trades exactness for speed |
| HNSW | Graph-based ANN index |
| IVF | ANN index that clusters vectors and searches only the nearest clusters |
| PQ | Product quantization; compresses vectors into short codes |
| nDCG@10 | Ranking-quality metric over the top 10 results, rewarding relevant documents ranked higher; 0–1, reported × 100 as "points" |
| Recall@k (ANN) | Share of the exact top-k neighbors an approximate index also returns |
| RRF | Reciprocal Rank Fusion; merges rankings using only rank positions |
| qrels | Human relevance labels for (query, document) pairs |
| BEIR | Public benchmark suite of retrieval datasets with qrels |
