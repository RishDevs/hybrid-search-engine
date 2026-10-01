# Hybrid Search Engine

Keyword search misses meaning; vector search misses exact terms like IDs and names. This project combines both — BM25 and dense embedding retrieval — fuses their results, reranks the top hits with a cross-encoder, and measures the tradeoff between search quality, speed, and memory.

> **Status:** In development. Results will be added once experiments are complete.

## Features

- **Hybrid retrieval** — BM25 and embedding search run on every query; their rankings are fused (Reciprocal Rank Fusion or weighted score fusion).
- **Cross-encoder reranking** — the top fused candidates are re-scored by a cross-encoder for higher precision.
- **FAISS index benchmark** — compares exact, HNSW, IVF, scalar-quantized, product-quantized, and binary indexes on recall, latency, memory, and build time to show which index wins under which constraint.
- **CI quality gate** — a GitHub Actions check evaluates retrieval quality on every pull request and blocks changes that make it worse.

## Goals

| Goal | Measured as |
|---|---|
| Better quality than vector-only and BM25-only search | nDCG@10 on BEIR test sets |
| ~32× smaller index with < 1 point quality loss | Bytes per vector vs. float32; nDCG@10 drop |
| ~20× faster than exact search at 95% recall | Query latency vs. exact FAISS search at recall@10 ≥ 0.95 |

## How it works

```
query ─┬─▶ BM25 ───────────────┐
       │                       ├─▶ fusion ─▶ cross-encoder rerank ─▶ top-10
       └─▶ encoder ─▶ FAISS ───┘
```

Documents are indexed offline: tokenized for BM25 and embedded once into cached vectors, from which every FAISS index variant is built.

## Zero-cost stack

Everything runs on free, open-source tools — no paid APIs or infrastructure.

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Keyword search | BM25 (open-source library) |
| Embeddings & reranking | `sentence-transformers` with open-weight models |
| Vector search | `faiss-cpu` |
| Data | [BEIR](https://github.com/beir-cellar/beir) benchmark datasets |
| CI | GitHub Actions |

## Getting started

Setup instructions will be added as the code lands. Requirements: Python 3.11, 8 GB RAM (16 GB recommended for large-corpus benchmarks), no GPU needed.

## Documentation

- [Project documentation](docs/01-project-documentation.md) — architecture, components, evaluation method, setup, and open design decisions.

## Roadmap

- [ ] Data loading and evaluation harness
- [ ] BM25 and dense retrieval baselines
- [ ] CI quality gate
- [ ] Fusion and reranking
- [ ] FAISS index benchmark
- [ ] Index compression study
- [ ] User interface and final write-up

## License

[MIT](LICENSE)
