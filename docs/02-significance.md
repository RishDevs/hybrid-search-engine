# What this project is and why it matters

## The short version

I built a search engine that combines two different ways of finding documents, keyword matching and meaning matching, and then measured honestly what each part buys you in quality, speed and memory. Everything runs on free tools and a laptop. Every number in this project comes from a script in the repo that anyone can rerun.

## What it does, in plain terms

You type a question. The system looks for answers in two ways at once:

1. **Keyword search (BM25).** It finds documents that share words with your question, giving more credit to rare words. If you search for an error code or a drug name, this is what finds it.
2. **Meaning search (dense embeddings).** A small neural network turns your question and every document into a list of 384 numbers that captures what they're about. Documents whose numbers point in a similar direction to your question's are considered relevant, even if they share no words. "Heart attack" finds "myocardial infarction."

The two result lists are merged into one with a method called reciprocal rank fusion. For queries where the top results are close together, meaning the system isn't sure, a third, slower and more careful model (a cross-encoder) reads the question and each of the top 30 candidates side by side and casts its own vote on the order.

Around that core I built:

- **An evaluation harness** that scores the system on three public benchmark datasets with human relevance labels (SciFact, NFCorpus, FiQA), using the same metric the research community uses (nDCG@10).
- **A FAISS benchmark** that tests 12 ways of storing and searching the vectors on a 200,000-document corpus, measuring how much accuracy, speed and memory each one trades.
- **A CI quality gate.** On every pull request, GitHub Actions re-runs the evaluation and blocks the merge if quality drops.
- **A FastAPI service**, so the search can be called over HTTP.

## The real-world problem

Most search systems in production today have the same problem. Keyword search is precise but literal. Embedding search understands paraphrases but is fuzzy about exact strings: product codes, people's names, version numbers, rare medical terms. Users don't care which method is "better." They want the right document on the first page.

This matters most in the places search is used now:

- **RAG systems** (an LLM answers using retrieved documents). If retrieval misses the right passage, the LLM confidently answers from the wrong one. Retrieval quality is the ceiling on answer quality.
- **Internal docs, support tickets, legal and medical search.** These are full of exact identifiers *and* paraphrased descriptions, so neither method alone is enough.

The second problem is cost. A vector for every document takes memory, and searching all of them exactly gets slow as the collection grows. At 200K documents, exact search takes about 4 ms per query on one CPU thread. That's fine for a demo, but it grows linearly with the corpus: 10× the documents means roughly 10× the time and memory. Teams have to pick an index type and a compression level, and that choice is usually made by copying a blog post rather than by measuring.

## Why it's not trivial

Gluing BM25 and a vector database together takes an afternoon. Making it actually better, and knowing that it's better, is the hard part. These are the things I ran into that a tutorial build doesn't show:

1. **Equal-weight fusion can make things worse.** On FiQA (financial questions), BM25 alone scores 23.8 nDCG@10 and dense alone scores 40.4. Fusing them with equal weights gave 36.4, worse than dense alone, because the weak retriever drags the strong one down. The fix was to weight BM25 lower, but the weight has to be chosen on separate tuning data, never on the test set, or the result is meaningless.
2. **Rerankers can hurt.** The standard move is "retrieve 100, rerank with a cross-encoder, return the new order." On SciFact, the MiniLM rerankers made results *worse* at every depth I tried (20, 50, 100). They're trained on web search queries, and scientific claims look different. Even the 278M-parameter `bge-reranker-base` only helped at depth 20 (72.7 → 73.1) and dropped to 71.9 at depth 100. I changed the design so the reranker becomes one more vote in the fusion instead of having the final say. Across the three datasets, "reranker replaces the order" averaged 47.7 nDCG@10; "reranker votes" averaged 49.8.
3. **Approximate-search recall doesn't tell you search quality.** HNSW at 0.968 recall@10 (vs. exact search) lost 1.0 nDCG point, while IVF at almost the same recall (0.966) lost only 0.3. HNSW's misses fall on the hard queries where the relevant documents live. If you tune an index by recall alone, you can ship a quality regression without knowing it.
4. **Two native libraries fight on macOS.** FAISS and PyTorch each bundle their own OpenMP runtime. Loading both segfaults inside FAISS search. I diagnosed it, verified a fix against brute-force results, and shipped it as a one-line setup command plus an import-time check that fails with a clear message instead of a crash.
5. **Quality regressions are silent.** A change to tokenization or a dependency bump can quietly cost a point of nDCG. Nobody notices until users complain. That's why the CI gate exists.

## What makes this different from a tutorial

- **Every claim is measured, and the measurements include what didn't work.** The results include the reranker hurting SciFact, equal-weight RRF hurting FiQA, and SQ8 being *slower* than exact search on one thread. Those findings are more useful than a clean table that only goes up.
- **The tuning protocol is honest.** Fusion weights are grid-searched on BEIR train/dev splits (`hse tune`) and then frozen. Test queries are scored once. The router is calibrated on half the test queries and reported on the other half.
- **BM25 is written from scratch** as a sparse matrix (about 100 lines) instead of being imported. It matches the published Anserini/Lucene numbers: 68.0 vs 67.9 on SciFact, 32.1 vs 32.2 on NFCorpus, 23.8 vs 23.6 on FiQA. That agreement is how I know the implementation is right.
- **The index benchmark reports two operating points** for each index: the fastest setting that hits 0.95 recall, and the fastest setting that stays within 0.5 nDCG points of exact search. The second is what you should actually deploy with.
- **Quality is enforced, not just reported.** The CI gate turns "we measured it once" into "it can't silently get worse."
- **It costs nothing to run.** Open models, public datasets, free CI, CPU-friendly defaults.

## Headline results

Measured on an Apple M4 Pro. Quality numbers are nDCG@10 on BEIR test splits.

| | SciFact | NFCorpus | FiQA | mean |
|---|---|---|---|---|
| BM25 | 68.04 | 32.07 | 23.83 | 41.31 |
| Dense (bge-small) | 70.57 | 34.28 | 40.35 | 48.40 |
| Hybrid (weighted RRF) | 72.18 | 35.34 | 41.13 | 49.55 |
| Hybrid + reranker vote | 71.71 | 35.72 | 41.96 | **49.80** |

- The full pipeline is **+1.4 points over dense-only and +8.5 over BM25-only** on average. The gain over dense is statistically significant on NFCorpus (p = 0.002) and FiQA (p < 0.001) but not on SciFact (p = 0.13).
- On a 200K-document corpus, **binary quantization shrinks the index exactly 32×** (1,536 → 48 bytes per vector). Without rescoring it loses 1.8 nDCG points; rescoring the top 100 with full-precision vectors brings the loss to 0.01.
- **HNSW is 51× faster than exact search at 0.968 recall@10**, and 32× faster at the stricter "within 0.5 nDCG points" bar.
- The reranker accounts for about 97% of query latency on CPU (about 250 of 257 ms). The calibrated router switched it off on SciFact, where it hurts: mean latency went from 252 to 7 ms and nDCG went up. On the other datasets the margin signal turned out to be no better than random per query. That's a measured negative result, explained in the defense document.

Full tables: [`results/RESULTS.md`](../results/RESULTS.md).

## Who would care

- **Anyone building RAG or search** who needs to decide whether hybrid retrieval and reranking are worth their latency for *their* data. This project is a template for answering that with numbers.
- **Anyone choosing a vector index** under a memory or latency budget. The benchmark shows which index wins under which constraint, and why recall alone is the wrong thing to tune.
- **Anyone maintaining a search system**, who needs regressions caught before users catch them.
