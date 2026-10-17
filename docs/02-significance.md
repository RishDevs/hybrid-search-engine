# What this project is and why it matters

## The short version

I built a search engine that combines two different ways of finding documents, keyword matching and meaning matching, adds a careful third model that only runs when it's likely to help, and measured honestly what each part buys in quality, speed and memory. Everything runs on free tools and a laptop. Every number in this project comes from a script in the repo that anyone can rerun.

## What it does, in plain terms

You type a question. The system looks for answers in two ways at once:

1. **Keyword search (BM25).** It finds documents that share words with your question, giving more credit to rare words. If you search for an error code or a drug name, this is what finds it.
2. **Meaning search (dense embeddings).** A small neural network turns your question and every document into a list of 384 numbers that captures what they're about. Documents whose numbers point in a similar direction to your question's are considered relevant, even if they share no words. "Heart attack" finds "myocardial infarction."

The two result lists are merged into one with a method called reciprocal rank fusion. Then a small learned model, the **router**, looks at cheap signals (do the two searches agree? is the meaning search confident? how long is the query?) and predicts whether a third, much slower model would improve this particular answer. If yes, that model, a **cross-encoder**, reads the question next to each of the top 30 candidates and casts a heavily weighted vote on the final order.

Around that core I built:

- **An evaluation harness** that scores the system on three public benchmark datasets with human relevance labels (SciFact, NFCorpus, FiQA), using the metric the research community uses (nDCG@10).
- **A tuning process that never looks at the test set.** The reranker, how many candidates it sees, and how much each method's vote counts were all chosen on separate dev and train questions.
- **A FAISS benchmark** that tests 12 ways of storing and searching the vectors on a 200,000-document corpus, measuring how much accuracy, speed and memory each one trades.
- **A CI quality gate.** On every pull request, GitHub Actions re-runs the evaluation and blocks the merge if quality drops.
- **A FastAPI service**, so the search can be called over HTTP.

## The real-world problem

Keyword search is precise but literal. Embedding search understands paraphrases but is fuzzy about exact strings: product codes, people's names, version numbers, rare medical terms. Users don't care which method is "better." They want the right document on the first page.

This matters most where search is used now:

- **RAG systems** (an LLM answers using retrieved documents). If retrieval misses the right passage, the LLM confidently answers from the wrong one. Retrieval quality is the ceiling on answer quality.
- **Internal docs, support tickets, legal and medical search.** These are full of exact identifiers *and* paraphrased descriptions, so neither method alone is enough.

The second problem is cost. The most accurate model in the pipeline, the cross-encoder, is also by far the slowest: on CPU it takes about 1.6 seconds per query, against under 10 milliseconds for everything else combined. Running it on every query is expensive; never running it throws away most of the quality. Deciding *per query* is the interesting engineering problem.

The third problem is memory and speed as the collection grows. At 200K documents exact vector search takes about 4 ms per query on one CPU thread, and that grows linearly with corpus size. Teams have to pick an index type and a compression level, and that choice is usually made by copying a blog post rather than by measuring.

## Why it's not trivial

Gluing BM25, a vector database and a reranker together takes an afternoon. Making it actually better, and knowing it's better, is the hard part. These are the things I ran into:

1. **Equal-weight fusion can make things worse.** On FiQA (financial questions), BM25 alone scores 23.8 nDCG@10 and dense alone scores 40.4. Fusing them with equal weights gave 36.4, worse than dense alone, because the weak retriever drags the strong one down. The fix was weighting, and the weight had to be chosen on separate tuning data.
2. **The obvious reranker hurts.** The default choice in every tutorial, `ms-marco-MiniLM-L-6-v2`, made SciFact *worse* (72.7 → 68.3) when used the standard way. It's trained on web search queries, and scientific claims look different. I compared four rerankers on dev data. The winner, `gte-reranker-modernbert-base` (150M parameters), also beat the much larger `bge-reranker-v2-m3` (568M) on SciFact-train (79.2 vs 77.3) and runs about 2.4× faster on CPU.
3. **Picking *which* queries to rerank is hard.** My first router used one signal, the gap between the top two fused results. With the weak reranker it was no better than random. With the strong reranker, a ridge regression over eight signals works: the strongest predictor of benefit is *disagreement* between keyword and meaning search, not low confidence.
4. **Approximate-search recall doesn't tell you search quality.** HNSW at 0.968 recall@10 (vs. exact search) lost 1.0 nDCG point, while IVF at almost the same recall (0.966) lost only 0.3. HNSW's misses fall on the hard queries where the relevant documents live. Tune an index by recall alone and you can ship a quality regression without knowing it.
5. **Two native libraries fight on macOS.** FAISS and PyTorch each bundle their own OpenMP runtime, and loading both segfaults inside FAISS search. I diagnosed it, verified a fix against brute-force results, and shipped it as a one-line setup command plus an import-time check.
6. **Quality regressions are silent.** A change to tokenization, a dependency bump or a lower candidate depth can quietly cost points. That's why the CI gate exists, and I tested that it actually fails when it should.

## What makes this different from a tutorial

- **The tuning protocol is honest.** Reranker choice, rerank depth and fusion weights were grid-searched on BEIR train/dev splits (`hse tune`), and the router was trained on the same dev queries. Test queries were scored once, at the end.
- **Every claim is measured, including what didn't work.** The weak reranker hurting SciFact, equal-weight fusion hurting FiQA, the margin router being no better than random, SQ8 being *slower* than exact search on one thread, and the router missing its dev recovery target on test are all in the results.
- **BM25 is written from scratch** as a sparse matrix (about 100 lines). It matches the published Anserini/Lucene numbers: 68.0 vs 67.9 on SciFact, 32.1 vs 32.2 on NFCorpus, 23.8 vs 23.6 on FiQA. That agreement is how I know the implementation is right.
- **Routing is evaluated against a random baseline.** "Recovers X% of the gain" only means something next to what random routing at the same share would recover.
- **The index benchmark reports two operating points** per index: fastest at 0.95 recall, and fastest within 0.5 nDCG points of exact search. The second is what you should deploy.
- **Quality is enforced, not just reported.** The CI gate turns "we measured it once" into "it can't silently get worse."
- **It costs nothing to run.** Open models, public datasets, free CI.

## Headline results

Measured on an Apple M4 Pro; latency on its CPU. Quality numbers are nDCG@10 on BEIR test splits.

| | SciFact | NFCorpus | FiQA | mean |
|---|---|---|---|---|
| BM25 | 68.04 | 32.07 | 23.83 | 41.31 |
| Dense (bge-small) | 70.57 | 34.28 | 40.35 | 48.40 |
| Hybrid (weighted RRF) | 72.12 | 35.63 | 40.49 | 49.42 |
| **Hybrid + reranker vote** | **77.78** | **37.35** | **46.54** | **53.89** |

- The full pipeline is **+5.5 nDCG@10 over dense-only and +12.6 over BM25-only** on average, significant on every dataset (paired t-test, p ≤ 1e-5).
- **The learned router reranks 49% of test queries and recovers 69% of the reranker's gain**, where random routing at the same share would recover 49%. Mean CPU latency drops from **1,594 ms to 678 ms (57% less)**.
- On a 200K-document corpus, **binary quantization shrinks the index exactly 32×** with a 0.01-point nDCG loss when the top 100 are rescored, and **HNSW is 51× faster than exact search at 0.968 recall@10** (32× within 0.5 nDCG points).

Full tables: [`results/RESULTS.md`](../results/RESULTS.md).

## Who would care

- **Anyone building RAG or search** who needs to decide whether hybrid retrieval and reranking are worth the latency for *their* data. This project is a template for answering that with numbers.
- **Anyone paying for reranking**, since a router that halves reranker calls while keeping most of the gain halves that bill.
- **Anyone choosing a vector index** under a memory or latency budget.
- **Anyone maintaining a search system**, who needs regressions caught before users catch them.
