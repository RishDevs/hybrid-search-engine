# Defending the project

This is for demos, presentations and interviews. Everything here is what I'd say out loud. Every number comes from `results/` and can be regenerated with the commands in the README.

---

## Numbers to know cold

| What | Number | Where it comes from |
|---|---|---|
| Full pipeline vs dense-only | **+5.49** nDCG@10 (mean of 3 datasets) | `results/eval/*.json` |
| Full pipeline vs BM25-only | **+12.58** nDCG@10 | same |
| Per dataset, full pipeline | SciFact 77.78 · NFCorpus 37.35 · FiQA 46.54 | same |
| Significance vs dense | p = 2.5e-8 · 1.1e-5 · 7.7e-15 (paired t-test) | same |
| Hybrid alone (no reranker) vs dense | +1.02 mean; significant only on NFCorpus | same |
| Reranker voting vs replacing the order | 53.89 vs 53.46 (small edge with a strong reranker) | `hybrid+rerank` vs `hybrid+ce-only` |
| Reranker selection on dev (mean nDCG@10) | MiniLM-L6 50.98 · bge-reranker-base 51.24 · **gte-modernbert 54.87** · dense 49.49 | `results/tune.json` |
| bge-reranker-v2-m3 (568M) on SciFact-train | 77.30, vs gte 79.20, and 2.4× slower | dev experiment |
| Tuned settings | gte-reranker-modernbert-base, depth 30, w_bm25 0.3, w_dense 1.0, w_rerank 4.0 | `results/tune.json` |
| Rerank depth on dev | 10 → 53.09 · 20 → 54.30 · 30 → 54.87 | same |
| Router on test (pooled) | reranks **49%**, recovers **69%** of the gain; random at 49% recovers 49% | `results/routing/summary.json` |
| Router nDCG (pooled test) | none 46.72 · random 49.10 · **routed 50.06** · always 51.58 | same |
| CPU latency per query | skip reranker ~8 ms · always rerank **1,594 ms** · routed **678 ms (−57%)** | same |
| Routed pipeline vs dense | 52.24 vs 48.40 mean (+3.8) at 43% of always-rerank latency | per-dataset routing files |
| My BM25 vs Anserini | 68.0/67.9 · 32.1/32.2 · 23.8/23.6 | sanity check |
| Exact search, 200K × 384 | 4.07 ms/query, 1 thread | `results/bench/quora-200k.json` |
| HNSW32 | **51×** faster at 0.968 recall (−1.02 nDCG); **32×** within 0.5 nDCG | same |
| Binary quantization | **32.0×** smaller; −1.80 nDCG raw, **−0.01** with top-100 rescoring | same |
| CI gate | fails at −0.5 points; cutting depth 100 → 10 failed it; Linux CI reproduced the Mac baseline exactly | verified |

---

## 30-second version

"I built a hybrid search engine that runs keyword search and embedding search together, merges them with weighted rank fusion, and lets a cross-encoder vote on the top results. On three BEIR datasets it's 5.5 nDCG points better than embedding search alone and 12.6 better than keyword search, with every setting chosen on dev data. The cross-encoder costs 1.6 seconds per query on CPU, so I trained a small router that predicts which queries it will actually help. That cuts mean latency 57% while keeping 69% of the gain, against 49% for random routing. I also benchmarked 12 FAISS index types on 200,000 documents, and a CI check blocks any change that makes search worse."

## 2-minute version

"Keyword search and embedding search fail in opposite ways. Keyword search can't match 'heart attack' to 'myocardial infarction.' Embedding search is fuzzy on exact things like product codes and names. So I run both and merge their rankings with reciprocal rank fusion, which works on rank positions instead of raw scores, so I don't have to make BM25 scores and cosine similarities comparable.

Equal-weight fusion made FiQA *worse* than embeddings alone, because BM25 is weak there. So I weight it. Then I add a cross-encoder, which reads the query and each candidate together. The standard tutorial reranker, MiniLM trained on web search, made scientific search worse. So I compared four rerankers on the dev splits, never the test set. gte-reranker-modernbert won clearly: 54.9 mean dev nDCG versus 51 for the others and 49.5 for embeddings alone. It even beat a model four times its size. I use it as a heavily weighted vote in the fusion rather than letting it overwrite the order. On test that's 53.9 average nDCG, plus 5.5 over dense, significant on every dataset.

The catch is cost. The cross-encoder is 1.6 seconds per query on CPU, versus 8 milliseconds for everything else. My first router used one signal, how close the top two results are, and it was no better than random. So I trained a ridge regression on eight cheap signals to predict each query's gain, using dev queries. The biggest signal turned out to be *disagreement* between keyword and embedding search, not low confidence. On test it reranks 49% of queries and recovers 69% of the gain, while random routing at the same rate gets 49%. Mean latency drops 57%.

Separately, I benchmarked 12 FAISS indexes on 200,000 documents. The key finding is that recall against exact search isn't enough: HNSW and IVF had the same recall but HNSW lost three times more quality. And a GitHub Actions gate re-runs the evaluation on every pull request and fails if quality drops half a point."

---

## The architecture, narrated at a whiteboard

*(Draw left to right as you talk.)*

"On the left is the query. It goes down two paths at the same time.

**Top path: BM25.** I wrote it myself as a sparse matrix, one row per document and one column per word, where each cell already holds that word's BM25 contribution. At query time I add up the columns for the query's words. Under a millisecond. Top 100 out.

**Bottom path: dense.** The query goes through bge-small, a 33-million-parameter model that outputs a normalized 384-number vector. FAISS finds the 100 documents with the highest inner product, which equals cosine similarity. Exact search at these corpus sizes; the benchmark tells you what to switch to when the corpus grows.

**Middle: fusion.** Both lists go into weighted reciprocal rank fusion. Each document gets weight over 60 plus rank from each list: BM25 weight 0.3, dense 1.0. One merged list.

**Then the router.** From things I already have, I compute eight numbers: how many of the top 10 the two searches share, how sharply the dense scores fall off, the BM25 gap, the query length, and a few more. A linear model trained on dev queries turns those into a predicted nDCG gain. Below the threshold, I return the fused list: about 8 milliseconds on CPU.

**Above the threshold,** the top 30 go to the cross-encoder, a 150-million-parameter ModernBERT that reads query and document together. That's about 1.6 seconds for 30 on CPU. Its ranking goes back into the same fusion as a third vote with weight 4, so it dominates, but BM25 and dense still break ties and keep anything it didn't see. Top 10 out.

**Off to the side,** draw two boxes. One is evaluation: three BEIR datasets with human labels, tuning and router training only on dev splits, test scored once, paired significance tests. The other is CI: every PR re-runs SciFact on CPU against a baseline file. Half a point worse fails the build.

**And the FAISS benchmark** is its own experiment: 12 index types, 200K docs, speed vs memory vs quality."

---

## Design decisions and what I rejected

| Decision | What I chose | Alternatives I rejected | Why |
|---|---|---|---|
| Combining BM25 and dense | **Weighted RRF** (k = 60, BM25 0.3) | Linear combination of normalized scores; learned fusion | BM25 scores are unbounded and shift per query, so normalization is fragile. RRF only needs ranks. Weighting fixed FiQA. |
| Which cross-encoder | **gte-reranker-modernbert-base** (150M, Apache-2.0) | ms-marco-MiniLM-L-6 (22M); bge-reranker-base (278M); bge-reranker-v2-m3 (568M) | Best mean dev nDCG by 3.6 points over the next one. v2-m3 was worse on SciFact-train and 2.4× slower. MiniLM is fastest but out of domain. |
| Role of the cross-encoder | **Weighted vote (4.0) in RRF** | Replace the top-30 order | With a weak reranker, voting was worth 2 points. With gte it's a smaller, consistent edge (53.89 vs 53.46; significant on NFCorpus). Voting also keeps BM25/dense signal for docs the reranker didn't see. |
| Rerank depth | **30** | 10, 20 | Best on dev. Depth 20 costs 0.57 dev points for a third less reranker compute, a reasonable cheaper setting. |
| Routing signal | **Ridge regression on 8 features predicting nDCG gain** | Single margin threshold; logistic "helps / doesn't"; per-dataset features | The margin alone was uncorrelated with gain. In cross-validation on dev, ridge-on-gain beat logistic and the one-hot variants. Linear means 9 numbers to inspect. |
| Router threshold | **Set on cross-validated dev predictions to recover 80% of the gain** | Fixed share; threshold on in-sample predictions | In-sample predictions are optimistic and would set the threshold too low. A gain target is meaningful; a fixed share isn't. |
| Embedding model | **bge-small-en-v1.5** (384-d) | all-MiniLM-L6-v2; bge-base; paid APIs | Stronger than MiniLM at the same size; bge-base doubles memory; paid APIs break the zero-cost rule. |
| Query instruction prefix | **Keep it** | Drop it | Dropping it helped SciFact test but lost on NFCorpus/FiQA dev. Decided on dev. |
| BM25 | **Own implementation on scipy.sparse** | Pyserini/Lucene; bm25s; Elasticsearch | No Java or server; fully explainable; validated against Anserini's published numbers. |
| Reranker score cache | **SQLite keyed by hash(model, query, passage)** | Recompute every run; pickle files | A cross-encoder score depends only on the text pair, so caching is exact. Tuning reruns in under a minute, and CI only scores new pairs. |
| Tuning protocol | **Dev/train splits only; test scored once** | Tune on test | Tuning on test inflates results and is the most common mistake in retrieval projects. |
| Serving index | **Flat (exact)** | HNSW, IVF | 4K–58K docs: exact is a few ms with no recall loss. |
| Benchmark latency | **Single query, single thread** | Batched throughput | Online search answers one query at a time; batching flatters Flat. |
| Benchmark operating points | **Recall ≥ 0.95, and within 0.5 nDCG** | Recall only | HNSW and IVF at the same recall differed 3× in quality loss. |
| CI gate | **SciFact on CPU, 0.5-point tolerance, cached models and scores** | All datasets; statistical test | Fits free runners; deterministic; caught a real regression in testing. |
| macOS OpenMP conflict | **Symlink faiss's libomp to torch's + import-time check** | `KMP_DUPLICATE_LIB_OK`; separate processes | The env var still segfaulted; IPC per query is worse. Verified against brute force. |

---

## Hard questions

**1. You changed the reranker after seeing the first test results. Isn't that test-set shopping?**

That's the right question. The first version, with MiniLM, was evaluated on test and came out at +1.4 over dense. I then went back and compared rerankers, but the comparison, the depth, the weights and the router were all decided on dev/train splits, and the new test numbers were computed once after that. The honest caveat is that knowing the first result motivated a second round, which is a researcher degree of freedom. The safeguards: the dev winner was clear (by 3.6 points), the test gain is significant at p ≤ 1e-5 on all three datasets, and the full dev grid is published in `results/tune.json`. The cleanest fix would be a fourth dataset nobody has touched, evaluated once.

**2. Why did gte-modernbert beat a model four times its size?**

Training data and architecture, not size. bge-reranker-v2-m3 is multilingual (XLM-RoBERTa), so a lot of its capacity covers languages I don't need. gte-reranker-modernbert is English, built on ModernBERT (a 2024 BERT redesign with better attention and training), and trained on a broad retrieval mix rather than MS MARCO web queries alone. On SciFact-train it scored 79.2 vs 77.3.

**3. 1.6 seconds per reranked query on CPU. Isn't that too slow?**

For interactive search, on CPU, yes. That's why the router exists: mean latency is 678 ms. Beyond that, in order: rerank depth 20 instead of 30 (−0.57 dev points, a third less compute), int8 ONNX export (often 2–3× faster on CPU), and a GPU, where the same model runs at about 34 pairs/s on my laptop's GPU versus 19 on its CPU, and much faster on a real GPU. For a RAG pipeline where an LLM call takes several seconds anyway, 1.6 seconds of reranking is often acceptable.

**4. The router recovers 69% of the gain on test, but you targeted 80%. Why the miss?**

The threshold was set on dev predictions and transferred to test, and the test queries looked different. On dev the threshold sent 40% of queries to the reranker; on test it sent 49%, mostly FiQA queries (74% of FiQA vs 18% of NFCorpus). Even so, the recovered share fell short, because the model's ranking of queries is noisier on new data than in cross-validation. Options: calibrate per dataset when you have in-domain dev data, or use a conformal-style margin to make the target hold with high probability.

**5. Is the router actually better than random, or just a different share?**

Better. At the same 49% share, random routing would score 49.10 pooled nDCG; the router scores 50.06. Equivalently, it recovers 69% of the gain where random recovers 49%. Per dataset: SciFact 54% of the gain at 28% share, NFCorpus 32% at 18%, FiQA 81% at 74%. All beat their random baselines, though NFCorpus only modestly.

**6. What does the router actually learn?**

The largest weights (on standardized features): overlap of the top 10 between BM25 and dense (−0.025: more agreement, less gain), how sharply the dense scores fall from rank 1 to 10 (−0.016: a clear winner means less to fix), query length (+0.015: longer queries gain more), and the BM25 top-two gap (−0.014). In plain terms: rerank when the two retrievers disagree, nothing stands out, and the query is long. The fused margin, my original signal, got almost zero weight.

**7. Why a linear model and not gradient boosting?**

There are 1,633 dev queries and the target (per-query nDCG gain) is very noisy: 57% of queries have zero gain and 13% get worse. A flexible model would overfit that. Ridge has nine numbers I can read and explain, trains in milliseconds, and in cross-validation it beat logistic regression. With 10× the queries, or click data, I'd try boosted trees.

**8. Hybrid without the reranker is only +1 over dense. Why keep BM25 at all?**

Three reasons. It's significant on NFCorpus. It's nearly free (under 2 ms). And it feeds the reranker: BM25 can pull a document with a rare exact term into the top 30 that dense ranked at 60, and the reranker can then promote it. BM25 is also an input to the router, through the overlap feature, which is its strongest signal.

**9. Where's the bottleneck at query time?**

The cross-encoder. On CPU: query encoding about 7 ms, FAISS under 2 ms, BM25 under 2 ms, fusion and features under 1 ms, reranking 30 pairs about 1,600 ms. The fixes are the ones in question 3.

**10. How would this scale to 100 million documents?**

- **Dense vectors:** 154 GB as float32, so IVF-PQ or HNSW over quantized vectors, sharded, with rescoring from SSD. My benchmark shows binary + rescore keeps quality.
- **Exact search** would take about 2 seconds per query on one thread, so approximate search becomes mandatory.
- **BM25:** mine keeps the whole matrix in memory and rebuilds to add documents. At that scale I'd use Lucene through OpenSearch.
- **Reranker and router:** unchanged. They only see 30 docs per query whatever the corpus size.

**11. HNSW or IVF: when would you pick which?**

HNSW for lowest latency when memory allows: 32× faster than exact within 0.5 nDCG, about 18% memory overhead for graph links, awkward deletes. IVF when memory is tight or data changes: 19× faster at the quality point, nearly no overhead, combines with PQ, but needs training and its centroids go stale.

**12. Why did HNSW lose 1 nDCG point at 0.968 recall when IVF lost 0.3 at 0.966?**

Average recall hides *which* queries miss. HNSW misses cluster on hard queries where the greedy graph walk gets stuck locally, and those are the queries whose relevant documents are unusual. IVF misses are spread out. That's why I added the quality-based operating point.

**13. SQ8 was 3× slower than exact search. Bug?**

No. Single-threaded flat float32 search is a tight SIMD loop. SQ8 decodes each byte back to a float first, and at this size that costs more than the memory bandwidth it saves. SQ8 is a memory optimization, not a speed one.

**14. 32× smaller, but PQ is 30.7×. Which is it?**

Binary quantization is exactly 32× (48 bytes per vector). PQ48 is 30.7× because I count the real serialized size including codebooks. I report the real number on purpose.

**15. Rescoring needs the full vectors. What did compression save?**

RAM. The compressed index (9.6 MB) searches in memory; the full vectors (307 MB) stay memory-mapped on disk and only 100 rows per query are read. "32× less RAM," not "32× less storage."

**16. How do you add or delete documents?**

Today I rebuild. BM25 IDF and average length change with every addition, so a correct update touches every weight; Lucene solves this with query-time IDF and segment merging. FAISS Flat and HNSW support adds; HNSW deletes are awkward; IVF needs periodic retraining. The reranker score cache is unaffected, since it's keyed by text.

**17. What happens with bad input?**

- **Empty query, or k outside 1..100:** a 422 from the API.
- **Only stopwords:** BM25 returns nothing, dense still works, and the router features handle zero BM25 hits (tested).
- **Non-English:** bge-small and the stemmer are English-only, so quality drops silently; I'd add language detection.
- **Long documents:** truncated at 512 tokens by both neural models.
- **Duplicate documents** take two slots.
- **Very long queries:** capped at 1,000 characters.

**18. What breaks under load?**

The cross-encoder. Each reranked query is about 1.6 s of CPU, and PyTorch already uses several cores per call, so concurrent requests fight for cores and p95 climbs fast. Fixes: dynamic batching of reranker calls across requests, a separate reranker service on GPU, and load shedding, where the router threshold rises when the queue is long. The router makes that last one natural: under load, only the queries with the highest predicted gain get reranked. Retrieval itself (under 10 ms) is fine.

**19. The CI gate only checks SciFact. Couldn't a change break FiQA and pass?**

Yes; it's a CI-time tradeoff. The first cold gate run took 88 minutes on GitHub's free runner, almost all of it reranking SciFact on CPU; with cached scores, later runs only score changed pairs. SciFact catches regressions in shared code (tokenizer, fusion, depth, model loading). Dataset-specific regressions are caught by `make eval` before a release. Adding FiQA would mean one more slow cold run, then cheap runs.

**20. Won't a 0.5-point tolerance be flaky?**

The evaluation is deterministic: two local runs gave +0.00 on every metric, and GitHub's Linux runner, scoring the reranker from scratch on CPU, reproduced the Mac baseline exactly on all six gated numbers. Cutting candidate depth from 100 to 10 dropped hybrid nDCG by 0.94 and recall@100 by 5.9 and failed the gate.

**21. Isn't caching reranker scores in CI cheating the test?**

No. A cross-encoder score is a pure function of (model, query text, passage text), and the cache key is a hash of exactly those. If code changes what text reaches the reranker (truncation, title handling, a different candidate set), the keys change and the new pairs are scored fresh. If the model changes, it's a different cache file. What the cache can't catch is a change inside the model library that alters scores for the same text, which is why the cache key in CI also changes with the config file.

**22. Multiple comparisons?**

I report five p-values per dataset. The pre-specified comparison, full pipeline vs dense, has p ≤ 1.1e-5 on every dataset, which survives any reasonable correction. The vote-vs-replace comparison is weak (p = 0.17, 0.016, 0.62) and I describe it as a small edge, not a finding.

**23. Why write BM25 yourself?**

To understand and explain every line, because it's about 100 lines, fast enough here (under 2 ms/query on 200K docs), and verifiable against Anserini's published BEIR numbers. At production scale I'd use Lucene.

**24. Your 200K Quora corpus keeps every relevant doc. Doesn't that make it easier?**

Yes. Fewer distractors, so absolute nDCG (93.2) is higher than on full Quora. The benchmark compares index types against exact search on the same corpus, which is what the numbers claim.

**25. Could bge-small or the reranker have seen these datasets in training?**

Possibly. These models are trained on large public retrieval mixes, and I can't rule out overlap with BEIR train splits. That's a reason not to compare my absolute numbers too hard with other papers. Every comparison I make uses the same models underneath.

**26. In production you have no labels. How would you know the reranker and router still help?**

Interleaving experiments, click-through and dwell time as implicit labels, a periodic small human-labeled sample, or an LLM judge on a sample. The router can be retrained on logged (features, observed gain) pairs from a small always-rerank traffic slice.

**27. Why not a vector database?**

Pinecone is paid and off-machine. Qdrant and Weaviate are free to self-host and would give persistence and filtering, but for an experiment about index tradeoffs I wanted FAISS's direct control. Most vector databases use HNSW internally, so the HNSW numbers carry over.

**28. What would you do differently?**

- Compare rerankers on dev from the start, instead of shipping the tutorial default first.
- Build the router around predicting gain from day one.
- Hold out a fourth dataset as a final untouched test.
- Use randomization tests instead of t-tests.

---

## Known limitations, and what the next version adds

**Limitations:**

1. **Reranker cost:** the cross-encoder is 1.6 s per query on CPU; the router halves the mean, but p95 for reranked queries is still about 1.6 s.
2. **Router transfer:** it missed its dev target on test (69% vs 80% of the gain); its decisions transfer imperfectly across datasets.
3. **Second-round selection:** the reranker was selected on dev, but after seeing the first test results (question 1).
4. **Static index:** BM25 needs a full rebuild to add documents, and everything is in memory on one machine.
5. **Coverage:** English only, and documents are truncated at 512 tokens.
6. **CI coverage:** the gate covers one dataset.

**Next version:**

1. **Faster reranking:** int8 ONNX cross-encoder, depth 20, GPU serving with dynamic batching.
2. **Better routing:** per-domain calibration, conformal thresholds, and retraining from logged traffic.
3. **Query-dependent fusion weights**, e.g. more BM25 weight when the query has IDs or rare terms.
4. **Incremental indexing** with query-time IDF.
5. **More evaluation:** more BEIR datasets including one held out from all decisions, and randomization tests.
6. **Production API:** load shedding through the router, per-stage metrics, auth and rate limits.

---

## Resume bullets

**Python, FAISS, BM25, Sentence-Transformers, cross-encoder, FastAPI, GitHub Actions**

- Built hybrid search (from-scratch BM25, bge-small + FAISS, weighted RRF, cross-encoder voting); **+5.5 nDCG@10 over dense and +12.6 over BM25** across 3 BEIR datasets (p < 1e-5), with reranker and weights selected on dev splits only.
- Benchmarked 12 FAISS indexes on 200K docs: **51× faster than exact search at 0.97 recall@10**; **32× smaller index with 0.01-point nDCG loss** via binary quantization + rescoring.
- Trained a ridge-regression router that predicts per-query reranker gain, **cutting mean CPU latency 57% while recovering 69% of the reranking gain on 49% of queries** (random routing: 49%); added a GitHub Actions gate that fails PRs on >0.5-point quality regressions; served via FastAPI.

Every number above traces to a file in `results/`:

| Claim | File |
|---|---|
| +5.5 / +12.6 nDCG@10, p-values | `results/eval/{scifact,nfcorpus,fiqa}.json` |
| Dev-only selection | `results/tune.json` |
| 12 indexes, 200K docs, 51×, 0.968 recall | `results/bench/quora-200k.json` (HNSW32,Flat, efSearch=16) |
| 32×, 0.01-point loss | same file (BFlat, `rescored`) |
| 57% latency cut, 69% gain on 49% of queries, random 49% | `results/routing/summary.json` |
| Router weights | `results/router.json` |
| 0.5-point gate | `src/hybrid_search/gate.py`, `.github/workflows/ci.yml`, `results/baseline.json` |

### Compared with the earlier bullets

| Original draft | First build (MiniLM reranker, margin router) | Now |
|---|---|---|
| +3.6 over dense, +11.4 over BM25 | +1.4, +8.5 | **+5.5, +12.6** |
| 32× compression, 0.8-point loss | 32×, 0.01 with rescoring | same |
| 18× faster at 0.96 recall | 51× at 0.968 recall | same |
| 34% latency cut, 97% of nDCG, 41% of queries | per-dataset switch only; per-query no better than random | **57% cut, 69% of the gain on 49% of queries (random: 49%)** |
