# Defending the project

This is for demos, presentations and interviews. Everything here is what I'd say out loud. Every number comes from `results/` and can be regenerated with the commands in the README.

---

## Numbers to know cold

| What | Number | Where it comes from |
|---|---|---|
| Full pipeline vs dense-only | **+1.40** nDCG@10 (mean of 3 datasets) | `results/eval/*.json` |
| Full pipeline vs BM25-only | **+8.48** nDCG@10 | same |
| Per dataset, full pipeline | SciFact 71.71 · NFCorpus 35.72 · FiQA 41.96 | same |
| Gain over dense significant? | NFCorpus p = 0.002, FiQA p < 0.001, SciFact p = 0.13 (no) | paired t-test |
| Reranker replacing the order vs voting | 47.68 vs 49.80 mean | `hybrid+ce-only` vs `hybrid+rerank` |
| Equal-weight RRF on FiQA | 36.4, vs dense 40.4 (worse) | early experiment that led to weighting |
| Tuned weights | BM25 0.1, dense 1.0, reranker 0.5 | `results/tune.json` (dev splits only) |
| My BM25 vs Anserini | 68.0/67.9 · 32.1/32.2 · 23.8/23.6 | sanity check |
| Exact search, 200K × 384 | 4.07 ms/query, 1 thread | `results/bench/quora-200k.json` |
| HNSW32 at recall ≥ 0.95 | **51×** faster (0.968 recall), −1.02 nDCG | same |
| HNSW32 within 0.5 nDCG | **32×** faster (efSearch=32, −0.44) | same |
| IVF4096 within 0.5 nDCG | 19× faster | same |
| Binary quantization | **32.0×** smaller (48 B/vector); −1.80 nDCG raw, **−0.01** with top-100 rescoring | same |
| PQ48 | 30.7× smaller (codebook overhead); −2.22 raw, −0.00 with rescoring | same |
| Query latency on CPU | ~7 ms without reranker, ~250 ms with | `results/routing/*.json` |
| Router on SciFact | reranks 0%, 252 → 7 ms, nDCG +0.8 vs always-rerank | same |
| Router per-query signal | about the same as random routing (NFCorpus 33.32 vs 33.24, FiQA 43.18 vs 43.16) | same |
| CI gate | fails at −0.5 points; cutting depth 100 → 10 failed it (recall@100 −11.4); GitHub's Linux runner reproduced the baseline exactly | verified locally and in Actions |

---

## 30-second version

"I built a hybrid search engine that runs keyword search and embedding search together, merges them with weighted rank fusion, and lets a cross-encoder vote on the top results. On three BEIR datasets it's 1.4 nDCG points better than embedding search alone and 8.5 better than keyword search alone, with all tuning done on separate dev data. I also benchmarked 12 FAISS index types on 200,000 documents. HNSW is about 32 times faster than exact search with under half a point of quality loss, and binary quantization makes the index 32 times smaller with almost no loss if you rescore. There's a CI check that blocks any change that makes search worse."

## 2-minute version

"Keyword search and embedding search fail in opposite ways. Keyword search can't match 'heart attack' to 'myocardial infarction.' Embedding search is fuzzy on exact things like product codes and names. So I run both, and merge their rankings with reciprocal rank fusion. RRF works on rank positions instead of raw scores, so I don't have to make BM25 scores and cosine similarities comparable.

The first surprise was that equal-weight fusion made things *worse* on FiQA, because BM25 is so weak there that it dragged the embedding results down. So I tuned a weight for BM25. I did it on the BEIR dev and train splits, never the test set, and it came out at 0.1.

The second surprise was the reranker. The textbook design is: retrieve 100, let a cross-encoder re-sort them. On SciFact that made results worse with every MiniLM reranker and depth I tried, because they're trained on web queries, not scientific claims. So I made the cross-encoder one weighted *vote* in the fusion instead of the final word. Across three datasets, voting averaged 49.8 nDCG@10 and replacing averaged 47.7.

The cross-encoder is also 97% of query latency on CPU, about 250 milliseconds out of 257. So I built a router that only reranks when the top fused results are close together. It learned to switch reranking off entirely on SciFact, where it hurts: mean latency went from 252 to 7 milliseconds and quality went up. But I'll be straight about it: per query, the margin signal is no better than random. It tells you which queries are hard, not which ones the reranker will fix.

The second half is the FAISS benchmark. I tested 12 index configurations on a 200,000-document corpus and measured recall, single-thread latency, memory and actual search quality. The key finding is that recall against exact search isn't enough. HNSW and IVF had the same recall, but HNSW lost three times more nDCG, because its misses land on the hard queries. So I report two operating points per index, and the quality-based one is what I'd deploy.

Finally, a GitHub Actions gate re-runs the SciFact evaluation on every pull request and fails if nDCG or recall drops more than half a point. I tested that it actually fails by cutting the candidate depth."

---

## The architecture, narrated at a whiteboard

*(Draw left to right as you talk.)*

"On the left is the query. It goes down two paths at the same time.

**Top path: BM25.** I wrote it myself as a sparse matrix: one row per document, one column per word, and each cell already holds that word's BM25 contribution to that document. At query time I take the columns for the query's words and add them up. That's one sparse matrix-vector product, under a millisecond on these corpora. It returns the top 100.

**Bottom path: dense.** The query goes through bge-small, a 33-million-parameter embedding model that outputs a 384-number vector, normalized to length one. That vector goes into FAISS, which finds the 100 documents with the highest inner product, which equals cosine similarity because everything's normalized. For these corpus sizes I use exact search. The benchmark tells you what to switch to when the corpus grows.

**Middle: fusion.** Both lists of 100 go into weighted reciprocal rank fusion. Each document gets weight divided by 60 plus its rank, from each list it appears in. BM25's weight is 0.1 and dense is 1.0. I sort and get one list of up to 200.

**Then the router.** I look at the gap between the top two fused scores. A big gap means confident, so I return the list. That's about 7 milliseconds total on CPU.

**If the gap is small,** the top 30 go to the cross-encoder. It reads query and document together, which is accurate but costs a full transformer pass per pair, about 250 milliseconds for 30 on CPU. Its ranking goes back into the same fusion as a third vote with weight 0.5, and I return the top 10.

**Off to the side,** draw two boxes. One is the evaluation harness: three BEIR datasets with human labels, nDCG@10, paired significance tests, and a tuning script that only touches dev splits. The other is CI: every PR re-runs the SciFact evaluation on CPU and compares it to a baseline file in the repo. Half a point worse fails the build.

**And the FAISS benchmark** is its own experiment: same embeddings, 12 index types, 200K docs, measuring the speed, memory and quality tradeoff."

---

## Design decisions and what I rejected

| Decision | What I chose | Alternatives I rejected | Why |
|---|---|---|---|
| How to combine BM25 and dense | **Weighted RRF** (k = 60) | Linear combination of normalized scores; learned fusion model | BM25 scores are unbounded and their distribution shifts per query, so min-max normalization is fragile. RRF only needs ranks. Weighting fixed the one case where plain RRF failed (FiQA). A learned model needs training labels and adds a component to maintain for a small expected gain. |
| Role of the cross-encoder | **One weighted vote in RRF** | Replace the top-N order (standard); no reranker | Replacing lost 2.1 points on average vs voting and was below dense-only on two datasets. No reranker is the right call on SciFact, which the router handles. |
| Which cross-encoder | **ms-marco-MiniLM-L-6-v2** (22M) | MiniLM-L-12; bge-reranker-base (278M); mxbai-rerank-base | L-12 was no better on SciFact. bge-reranker-base was slightly better on SciFact as a voter (74.1 vs 73.1 in my equal-weight test) but about 4× slower on CPU (24 vs 110 pairs/s). mxbai ran at about 1 pair/s on CPU. For a CPU-served system the small one wins. |
| Embedding model | **bge-small-en-v1.5** (384-d) | all-MiniLM-L6-v2; bge-base (768-d); paid APIs | bge-small is much stronger than MiniLM at the same size. bge-base doubles memory and encoding time. Paid APIs break the zero-cost rule and send data to a third party. |
| Query instruction prefix | **Keep it** | Drop it | Dropping it helped SciFact test by 1.6 points, but on the dev splits it lost on NFCorpus and FiQA (mean 49.48 vs 49.35). I decided on dev, not test. |
| BM25 implementation | **My own, ~100 lines on scipy.sparse** | Pyserini/Lucene; bm25s; rank_bm25; Elasticsearch | Pyserini needs Java. rank_bm25 is slow. bm25s would work but hides the mechanics I need to explain. Elasticsearch is a server to run and pay for at scale. I validated mine against Anserini's published numbers. |
| Tuning protocol | **Grid search on train/dev splits, test scored once** | Tune on test; no tuning | Tuning on test inflates results and is the most common mistake in retrieval projects. No tuning would have kept the FiQA regression. |
| Serving index | **Flat (exact)** | HNSW, IVF | The served corpora are 4K–58K docs; exact search is a few ms and has no recall loss. The benchmark shows when to switch. |
| Router target | **Recover 90% of the reranker's gain** | Keep 97% of full-rerank nDCG | With a reranker gain under 1 point, "97% of nDCG" is satisfied by never reranking, which is a meaningless target. I caught that and changed it. |
| Router calibration | **Calibrate on one half of queries, report on the other** | Calibrate and report on all | Same-data calibration overfits the threshold. |
| Benchmark latency | **Single query, single thread, per query** | Batched throughput | Online search answers one query at a time. Batched search flatters Flat because BLAS amortizes it. |
| Benchmark operating point | **Two: recall ≥ 0.95, and within 0.5 nDCG** | Recall only | HNSW and IVF at the same recall differed 3× in nDCG loss. Recall alone would ship a hidden regression. |
| Index size | **Real serialized bytes** | Theoretical code size | Includes codebooks, IDs and graph links. That's why PQ48 is 30.7×, not 32×. |
| 200K benchmark corpus | **Quora sample that keeps every judged doc** | Random 200K sample; synthetic vectors (SIFT1M) | A random sample would drop most relevant docs and break nDCG. Synthetic vectors aren't my embeddings and have no relevance labels. |
| CI gate dataset | **SciFact, CPU, 0.5-point tolerance** | All datasets; GPU; statistical test | SciFact runs in about 2 minutes on CPU, fast enough to run on every PR. Free runners have no GPU. A fixed tolerance is simple and predictable. |
| macOS OpenMP conflict | **Symlink faiss's libomp to torch's + import-time check** | `KMP_DUPLICATE_LIB_OK=TRUE`; separate processes | The env var still segfaulted. Separate processes would add IPC to every query. The symlink gives one runtime and I verified it against brute force. |
| Dependency management | **uv + lockfile, CPU torch on Linux** | pip + requirements.txt | Exact reproducibility; CI doesn't download gigabytes of CUDA libraries it can't use. |
| API | **FastAPI with a factory + lifespan** | Flask; load per request | Typed validation in the signature, auto docs, models loaded once, testable with a fake searcher. |

---

## Hard questions

**1. Your full pipeline is only 1.4 points better than dense retrieval. Is that worth the complexity?**

It depends on where the 1.4 comes from. Most of it is BM25 rescuing queries with rare exact terms, at about 1 ms of extra latency. That part is cheap and I'd always keep it. The reranker's share is smaller and costs 250 ms on CPU, and that's exactly why I built the router and why it turns reranking off on SciFact. The gain is significant on NFCorpus and FiQA (p ≤ 0.002) but not on SciFact (p = 0.13). So my honest answer: hybrid retrieval is worth it, and reranking is worth it only where you've measured that it helps.

**2. Why not just use a bigger reranker and get a bigger number?**

I tested bge-reranker-base. As a voter it was about a point better on SciFact but about 4× slower on CPU. I'm optimizing for something that runs on free CPU hardware, so I took the small model. On a GPU budget I'd revisit it, and the code takes the model name as a flag.

**3. Your reranker made SciFact worse. Why ship a component that hurts?**

Because it doesn't hurt everywhere. It helps FiQA by 0.8 points (p = 0.007). The fix is to not use it where it hurts, which is what the router does: it learned a threshold of zero on SciFact and never reranks there. In production you'd calibrate per collection, or per query type if you had the labels.

**4. Why does an MS MARCO cross-encoder hurt on SciFact?**

Domain shift. MS MARCO is web search: short natural-language questions and web passages. SciFact queries are scientific claims, and the relevant documents are abstracts that support or refute them. The cross-encoder learned "does this passage answer this question," which isn't the same task. The bi-encoder (bge) was trained on a much broader mix, so it transfers better.

**5. How do I know you didn't tune on the test set?**

`hse tune` loads SciFact train, NFCorpus dev and FiQA dev through `data.load(name, split=...)`, which only keeps queries that appear in that split's qrels. The best weights are written to `results/tune.json`, and I set them in `Config` before running `hse eval` on test once. When removing the query prefix looked better on the SciFact test set, I checked dev, it lost, and I kept the prefix.

**6. Your best BM25 weight, 0.1, is the smallest value in your grid. Doesn't that mean the true optimum is outside it?**

Possibly, yes. That's a fair catch. The next step is extending the grid down (0.05, 0.02, 0). For what it's worth, 0 is just "no BM25," which is dense-only, and that loses on all three dev sets, so the optimum is somewhere between 0 and 0.2.

**7. BM25 at weight 0.1 barely counts. Isn't it basically dense-only with extra steps?**

No. With RRF, a weight of 0.1 means BM25's top document adds 0.1/61 ≈ 0.0016, which is enough to break ties and reorder documents that dense ranks close together. Hybrid vs dense is significant on SciFact (p = 0.001) and NFCorpus (p = 0.001). And for queries with an exact identifier dense misses entirely, BM25 still gets the document into the candidate pool for the reranker to see.

**8. Your router is no better than random per query. Why is it in the project?**

Two reasons. First, at the dataset level it makes the right decision: it turns the reranker off where it hurts, which cut SciFact latency from 252 to 7 ms with better quality. Second, it's a measured negative result I can explain: the margin predicts how hard a query is (low-margin SciFact queries score 66 vs 78 for high-margin), but not whether the cross-encoder will fix it. The next version would predict reranker *gain* from several features (margin, BM25–dense overlap, query length, top-1 dense score) with a small classifier trained on the calibration half.

**9. Where's the bottleneck at query time?**

The cross-encoder, by far. On CPU: query encoding about 7 ms warm, FAISS under 2 ms, BM25 under 2 ms, fusion under 0.1 ms, and reranking 30 pairs about 250 ms. Everything except the reranker is under 10 ms. The fixes, in order: route fewer queries to it, rerank fewer candidates, run it as int8 ONNX on CPU, or move it to a GPU.

**10. How would this scale to 100 million documents?**

Each part breaks differently:
- **Dense vectors:** 100M × 1,536 bytes is 154 GB as float32. That's past one machine's RAM, so I'd use IVF-PQ or HNSW over quantized vectors, sharded across machines, with rescoring from SSD. My benchmark shows binary + rescore keeps quality.
- **Exact search** would take about 2 seconds per query on one thread (linear scaling from 4 ms at 200K), so approximate search becomes mandatory.
- **BM25:** my implementation keeps the whole matrix in memory and has to rebuild to add documents. At that scale I'd use Lucene through OpenSearch or Elasticsearch, which have segment-based incremental indexing and sharding.
- **Reranker:** unchanged, since it only sees 30 docs per query whatever the corpus size.
- **Evaluation:** the same harness works; it just needs a bigger labeled set.

**11. HNSW or IVF: when would you pick which?**

HNSW when you have the memory and want the lowest latency: 32× faster than exact within 0.5 nDCG, but it needs about 18% more memory than the raw vectors for graph links, and it's harder to delete from. IVF when memory is tight or the data changes a lot: it's 19× faster at the quality point, nearly no memory overhead, and it combines with PQ for compression. But it needs training, and its centroids go stale if the data distribution drifts.

**12. Why did HNSW lose 1 nDCG point at 0.968 recall when IVF lost 0.3 at 0.966?**

Average recall hides *which* queries miss. IVF misses tend to be spread out: a true neighbor sits in a cluster just past the `nprobe` cutoff. HNSW misses cluster on hard queries, where the greedy graph walk gets stuck in a local neighborhood, and those are the same queries whose relevant documents are unusual. So HNSW loses its misses exactly where they hurt. That's why I added the quality-based operating point.

**13. SQ8 was 3× slower than exact search. Isn't that a bug?**

No. Single-threaded, flat float32 search is a tight SIMD inner-product loop that the CPU is extremely good at. SQ8 has to decode each 8-bit code back to a float before computing, and that extra work per element costs more than the memory bandwidth it saves at this size. SQ8 is a *memory* optimization (4× smaller), not a speed one. It pays off when the float vectors don't fit in RAM or cache.

**14. You say 32× smaller. PQ is 30.7×. Which is it?**

Binary quantization is exactly 32×: 48 bytes per vector, no overhead. PQ48 is 30.7× because I count the real serialized size, including the codebooks (384 dims × 256 centroids × 4 bytes ≈ 393 KB, which adds about 2 bytes per vector at 200K). I report the real number on purpose.

**15. Rescoring needs the full vectors. So what did compression actually save?**

RAM. The compressed index (9.6 MB for 200K docs) lives in memory and does the search. The full vectors (307 MB) stay in a memory-mapped file on disk, and only 100 rows per query are read. RAM is the expensive, limited resource; disk isn't. The honest framing is "32× less RAM," not "32× less storage."

**16. Isn't single-thread latency unrealistic? Servers have many cores.**

The cores go to handling many queries concurrently, one per thread, not to splitting one query. Single-thread per-query latency is the fair comparison between index types. I'd also report batched throughput for an offline workload, and that's a one-line change.

**17. How do you add or delete documents?**

Today I rebuild. That's the main production gap.
- **BM25:** adding documents changes every term's IDF and the average document length, so a correct update touches every weight. Lucene handles this by computing IDF at query time and merging segments.
- **FAISS Flat and HNSW** support `add`. Deletion in HNSW is awkward (mark deleted, rebuild periodically).
- **IVF** needs periodic retraining if the new data shifts.

The next version would compute BM25 at query time from term frequencies, so that adding documents doesn't rebuild every weight.

**18. What happens with bad input?**

- **Empty query or `k` out of range:** the API rejects it with a 422 before any search runs.
- **Query of only stopwords or unknown words** ("the of and"): BM25 returns nothing and dense still returns results, so fusion degrades to dense-only. There's a test for the BM25 side.
- **Non-English text:** BM25 tokenizes and stems it wrongly, and bge-small is English-only, so quality drops sharply and nothing tells the user. I'd add language detection.
- **Very long documents:** truncated at 512 tokens by both models, so content past that is invisible to dense search and the reranker (BM25 still sees all of it).
- **Duplicate documents:** both copies get retrieved and take two slots. Near-duplicate removal at index time would fix it.
- **Very long queries:** capped at 1,000 characters at the API.

**19. What breaks under load, say 100 concurrent requests?**

The cross-encoder. Each reranked query costs about 250 ms of CPU, and PyTorch already uses several cores per inference, so concurrent requests fight over the same cores and latency climbs roughly linearly. FastAPI runs the sync handler in a threadpool, so the server stays up, but p95 would explode. Fixes: cap concurrency, batch reranker calls across requests (dynamic batching), run the reranker as its own service on a GPU, and use the router to shed reranking under load (raise the threshold when the queue is long). Retrieval itself (under 10 ms) would handle the load fine.

**20. The CI gate only checks SciFact. Couldn't a change break FiQA and pass?**

Yes. It's a tradeoff with CI time. SciFact takes about 2 minutes on my 14-core laptop CPU, but about 19 minutes on GitHub's free 4-vCPU runner with a cold cache (the encoding is cached after that). FiQA would add far more. The gate catches regressions in shared code (tokenizer, fusion, model loading, depth), which hit every dataset. Dataset-specific regressions are caught by running `make eval` before a release. With a self-hosted or GPU runner I'd gate all three.

**21. Why a 0.5-point tolerance? Won't it be flaky?**

The evaluation is deterministic: I ran it twice on my Mac and got +0.00 on every metric, and GitHub's Linux runner reproduced the Mac baseline exactly, +0.00 on all six gated numbers. Across other hardware, floating-point differences in BLAS could still flip a near-tie and move nDCG by hundredths. 0.5 points absorbs that and still catches real regressions: cutting depth from 100 to 10 dropped hybrid nDCG by 0.94 and recall@100 by 5.9. It's an absolute threshold, not a statistical test. With 300 queries, a significance test would miss real regressions under about 1 point.

**22. You ran a lot of t-tests. Multiple comparisons?**

Fair. I report five p-values per dataset without correction. The headline comparison is pre-specified: full pipeline vs dense. Its p-values on NFCorpus (0.002) and FiQA (0.0007) survive a Bonferroni correction for 5 tests (threshold 0.01). SciFact's 0.13 doesn't, and I say that. Per-query nDCG also isn't normal, so a randomization test would be better. That's on the next-version list.

**23. Why write BM25 yourself instead of using Lucene?**

Three reasons. I wanted to understand and explain every line. It's about 100 lines and fast enough at this scale (under 2 ms/query on 200K docs). And it's verifiable: it matches Anserini's published BEIR numbers to within 0.2 points on all three datasets. At production scale I'd use Lucene, and I say so in question 10.

**24. Patching site-packages with a symlink is a hack. Why is that acceptable?**

It's the least-bad option among the ones I tested. The officially suggested env var segfaulted. Import order doesn't help. Separate processes would add IPC to every query. The symlink makes both libraries use one runtime, which is what OpenMP's own error message says is correct. It's guarded: the code checks for the conflict at import and tells you the exact command, instead of segfaulting. And it only applies on macOS dev machines; Linux servers and CI don't have the conflict. The proper fix is upstream (FAISS's macOS wheel linking against a shared libomp).

**25. Your 200K Quora corpus keeps every relevant doc. Doesn't that make it easier?**

Yes. There are fewer distractors than in the full 523K corpus, so absolute nDCG (93.2) is higher than it would be on full Quora. But the benchmark compares index types against exact search *on the same corpus*, and that relative comparison is what the numbers claim. Recall against exact doesn't depend on the labels at all.

**26. Why nDCG@10? For RAG, isn't recall what matters?**

Both matter, and I report both. nDCG@10 is the BEIR standard, so my numbers compare with published ones. For RAG, recall@k at the number of chunks you pass to the LLM matters more. My recall@100 shows whether the right document reaches the candidate pool at all. Fusion didn't change recall@100 here, because with BM25 at 0.1 the top 100 is mostly dense's top 100. The gains are in ordering.

**27. Could bge-small have seen these datasets during training?**

Possibly. BGE's training mix is large and includes MS MARCO and other public retrieval data, and I can't rule out overlap with BEIR train splits. That's a reason not to compare my absolute numbers too hard against other papers. My comparisons are relative: the same embedding model sits under every system I compare.

**28. In production you have no relevance labels. How would you know if the reranker helps?**

Online signals and cheap labels. Interleaving experiments (mix results from two rankers and see which ones users click) are more sensitive than A/B tests. Click-through and dwell time give implicit labels. A small human-labeled set from real queries gives a periodic check. An LLM judge rating relevance on a sample is cheap and correlates reasonably with humans for this kind of task. The gate would then run on a frozen set of real queries instead of SciFact.

**29. Why not use a vector database like Qdrant or Pinecone?**

Pinecone is paid and sends data off-machine, which breaks the zero-cost constraint. Qdrant and Weaviate are free to self-host and would give persistence, filtering and updates. But for an experiment about index tradeoffs I wanted FAISS's direct control over the exact index type and parameters. Most vector databases use HNSW internally, so my HNSW numbers carry over.

**30. What would you do differently if you started over?**

- Start the CI gate on day one. I built it after the baselines, and it would have caught my early experiments drifting.
- Include a stronger reranker in the tuning grid from the start, and tune the rerank depth too (I fixed it at 30).
- Design the router around predicting reranker *gain*, not query difficulty. I'd have seen the problem in the first analysis.
- Use a randomization test instead of t-tests from the start.

**31. Is the latency comparison in the router fair? The reranker numbers are on CPU.**

They're CPU on purpose: free deployment targets (HF Spaces, a small VM) are CPU-only, and that's where the routing tradeoff bites. On the M4's GPU (MPS) the reranker is about 3× faster, so the savings shrink proportionally, but the ratio between retrieval (~7 ms) and reranking stays large.

**32. Your evaluation has 300–648 queries per dataset. Is that enough?**

For differences of about 1 point, it's borderline, which is exactly what the p-values show: significant on FiQA (648 queries) and NFCorpus, not on SciFact (300). I don't claim more than the tests support. More datasets is the cheapest way to strengthen the conclusions.

---

## Known limitations, and what the next version adds

**Limitations, stated plainly:**

1. **The quality gain over dense retrieval is real but modest:** +1.4 nDCG@10 on average, significant on two of three datasets.
2. **The cross-encoder is out of domain** for SciFact and hurts there; it's a small model chosen for CPU speed.
3. **The router works per dataset, not per query.** Its per-query decisions are about as good as random.
4. **BM25 needs a full rebuild to add documents**, and everything lives in memory on one machine.
5. **Models truncate at 512 tokens** and are English-only.
6. **The CI gate covers one dataset.**
7. **Fusion weights are global**, not per domain or per query.

**Next version:**

1. **Gain-predicting router:** a small classifier on margin, BM25–dense overlap, query length and top score, trained to predict whether reranking will help each query.
2. **Better reranking per dollar:** int8 ONNX cross-encoder for CPU, rerank depth added to the tuning grid, and an optional bigger reranker on GPU.
3. **Query-dependent fusion weights**, e.g. more BM25 weight when the query contains IDs, numbers or rare terms.
4. **Incremental indexing:** query-time IDF for BM25, and FAISS add/remove with periodic IVF retraining.
5. **Stronger evaluation:** all 15 public BEIR datasets, randomization tests with multiple-comparison correction, and recall@k at RAG-sized k.
6. **Production API:** dynamic batching for the reranker, load shedding through the router, metrics (p50/p95 per stage), auth and rate limits.
7. **Chunking** for long documents, so content past 512 tokens is searchable.

---

## Resume bullets that match the measured results

These are the bullets I can defend line by line with the files in `results/`:

- Built hybrid retrieval in Python (from-scratch BM25 on scipy.sparse, bge-small embeddings in FAISS, weighted reciprocal rank fusion, cross-encoder as a fusion vote); **+1.4 nDCG@10 over dense and +8.5 over BM25** averaged across 3 BEIR datasets, with all weights tuned on dev splits only.
- Benchmarked **12 FAISS configurations on a 200K-document corpus**: **32× smaller index** via binary quantization with **0.01-point nDCG@10 loss** using top-100 rescoring; HNSW **32× faster than exact search within 0.5 nDCG points** (51× at 0.97 recall@10).
- Added a **GitHub Actions quality gate** that re-evaluates retrieval on every PR and fails on >0.5-point nDCG/recall regressions; built a calibrated margin router that skips the cross-encoder where it doesn't help (**mean CPU latency 252 → 7 ms** on SciFact with higher nDCG). Served it through FastAPI.
