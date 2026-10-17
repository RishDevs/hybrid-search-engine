# Code walkthrough

This goes through the project file by file and block by block. Every significant block gets two explanations:

- **Plain:** the idea with a concrete analogy, no jargon.
- **Technical:** the exact mechanics, data shapes, and what breaks if it's written differently.

Section 1 traces the whole system end to end, so you can hold it in your head before the details. Section 2 lists the blocks worth knowing cold. Section 3 is the file-by-file walkthrough.

---

## 1. The whole system, end to end

There are seven entry points. All of them go through `cli.py` (the `hse` command), except the API, which goes through `api.py`.

### Path A: building the indexes (`hse index scifact`)

```
cli.cmd_index
 └─ data.load("scifact")                      download zip if missing → parse corpus.jsonl, queries.jsonl, qrels/test.tsv
     returns Dataset(doc_ids[N], texts[N], queries{qid: text}, qrels{qid: {doc_id: grade}})
 └─ pipeline.build_artifacts(ds, cfg)
     ├─ BM25(k1=0.9, b=0.4).fit(texts)        → sparse weight matrix (N × V), saved as artifacts/scifact/bm25-....npz
     ├─ Encoder("BAAI/bge-small-en-v1.5")
     ├─ embed.cached_doc_vectors              → float32 (N × 384), L2-normalized, saved as doc_vectors.npy
     └─ DenseIndex.build(vectors, "Flat")     → FAISS IndexFlatIP, saved as flat.faiss
```

Everything is cached on disk. A second run loads the files instead of rebuilding them.

### Path B: answering one query (`hse search "..."` or `GET /search?q=...`)

```
HybridSearcher.open(dataset)                  load Dataset + artifacts + both models + results/router.json (once)
HybridSearcher.search(query, k=10, rerank="auto")
 1. encoder.encode_queries([query])           "Represent this sentence...: " + query → (1 × 384) float32
 2. index.search(qvec, 100)                   FAISS inner product → top-100 doc indices + cosine scores (dense ranking)
 3. bm25.search(query, 100)                   analyze → look up term columns → sum weights → top-100 + scores (keyword ranking)
 4. _fuse(bm_ids, dense_ids)                  weighted RRF: 0.3/(60+rank_bm25) + 1.0/(60+rank_dense) → fused ranking (≤200 docs)
 5. features(query, bm, dense, fused)         8 numbers: fused margin, dense top-1 and gaps, BM25/dense overlap@10, BM25 gap, hits, query length
 6. router.predict(features)                  linear model → predicted nDCG gain from reranking
 7. predicted ≥ router.threshold?
    ├─ no  → return the fused top-k
    └─ yes → reranker.rerank(query, fused[:30])     gte cross-encoder scores 30 (query, doc) pairs → its own ranking
             _fuse(bm_ids, dense_ids, ce_ids)       RRF over three lists with weights (4.0, 0.3, 1.0)
             return that top-k
 8. return SearchResult(hits, reranked, predicted_gain, timings_ms)
```

On CPU, steps 1–7 take about 8 ms in total and reranking 30 pairs takes about 1,600 ms. That ratio is why the router exists.

### Path C: evaluating quality (`hse eval`)

```
experiments.evaluate(dataset)
 └─ HybridSearcher.open(..., cache_scores=True)
 └─ run_batch(all test queries)              same logic as Path B, but batched:
     ├─ encode all queries at once, one FAISS call for all queries
     ├─ BM25 per query
     ├─ fuse per query                         → "hybrid", plus the 8 router features per query
     └─ cross-encoder over every (query, top-30) pair in one big batch (cached scores reused)
         ├─ "hybrid+ce-only"   cross-encoder order replaces the top 30
         └─ "hybrid+rerank"    cross-encoder votes in RRF (the real pipeline)
 └─ metrics.to_run → metrics.summarize       nDCG@10, recall@100, MRR@10 per system
 └─ metrics.paired_test                      paired t-test on per-query nDCG
 └─ writes results/eval/<dataset>.json and .per_query.json
```

### Path D: choosing the reranker and weights (`hse tune`)

Loads the **train/dev** splits (SciFact train, NFCorpus dev, FiQA dev) and runs BM25 and dense once. For every query it builds one candidate pool: the union of the fused top 30 under every BM25 weight in the grid. Each candidate reranker (MiniLM-L6, bge-reranker-base, gte-reranker-modernbert) scores each pool once, with scores going into the SQLite cache. Then it grid-searches rerank depth {10, 20, 30} × BM25 weight {0.1 … 0.5} × reranker weight {0.5 … 10} using lookups only, and writes every combination to `results/tune.json`. The winner (gte, depth 30, 0.3, 4.0) is the default in `Config`. Test queries are never touched.

### Path E: FAISS benchmark (`hse bench`)

Loads Quora, subsamples it to 200K docs (keeping every labeled doc), and encodes everything. Exact search (`Flat`) gives the ground truth. For each of the 12 index configs, it builds the index and sweeps its knob. At each setting it measures recall@10 against exact, single-thread latency, and nDCG@10. It then picks two operating points: the fastest setting with recall ≥ 0.95, and the fastest within 0.5 nDCG points. For compressed indexes it also measures rescoring. Results go to `results/bench/`.

### Path F: routing (`hse route`)

Runs the full pipeline (with cached reranker scores) on the same dev/train queries and records each query's 8 features and its nDCG without and with reranking. `Router.train` fits a ridge regression from features to gain, picks the threshold from cross-validated predictions so that 80% of the gain is recovered, refits on all dev queries, and saves `results/router.json`. Only then does it load the **test** per-query file from Path C, apply the frozen router, and compare against reranking always, never, and at random with the same share. Finally it times 100 test queries per dataset end to end on CPU in `never`, `always` and `auto` modes.

### Path G: CI gate (`hse gate`, run by GitHub Actions)

Runs Path C on SciFact on CPU and compares dense, hybrid and hybrid+rerank against `results/baseline.json`. If nDCG@10 or recall@100 dropped by more than 0.5 points, it exits with code 1, which fails the GitHub check. Model weights and reranker scores are cached between CI runs.

### How the files depend on each other

```
cli ──▶ experiments ──▶ pipeline ──▶ bm25, dense, embed, fusion, rerank, router, data, config
 │           └────────▶ metrics
 ├──▶ gate ──▶ experiments
 ├──▶ report
 └──▶ api ──▶ pipeline
dense ──▶ openmp   (checked before faiss is imported)
```

---

## 2. The blocks to know cold

If you only have an hour, learn these. They carry the system, and they're what interviewers will poke at.

| # | Block | Why it matters |
|---|---|---|
| 1 | `BM25.fit` and `BM25.score` (bm25.py) | The whole keyword side. Precomputing BM25 weights into a sparse matrix is what makes it fast. |
| 2 | `rrf` (fusion.py) | Combines everything. Weighted RRF is the main design decision of the project. |
| 3 | `HybridSearcher.search` and `_fuse` (pipeline.py) | The actual query path, including the reranker-as-a-voter decision. |
| 4 | `DenseIndex.build` / `search` (dense.py) | Every FAISS index type goes through it: training, binary quantization, metric choice. |
| 5 | `ndcg_at_k` and `to_run` (metrics.py) | Every number in the project is computed here. If this is wrong, everything is wrong. |
| 6 | `query_features`, `Router.train`, `pick_threshold` (router.py) | Decides when the 1.6-second model runs; trained on dev, measured on test. |
| 7 | `tune` and the test/dev separation (experiments.py) | Why the results can be trusted; it picked the reranker that made the biggest difference. |
| 8 | `run_gate` (gate.py) + `ci.yml` | How quality is enforced. |
| 9 | `ensure_single_runtime` / `fix` (openmp.py) | The hardest debugging problem in the project. |

---

## 3. File by file

### 3.1 `config.py`

**What it is:** one place for every setting and every folder path.

```python
HOME = Path(os.environ.get("HSE_HOME", Path.cwd()))
DATA_DIR = HOME / "data"
ARTIFACTS_DIR = HOME / "artifacts"
RESULTS_DIR = HOME / "results"
CACHE_DIR = HOME / "cache"
ROUTER_FILE = RESULTS_DIR / "router.json"
```

- **Plain:** It's like deciding where the pantry, fridge and recipe book live before you start cooking. Everyone knows where to look.
- **Technical:** All paths are relative to `HSE_HOME`, which defaults to the current directory. `cache/` holds reranker scores; the trained router lives in `results/` so it's committed with the numbers it produced. That lets tests or CI point the project at a temp folder by setting one environment variable. If the paths were hardcoded relative to the package's install location, `pip install` (non-editable) would try to write datasets into `site-packages`.

```python
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
```

- **Plain:** the settings panel. Every dial in the system is here, with the value I chose.
- **Technical:**
  - `k1=0.9, b=0.4` are the Anserini BM25 defaults used for BEIR baselines. `k1` controls how fast repeated words stop adding score. `b` controls how much long documents are penalized.
  - `depth=100` means each retriever returns 100 candidates.
  - `rrf_k=60` is the constant from the original RRF paper (Cormack et al., 2009).
  - The reranker, `w_bm25=0.3`, `w_rerank=4.0` and `rerank_depth=30` came out of `hse tune` on dev splits. The cross-encoder sees only the top 30 fused docs, and its vote counts four times as much as dense search's.
  - `device=None` means "pick automatically."
  - Because it's a dataclass, `to_dict()` writes the exact config into every result file, so any number can be traced back to the settings that produced it.

```python
def slug(name): ...
def artifact_dir(dataset, cfg):
    return ARTIFACTS_DIR / dataset / slug(cfg.dense_model)
```

- **Technical:** Artifacts are stored per dataset *and* per embedding model, e.g. `artifacts/scifact/baai-bge-small-en-v1-5/`. Switching models can't silently reuse vectors from a different model, which would give garbage results with no error.

### 3.2 `data.py`

**What it is:** downloads BEIR datasets and loads them into one simple structure.

```python
@dataclass
class Dataset:
    name: str
    doc_ids: list      # ["4983", "5836", ...]       length N
    texts: list        # ["Title. Body text", ...]   length N, same order
    queries: dict      # {"1": "0-dimensional biomaterials show ..."}
    qrels: dict        # {"1": {"31715818": 1}}       query → {relevant doc: grade}
```

- **Plain:** a library catalog. A list of books (in a fixed order), a list of questions, and an answer key saying which books answer which question.
- **Technical:** Everything downstream refers to documents by **position** (an integer 0..N−1), not by string ID. FAISS and the BM25 matrix only understand integer row numbers. `doc_ids[i]` converts back at the very end. `doc_ids` and `texts` must stay aligned; a single off-by-one would scramble every result.

```python
def parse_name(name):
    m = re.fullmatch(r"(.+)-(\d+)k", name)
```

- **Technical:** `"quora-200k"` means "Quora, subsampled to 200,000 docs." The regex only matches a trailing `-<digits>k`, so a dataset with a hyphen in its real name (like `trec-covid`) isn't misread as a size. My first version split on the first hyphen and would have broken on exactly that.

```python
def download(name, root=DATA_DIR):
    ...urllib.request.urlopen(BEIR_URL.format(name))... zipfile.extractall
```

- **Technical:** Plain `urllib`, so there's no extra dependency. It's idempotent: if `corpus.jsonl` already exists, it returns immediately. The zip is deleted after extraction.

```python
def load(name, split="test", ...):
    qrels = read_qrels(folder / "qrels" / f"{split}.tsv")
    queries = {...only queries that appear in qrels...}
    texts.append(f"{title}. {body}" if title else body)
```

- **Plain:** It only keeps the questions that have an answer key for this split, and it glues each document's title onto its body.
- **Technical:** BEIR's `queries.jsonl` holds queries for *all* splits. Filtering by the split's qrels is what keeps dev and test apart. `hse tune` calls `load(name, split="dev")` and gets a completely different set of queries from evaluation. Prepending the title follows the BEIR convention; titles carry a lot of signal in SciFact and NFCorpus.

```python
def subsample(ds, size, seed=13):
    judged = {d for rel in ds.qrels.values() for d in rel}
    keep = [i for i, d in enumerate(ds.doc_ids) if d in judged]
    rest = ...shuffled other docs...
    chosen = sorted(keep + rest[: size - len(keep)])
```

- **Plain:** To make a smaller library, keep every book that's in the answer key, then fill the remaining shelf space with random other books.
- **Technical:** Quora has 523K docs. I sample 200K for the benchmark but keep all 15,675 docs that appear in the test qrels, so every query still has its answers and nDCG stays meaningful. If I sampled purely at random, about 60% of the relevant docs would vanish and nDCG would collapse for reasons unrelated to the index. The tradeoff, which I state openly, is that the corpus has fewer distractors than the full one, so absolute nDCG (93.2) is higher than on full Quora. `sorted` keeps the original document order, and the fixed seed makes the sample identical on every machine.

### 3.3 `bm25.py`: keyword search from scratch

```python
LUCENE_STOPWORDS = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then there these they this to was will with".split())
```

- **Plain:** words so common they don't help find anything, like "the" and "is." We throw them away.
- **Technical:** This is exactly Lucene's default English stop set (33 words). I picked it so my BM25 can be compared with the published Lucene/Anserini numbers. A larger stoplist would change scores and break that comparison.

```python
class Analyzer:
    def __call__(self, text):
        words = [w for w in _word.findall(text.lower()) if w not in LUCENE_STOPWORDS]
        return self.stemmer.stemWords(words)
```

- **Plain:** It chops text into words, lowercases them, drops filler words, and cuts each word down to its root, so "running," "runs" and "run" all become "run."
- **Technical:** The regex `\w+` splits on anything that isn't a letter, digit or underscore. PyStemmer's Snowball English (Porter2) stemmer is a C extension, so it's fast in batch. The same analyzer runs on documents at index time and on queries at search time. If they differed, query terms wouldn't match index terms and recall would crash with no error.

```python
def fit(self, texts):
    for i, text in enumerate(texts):
        for term, tf in Counter(self.analyzer(text)).items():
            cols.append(self.vocab.setdefault(term, len(self.vocab)))
            rows.append(i); counts.append(tf)
    tf = sparse.csr_matrix((counts, (rows, cols)), shape=(n, v))
```

- **Plain:** Build a giant table with one row per document and one column per word in the whole collection. Each cell says how many times that word appears in that document. Almost every cell is zero, so we only write down the non-zero ones.
- **Technical:**
  - It builds a COO triplet list and converts it to CSR, giving an (N × V) float32 matrix. `vocab.setdefault(term, len(vocab))` assigns column IDs in order of first appearance.
  - For FiQA, the matrix is 57,638 docs × 59,966 terms, but only 3.8 million cells are non-zero. Dense, it would be 3.5 billion cells (about 14 GB). Sparse, it's about 30 MB.

```python
    doc_len = np.asarray(tf.sum(axis=1)).ravel()
    avg_len = max(doc_len.mean(), 1e-9)
    df = np.bincount(tf.indices, minlength=v)
    idf = np.log1p((n - df + 0.5) / (df + 0.5))
```

- **Plain:** For each word, count how many documents contain it. Rare words get a big multiplier and common words a small one. Also measure each document's length.
- **Technical:**
  - `tf.indices` holds the column index of every non-zero entry, so `bincount` over it is the document frequency of every term in one vectorized call.
  - IDF is the Lucene variant, `ln(1 + (N − df + 0.5)/(df + 0.5))`. The `1 +` keeps it positive even for terms in more than half the documents. Classic Robertson IDF goes negative there, which would make a common query word *lower* a document's score.
  - `max(…, 1e-9)` guards against dividing by zero on an empty corpus.

```python
    row_of = np.repeat(np.arange(n), np.diff(tf.indptr))
    length_norm = self.k1 * (1 - self.b + self.b * doc_len / avg_len)
    tf.data = idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + length_norm[row_of])
    self.weights = tf.tocsc()
```

- **Plain:** Work out each word's final score in each document ahead of time, so answering a query becomes just adding up a few numbers.
- **Technical:** This is the core of the BM25 implementation. Per entry, `w(d,t) = idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))`.
  - `np.diff(indptr)` gives the number of non-zeros per row, and `np.repeat` expands that into the row index of every non-zero entry, so all of this is vectorized over the non-zeros, with no Python loop.
  - Precomputing the weights at index time means query time is just a sum of matrix columns.
  - Converting to **CSC** (column-major) matters: a query needs a few *columns* (its terms), and CSC slices columns in O(nnz in those columns). Slicing columns from CSR would scan the whole matrix every query.
  - The tradeoff: changing `k1` or `b` needs a re-index, which is why the file name includes them.

```python
def score(self, query):
    terms = Counter(t for t in self.analyzer(query) if t in self.vocab)
    cols = [self.vocab[t] for t in terms]
    qtf = np.fromiter(terms.values(), dtype=np.float32)
    return np.asarray(self.weights[:, cols] @ qtf).ravel()
```

- **Plain:** Look up the columns for the query's words and add them up for every document. Words that never appear in the collection are skipped.
- **Technical:** `weights[:, cols]` is an (N × q) sparse slice, and multiplying it by the query term-count vector gives an (N,) dense score array. Repeated query words count multiple times (`qtf`), as in Lucene. Unknown terms are filtered out because they have no column. If there are no known terms, it returns all zeros.

```python
def search(self, query, k=100):
    k = min(k, int(np.count_nonzero(scores)))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
```

- **Plain:** Find the 100 highest scores without sorting all 200,000 documents.
- **Technical:** `argpartition` is O(N) and only guarantees that the top k are in the first k slots, so I then sort just those k (O(k log k)). A full `argsort` would be O(N log N) per query. Documents with score 0 are dropped, since sharing zero words with the query isn't a match. `kind="stable"` breaks ties deterministically, which matters for reproducible numbers.

```python
def save(self, path): np.savez_compressed(path, data=..., indices=..., indptr=..., shape=..., terms=..., params=...)
```

- **Technical:** It stores the three raw CSC arrays plus the vocabulary in column order and `(k1, b)`. `load` rebuilds the matrix with no recomputation. Terms are stored as an object array, so loading needs `allow_pickle=True`. That's fine for files this project writes itself, but you shouldn't load untrusted `.npz` files this way.

**How I know it's right:** the BM25 nDCG@10 matches Anserini's published BEIR numbers: 68.0 vs 67.9 (SciFact), 32.1 vs 32.2 (NFCorpus), 23.8 vs 23.6 (FiQA). `tests/test_bm25.py` also checks one score against the formula computed by hand.

### 3.4 `embed.py`: turning text into vectors

```python
def pick_device(requested=None):
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"
```

- **Technical:** It uses the fastest available hardware: NVIDIA GPU, then Apple GPU, then CPU. `torch` is imported inside the function so that importing the module doesn't load PyTorch. That keeps `hse fix-openmp` and `hse --help` fast and free of torch.

```python
class Encoder:
    def _encode(self, texts, progress=False):
        vecs = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True)
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def encode_queries(self, texts):
        return self._encode([self.query_prefix + t for t in texts])
```

- **Plain:** A small neural network reads a sentence and writes down 384 numbers that describe what it means. Sentences about the same thing get similar numbers.
- **Technical:**
  - `bge-small-en-v1.5` is a 33M-parameter BERT-style bi-encoder that outputs 384-dim vectors and reads up to 512 tokens.
  - `normalize_embeddings=True` makes every vector unit length, so inner product equals cosine similarity. That's why FAISS uses `METRIC_INNER_PRODUCT` everywhere. Without normalization, longer or "louder" vectors would win regardless of direction.
  - The query prefix ("Represent this sentence for searching relevant passages: ") is how bge was trained for asymmetric search. Documents get no prefix.
  - I tested removing it. It helped SciFact test by 1.6 points but lost on NFCorpus and FiQA dev, so it stayed (decided on dev, not test).
  - `ascontiguousarray(float32)` is required because FAISS needs C-contiguous float32 and will reject or silently copy anything else.

```python
def corpus_fingerprint(doc_ids, model_name): sha1(model_name + every doc_id)[:16]

def cached_doc_vectors(encoder, ds, folder):
    if fingerprint matches: return np.load(vec_path, mmap_mode="r")
    vecs = encoder.encode_docs(ds.texts, progress=True)
    np.save(...); write fingerprint json
    return np.load(vec_path, mmap_mode="r")
```

- **Plain:** Encoding 200,000 documents takes minutes, so we save the result and reuse it. A fingerprint confirms the saved file belongs to the same documents and model.
- **Technical:**
  - The fingerprint hashes the model name and the full ordered list of doc IDs, so a different subsample or model forces a re-encode. Without it, the 200K sample and a 100K sample would share a cache and produce nonsense.
  - It doesn't hash the encoding code itself. In CI that's covered by the cache key including `embed.py`'s hash.
  - `mmap_mode="r"` maps the file into memory instead of reading it all, so the OS loads pages on demand and shares them between processes. 200K × 384 × 4 bytes = 307 MB.

### 3.5 `openmp.py`: the macOS crash fix

- **Plain:** Two libraries each brought their own copy of the same engine part, and when both ran at once the program crashed. The fix is to make one library borrow the other's part, so there's only one.
- **Technical:**
  - On macOS, the `faiss-cpu` wheel ships `faiss/.dylibs/libomp.dylib` and PyTorch ships `torch/lib/libomp.dylib`. They're different builds of the LLVM OpenMP runtime.
  - When both load into one process, OpenMP aborts with `OMP: Error #15`. With the "workaround" env var `KMP_DUPLICATE_LIB_OK=TRUE` set, FAISS **segfaulted** during parallel search instead. I reproduced both.
  - Importing in either order doesn't help.

```python
def conflict():
    ours, theirs = faiss/.dylibs/libomp.dylib, torch/lib/libomp.dylib
    return sys.platform == "darwin" and ours.resolve() != theirs.resolve()

def fix():
    ours.rename(backup)                                  # keep the original
    ours.symlink_to(os.path.relpath(theirs, ours.parent))
```

- **Technical:**
  - `find_spec` locates both packages *without importing them*. Importing them would load the libraries we're about to change.
  - The fix replaces FAISS's copy with a relative symlink to torch's, so the dynamic loader opens one file and there's one runtime. I verified it by checking that FAISS flat search matches NumPy brute force exactly on 50K random vectors with torch loaded.
  - `ensure_single_runtime()` runs at the top of `dense.py`, *before* `import faiss`. If the conflict exists, it raises a clear error that says which command to run, instead of crashing later inside C code with a stack trace that points nowhere useful.
  - On Linux the wheels don't conflict, so all of this is a no-op.
  - The tradeoff: it patches the virtualenv. Recreating the venv means running `hse fix-openmp` again, which `make setup` does.

### 3.6 `dense.py`: the FAISS wrapper

```python
from .openmp import ensure_single_runtime
ensure_single_runtime()
import faiss  # noqa: E402
```

- **Technical:** The check must run before `import faiss`, so the import sits below it. The `noqa` tells the linter this out-of-order import is intentional.

```python
@classmethod
def build(cls, vectors, factory="Flat", train_size=100_000, seed=0):
    if factory.startswith("B"):
        index = faiss.index_binary_factory(dim, factory)
        data = binarize(vectors)
    else:
        index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
        data = vectors
    if not index.is_trained:
        sample = random 100K rows (if more)
        index.train(sample)
    index.add(data)
```

- **Plain:** One function that can build any kind of vector index from a short recipe string like "HNSW32,Flat" or "IVF1024,PQ48." Some index types first need to study a sample of the data to learn its shape (training) before you can add everything.
- **Technical:**
  - `index_factory` parses the string into the right composite index. `"OPQ48,PQ48"` becomes an `IndexPreTransform` (rotation) wrapping an `IndexPQ`.
  - The metric is always inner product, matching the normalized vectors.
  - What gets trained: IVF learns cluster centroids (k-means), PQ learns codebooks (k-means per sub-vector), and OPQ learns a rotation. Flat, SQfp16 and HNSW need no training.
  - Training on a 100K sample keeps OPQ and IVF4096 builds to seconds instead of minutes. FAISS wants at least 39 training points per centroid, and IVF4096 needs 160K, so the 200K corpus is right at the limit for the largest IVF I tested.
  - Names starting with `B` go to FAISS's separate binary index family, which works on packed bits and Hamming distance.

```python
def binarize(vectors):
    return np.packbits(vectors > 0, axis=1)
```

- **Plain:** Keep only whether each of the 384 numbers is positive or negative, one bit each. 384 bits = 48 bytes, instead of 1,536 bytes. That's exactly 32× smaller.
- **Technical:** Sign-quantization of a normalized embedding. Hamming distance between sign vectors approximates angular distance (it's the idea behind SimHash). It works because dimensions of a trained embedding are roughly zero-centered. The cost is precision: recall@10 drops to 0.654 vs exact. That's why rescoring exists.

```python
def tune(self, **params):
    if self.binary:
        index.hnsw.efSearch = value / index.nprobe = value
    else:
        faiss.ParameterSpace().set_index_parameter(self.index, name, value)
```

- **Technical:** `ParameterSpace` sets `efSearch` or `nprobe` on whatever sub-index has it, even when it's nested inside a pre-transform. Setting `index.nprobe` directly on an OPQ-wrapped IVF would fail silently: Python would create a new attribute on the wrapper and the search would ignore it. Binary indexes aren't supported by `ParameterSpace`, so they're set directly.

```python
def search(self, queries, k):
    if self.binary:
        dist, ids = self.index.search(binarize(queries), k)
        return -dist.astype(np.float32), ids
    return self.index.search(queries, k)
```

- **Technical:** It returns `(scores, ids)` as (nq × k) arrays. IDs are int64 and are `-1` when fewer than k results exist (possible with IVF at small `nprobe`). For binary indexes, Hamming distance is negated so that "higher is better" holds everywhere. Every caller filters `ids >= 0`.

```python
def rescore(queries, candidates, doc_vectors, chunk=256):
    gathered = doc_vectors[candidate ids]           # (chunk × k × 384)
    s = np.einsum("qkd,qd->qk", gathered, queries)  # exact inner products
    -inf for missing ids, then argsort
```

- **Plain:** The compressed index gives a shortlist of 100 that's roughly right. Then we look up the real, full-precision vectors for just those 100 and re-sort them exactly.
- **Technical:**
  - Two-stage search. The compressed index fits in RAM (48 B/vector); the float32 vectors stay in a memory-mapped file and only 100 rows per query are touched.
  - It's chunked by 256 queries so the gathered tensor stays around 40 MB. 10,000 queries at once would be 1.5 GB.
  - Missing candidates get `-inf` so they sort last.
  - The honest caveat: RAM shrinks 32×, but the full vectors still live on disk.
  - Measured result: binary + rescore loses 0.01 nDCG instead of 1.80.

```python
@contextmanager
def single_threaded():
    before = faiss.omp_get_max_threads(); faiss.omp_set_num_threads(1)
    try: yield
    finally: faiss.omp_set_num_threads(before)
```

- **Technical:** Latency is benchmarked on one thread so indexes compete fairly: a single query on flat search would otherwise use all 14 cores while HNSW can't parallelize one query. `try/finally` restores the thread count even if the timing code throws.

### 3.7 `fusion.py`: reciprocal rank fusion

```python
def rrf(rankings, k=60, weights=None, limit=None):
    for ranking, w in zip(rankings, weights):
        if not w: continue
        for rank, doc in enumerate(ranking, start=1):
            if doc < 0: continue
            fused[doc] = fused.get(doc, 0.0) + w / (k + rank)
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
```

- **Plain:** Each search method hands in its ranked list. A document gets points from each list it appears in, more points the higher it's ranked: first place earns 1/61, second 1/62, and so on. Add up the points, sort, and done. Being near the top of *both* lists beats being first in one and missing from the other. The weights say how much each list's vote counts.
- **Technical:**
  - Score: `RRF(d) = Σᵢ wᵢ / (k + rankᵢ(d))`.
  - It uses **ranks, not scores**. BM25 scores are unbounded (0 to 30+) and cosine scores sit in [−1, 1], so adding raw scores would let BM25 dominate. Normalizing them is fragile because the score distribution changes per query. Ranks are on the same scale for every retriever.
  - `k=60` flattens the curve: rank 1 vs rank 2 is 1/61 vs 1/62, a 1.6% difference, so one retriever's top pick can't dominate alone.
  - A document missing from a list gets 0 from it, not a penalty.
  - Weights are what made hybrid work on FiQA. At equal weights, BM25 (23.8 nDCG) dragged dense (40.4) down to 36.4. Down-weighting BM25 fixed it. The final 0.3 was tuned jointly with the reranker, so it's chosen for the whole pipeline, not for hybrid alone.
  - Sorting by `(-score, doc_id)` makes ties deterministic. Without that, Python's dict order would decide ties, and reruns could differ.
  - `-1` IDs (FAISS padding) are skipped. A weight of 0 skips a list entirely, which is how the tuning grid tests "no reranker."

```python
def top_margin(scores):
    return (scores[0] - scores[1]) / scores[0]
```

- **Plain:** How far ahead is the winner? If the top two are nearly tied, the system looks unsure.
- **Technical:** It's the relative gap between the top two fused scores, in [0, 1]. Dividing by `scores[0]` makes it scale-free. Edge cases: one result gives 1.0; none gives 0.0. This was my first routing signal on its own. It turned out to be uncorrelated with reranker gain, so it's now just one of the router's eight features, and the router gives it almost no weight.

### 3.8 `rerank.py`: the cross-encoder and its score cache

```python
class Reranker:
    def __init__(self, model_name, device=None, batch_size=32, max_length=512, cache=False):
        self.model = CrossEncoder(model_name, device=..., max_length=max_length)
        self.cache = ScoreCache(model_name) if cache else None

    def rerank(self, query, doc_ids, texts):
        scores = self.score(query, [texts[i] for i in doc_ids])
        order = np.argsort(-scores, kind="stable")
        return np.asarray(doc_ids)[order], scores[order]
```

- **Plain:** The embedding model reads the question and each document *separately*, like comparing two summaries. The cross-encoder reads them *together*, word against word, like a person reading both side by side. It's more accurate but much slower, so it only reads the top 30.
- **Technical:**
  - The default model is `Alibaba-NLP/gte-reranker-modernbert-base`: 150M parameters, Apache-2.0, built on ModernBERT. It takes the query and the passage as one sequence and outputs one relevance score.
  - The cost is one full transformer forward pass per (query, doc) pair, with attention across both texts. That's why it can't be precomputed or indexed like embeddings.
  - Measured on CPU: about 19 pairs/s, so 30 pairs is about 1.6 s, versus about 8 ms for everything else.
  - `max_length=512` truncates long documents. ModernBERT could read more, but longer inputs would make it slower still.
  - The reranker was chosen on dev data from four candidates. Three are in `hse tune`; bge-reranker-v2-m3 (568M) was dropped earlier because it lost to gte on SciFact-train (77.3 vs 79.2) and is 2.4× slower. The tutorial default, MiniLM-L6, made SciFact worse.

```python
class ScoreCache:
    key = sha1(query + "\x1f" + passage)
    table s(k TEXT PRIMARY KEY, v REAL) in cache/ce/<model>.sqlite
    get_many(keys) in chunks of 900; put_many(items)
```

- **Plain:** The expert's opinion on a given question and document never changes. So we write every opinion down, and next time we look it up instead of asking again.
- **Technical:**
  - A cross-encoder score is a pure function of (model, query text, passage text). The key hashes the two texts, with `\x1f` (an ASCII separator character) between them so "ab"+"c" and "a"+"bc" can't collide, and the model gets its own file.
  - SQLite is in the standard library, handles millions of rows, and supports fast `IN` lookups. The 900-key chunks stay under SQLite's limit on query parameters.
  - Effect: `hse tune` reruns in under a minute instead of 45, and the CI gate only scores pairs it hasn't seen.
  - Serving (`search`) doesn't use the cache. Live latency is measured honestly.

```python
def _predict(self, pairs, progress=False):
    keys → known = cache.get_many(unique keys)
    missing = first index of every key not in known
    fresh = model.predict(only the missing pairs) → cache.put_many
    return scores in the original order
```

- **Technical:** Only pairs not already cached go to the model, and duplicates within one call are scored once. The output is rebuilt in the caller's order, so callers can't tell whether a score came from the cache.

```python
def rerank_many(self, queries, candidate_lists, texts, progress=False):
    flatten all (query, doc) pairs, remember bounds, one _predict call, split back, sort each
```

- **Technical:** For evaluation, all pairs from all queries are flattened into one list so the model runs in full batches instead of one small batch per query. `bounds` records where each query's pairs start, to split the scores back. Results are identical to calling `rerank` per query.

### 3.9 `router.py`: deciding when to pay for the reranker

```python
def query_features(n_terms, bm_ids, bm_scores, dense_ids, dense_scores, fused_scores):
    fused_margin      (s1 − s2) / s1 of the fused scores
    dense_top1        cosine of the best dense hit
    dense_gap_1_2     dense s1 − s2
    dense_gap_1_10    dense s1 − s10
    overlap_at_10     |BM25 top 10 ∩ dense top 10| / 10
    bm25_gap_1_2      (b1 − b2) / b1
    log_bm25_hits     log(1 + number of BM25 matches)
    query_terms       analyzed query length
```

- **Plain:** Before deciding whether to call the expert, look at a few clues you already have. Do the two searches agree? Is there a clear winner? How long is the question? None of these costs anything, because they come from work already done.
- **Technical:**
  - All eight are byproducts of retrieval, so computing them costs microseconds.
  - Missing values (fewer than 2 or 10 results, zero BM25 hits) default to 0, so every query gets a finite vector. There's a test for the zero-BM25 case.
  - The router's learned weights say which clues matter: low **overlap** between BM25 and dense (they disagree) and a flat **dense score curve** (no clear winner) predict gain; longer queries gain more.

```python
def _ridge(X, y, lam):
    standardize X; add a bias column
    w = solve(ZᵀZ + λI, Zᵀy)
```

- **Plain:** Fit a straight-line rule: predicted improvement = a weighted sum of the clues. The λ term keeps the weights small, so a few odd practice questions can't push them around.
- **Technical:**
  - Closed-form ridge regression, with λ = 10. Features are standardized first, so λ penalizes them equally and the weights can be compared.
  - The target is each dev query's actual nDCG gain from reranking. It's noisy: 57% of dev queries gain nothing and 13% get worse.
  - In cross-validation on dev, ridge-on-gain beat logistic regression on "gain > 0", which throws away how *much* a query gains.

```python
def pick_threshold(pred, base, full, recover):
    if full.mean() ≤ base.mean(): never rerank
    walk queries from highest predicted gain down, adding each one's real gain
    stop when the total reaches recover × total gain → threshold = that query's prediction
```

- **Plain:** Sort the practice questions by how much we *think* the expert will help. Hand them to the expert one by one from the top, keeping a running total of the *actual* improvement. Stop when you've collected 80% of what the expert could give. The predicted score of the last question handed over becomes the cutoff.
- **Technical:** It's a greedy scan, O(n log n) for the sort. It picks the smallest share that meets the target *given the model's ranking*. If reranking doesn't help on average, the threshold is +∞ (never rerank).

```python
@classmethod
def train(cls, X, base, full, recover=0.8, lam=10.0, folds=5, seed=0):
    5-fold: fit on 4/5, predict the held-out 1/5 → out-of-fold predictions for every dev query
    threshold, share = pick_threshold(out-of-fold predictions, base, full, recover)
    refit on all dev queries → final weights
```

- **Plain:** If you set the cutoff using predictions on the same questions the rule learned from, the rule looks better than it is, and the cutoff ends up too low. So the cutoff is set using predictions on questions each rule *didn't* learn from.
- **Technical:**
  - The threshold comes from out-of-fold predictions; the weights come from a final fit on all the data. That's standard practice for calibrating a decision threshold.
  - Even so, on test the router recovered 69% of the gain, not the 80% target. The test queries' predicted gains were distributed differently (49% routed vs 40% on dev).
  - `always()` is a zero-weight router with threshold −∞, used when no trained router exists.
  - `save` and `load` write plain JSON, including the feature names, to `results/router.json`. Python's `json` reads and writes ±Infinity.

### 3.10 `pipeline.py`: wiring it together

```python
def build_artifacts(ds, cfg, encoder=None):
    BM25: load bm25-k1_0.9-b_0.4.npz or fit+save
    vectors = cached_doc_vectors(...)
    FAISS: load <factory>.faiss or build+save
    write config.json
```

- **Technical:** It's "build or load" for each piece. The BM25 file sits at the dataset level (it doesn't depend on the embedding model), and its name encodes `k1`/`b`. The FAISS file is named after its factory string, so several index types can coexist. Writing `config.json` next to the artifacts records what built them.

```python
class HybridSearcher:
    @classmethod
    def open(cls, dataset, cfg=None, with_reranker=True, cache_scores=False):
        ds = data.load(dataset); bm, index, encoder = build_artifacts(ds, cfg)
        reranker = Reranker(..., cache=cache_scores) if with_reranker else None
        router = Router.load(results/router.json) if it exists else None   → always rerank
```

- **Technical:** One object holds everything a query needs, loaded once. `with_reranker=False` skips loading the cross-encoder for runs that don't need it. `cache_scores=True` is used by evaluation and tuning, never by serving. With no trained router, it falls back to always reranking: the safe default for quality.

```python
def _fuse(self, bm_ids, dense_ids, ce_ids=None):
    if ce_ids is None:
        return rrf([bm_ids, dense_ids], k, [w_bm25, w_dense])
    return rrf([ce_ids, bm_ids, dense_ids], k, [w_rerank, w_bm25, w_dense])
```

- **Plain:** Without the expert, combine two votes. With the expert, combine three, and the expert's vote counts four times as much as the meaning search's. The expert dominates but doesn't erase the other two.
- **Technical:**
  - Standard practice replaces the top-N order with the cross-encoder's order (`hybrid+ce-only` in the results).
  - With the weak MiniLM reranker, voting beat replacing by 2.1 points. With gte it's a smaller edge: 53.89 vs 53.46 mean, significant only on NFCorpus.
  - Voting still matters for two reasons. BM25 and dense break ties among docs the reranker scores alike. And docs outside the top 30 keep their votes, so a relevant doc at fused rank 31 isn't buried.
  - Both `search` and `run_batch` go through `_fuse`, so the evaluated system is the served system.

```python
def features(self, query, bm, dense, fused_scores):
    return query_features(len(self.bm25.analyzer(query)), *bm, *dense, fused_scores)

def search(self, query, k=10, rerank="auto"):
    encode → dense (ids + scores) → BM25 (ids + scores) → fuse
    feats = self.features(...); predicted = router.predict(feats)
    use_reranker = reranker exists and (rerank == "always" or (rerank == "auto" and predicted ≥ threshold))
    if use_reranker: rerank top 30 → three-way fuse
    return SearchResult(hits, reranked, predicted_gain, timings_ms)
```

- **Technical:**
  - The dense and BM25 *scores* are kept now, not just IDs, because the router needs them.
  - The predicted gain is returned with every result, so the API shows why a query was or wasn't reranked.
  - `time.perf_counter()` gives per-stage timings, which is how the routing experiment measures latency.
  - The returned `score` is the fused RRF score, which only means something for ordering.

```python
def run_batch(self, queries, rerank=True, progress=False):
    one encoder call, one FAISS call; BM25 per query; fuse per query; features per query
    out = {"bm25", "dense", "hybrid", "features", "hybrid+ce-only", "hybrid+rerank"}
```

- **Technical:** The same logic as `search`, batched for evaluation speed. A single `index.search` on a (Q × 384) matrix lets FAISS use a BLAS matrix multiply. `features` is a (Q × 8) array saved into the per-query results file, which is what the router is applied to on test. `hybrid+ce-only` reuses the same cross-encoder scores, so that ablation costs nothing extra.

### 3.11 `metrics.py`: how quality is scored

```python
def ndcg_at_k(ranked, relevant, k=10):
    gains = [relevant.get(d, 0) for d in ranked[:k]]
    dcg = sum(g / log2(i + 2) for i, g in enumerate(gains) if g > 0)
    ideal = sorted(positive grades, reverse=True)[:k]
    idcg = sum(g / log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg
```

- **Plain:** A good result list puts the right answers near the top. Each right answer earns points, but answers further down the list earn less (position 1 counts fully, position 2 about 63%, position 10 about 29%). Then divide by the best possible score, so a perfect list scores 1.0 whatever the question.
- **Technical:**
  - DCG@k = Σ gainᵢ / log₂(i + 1) for 1-based rank i. In the code `i` is 0-based, hence `+2`.
  - Gains are the raw relevance grades (linear), which is how `trec_eval` and `pytrec_eval` compute `ndcg_cut_10`, the same tool BEIR uses. The alternative, exponential gain (2^g − 1), would give different numbers on graded datasets like NFCorpus and break comparability.
  - IDCG uses all positive judgments for the query, not only those retrieved.
  - Grade-0 judgments contribute nothing. NFCorpus and FiQA have some.
  - Queries with no positive judgments score 0.

```python
def recall_at_k(ranked, relevant, k=100): |relevant ∩ top-k| / |relevant|
def mrr_at_k(ranked, relevant, k=10):    1 / rank of first relevant
```

- **Technical:** Recall@100 answers "did the right document even reach the candidate pool?" That's the ceiling for anything downstream, including the reranker. MRR@10 is about the first hit only.

```python
def to_run(results, query_ids, doc_ids, drop_self=True):
    ranked = [doc_ids[i] for i in idx if i >= 0]
    if drop_self: ranked = [d for d in ranked if d != qid]
```

- **Technical:** It converts integer indices back to string doc IDs and drops any document whose ID equals the query ID. This is BEIR's `ignore_identical_ids` convention: in Quora and ArguAna the queries also exist as documents, and finding the query itself isn't a real result. It's harmless on datasets where IDs don't overlap.

```python
def paired_test(a, b): scipy.stats.ttest_rel over per-query scores (same query order)
```

- **Plain:** Is the difference real or luck? Compare the two systems question by question, not just their averages.
- **Technical:** It's a paired t-test, because both systems answer the same queries, and pairing removes the large per-query difficulty variance. Identical inputs return p = 1.0 (scipy would return NaN). The caveat is that per-query nDCG isn't normally distributed. A randomization test would be more robust, and I'd add one next.

### 3.12 `experiments.py`: the experiments

**`evaluate`** runs Path C. It saves aggregate metrics, p-values for the full pipeline vs each other system, wall time, and the environment (Python, FAISS, torch versions, CPU, git commit). The per-query file (query IDs, the 8 router features, per-query nDCG for every system) is what the trained router is applied to on test, so routing quality never re-runs the expensive models.

**`_held_out_runs`** loads one split per dataset (SciFact train, NFCorpus dev, FiQA dev by default) on top of the test-split index, since the corpus is the same, and runs the batch pipeline. Both `tune` and `routing` use it, so they see exactly the same dev queries.

**`tune`**

```python
pool[q] = union over w_bm25 in grid of fused top-30          # one candidate set per dev query
for model in (MiniLM-L6, bge-reranker-base, gte-modernbert):
    score every pool once (cached) → dict doc → score
    for depth in (10, 20, 30), w_bm25 in (0.1, 0.2, 0.3, 0.5), w_rerank in (0.5 … 10):
        candidates = fused top-depth at this w_bm25
        reranker order = candidates sorted by cached score
        final = rrf([reranker order, bm25, dense], [w_rerank, w_bm25, 1.0]) → dev nDCG@10
best = max over all 252 combinations of the unweighted mean across the 3 datasets
```

- **Plain:** Ask each candidate expert to grade every document that could possibly reach them, once, and write the grades down. Then try 252 combinations of settings using only the written grades, on practice questions only, and keep the best.
- **Technical:**
  - The pool trick is what makes this cheap and *exact*. The top-30 candidates change with the BM25 weight, but they always come from the union of the top 30s across the grid, so scoring that union once covers every combination.
  - Sorting the candidates by cached score with Python's stable sort gives the same order the live reranker produces.
  - The results were MiniLM 50.98, bge-reranker-base 51.24, **gte 54.87**, against dense alone at 49.49 (mean dev nDCG@10). The winner sits on a flat plateau (BM25 0.2–0.3, reranker 3–4), so small changes don't matter, which is reassuring.
  - The full grid is written to `results/tune.json`.

**`routing`**

```python
dev:  run the full pipeline on dev queries (cached scores) → features X, nDCG without / with reranking
      router = Router.train(X, base, full, recover=0.8) → results/router.json
test: per-query file from evaluate → router.predict(features) ≥ threshold → routed nDCG
      compare with always, never, and random routing at the same share = base + share × (full − base)
latency: 100 test queries per dataset × {never, always, auto}, 3 warm-up queries first
```

- **Technical:**
  - The router never sees a test query while it's trained or calibrated. The test numbers come from applying a frozen model.
  - The "random routing at the same share" baseline is the control that matters. A router that reranks 49% of queries has to beat random at 49%, not just "keep most of the quality." Measured pooled nDCG: routed 50.06, random 49.10.
  - Latency is measured on CPU with real model calls (no cache), because that's what serving pays.
  - Warm-up queries exclude one-time costs from the timings.
  - Per-dataset results go to `results/routing/<dataset>.json`; pooled results and the router weights go to `results/routing/summary.json`.

**`bench`**

```python
exact = Flat; truth = exact top-10; exact_ndcg
for each of 12 configs:
    build (timed), size = serialized bytes
    for each knob value:
        search all 10K queries for k=100 → recall@10, nDCG@10
        single-thread, one query at a time, 1000 queries → mean latency
    recall point  = fastest with recall ≥ 0.95
    quality point = fastest within 0.5 nDCG of exact
    compressed indexes: + rescoring numbers
```

- **Technical:**
  - Index size comes from `serialize_index`, the real bytes on disk including codebooks, centroids and HNSW links. That's why PQ48 shows 50 B/vector (48 codes + codebook overhead), so 30.7× rather than the theoretical 32×. Only the binary index hits exactly 32×.
  - Latency is per query, one at a time, on one thread, because that's the online case. Batched search would flatter Flat, since BLAS amortizes well across queries.
  - Latency excludes query encoding (the same for all indexes) and is measured with k=10. nDCG is measured with k=100.
  - The two operating points exist because I found HNSW at ef=16 hits 0.968 recall but loses 1.0 nDCG. Recall alone would have picked a setting that quietly costs quality.

**`_recall`** counts the overlap between returned and exact top-10 IDs, divided by the total. That's mean recall over queries, since every query has exactly 10 truth IDs.

**`plot_bench`** draws two panels: recall vs latency for every knob setting (log scale, with a dashed 0.95 line), and nDCG vs bytes per vector. `matplotlib.use("Agg")` renders without a display, which CI and SSH sessions need.

### 3.13 `report.py`

It reads every JSON in `results/eval` and `results/bench`, plus `results/routing/summary.json`, and writes `results/RESULTS.md` with the quality table, means, p-values, the benchmark table with both operating points, rescoring results, the routing table with the random baseline and the router's weights, and the environment. The report is generated from data, not typed by hand, so the numbers in it can't drift from the experiments.

### 3.14 `gate.py`

```python
GATED = ("dense", "hybrid", "hybrid+rerank")
METRICS = ("ndcg@10", "recall@100")

def run_gate(dataset="scifact", tolerance=0.5, update=False, device="cpu"):
    report = evaluate(dataset, Config(device=device), out_dir=results/gate)
    if update or no baseline: write baseline, pass
    for each system × metric: delta = (now − before) × 100; fail if delta < −tolerance
    return ok, markdown table lines
```

- **Plain:** It's a test that fails if search got worse. The answer key is a file in the repo with the last known-good scores.
- **Technical:**
  - It gates three systems so a regression is localized: if dense drops, the embedding side broke; if only hybrid+rerank drops, look at fusion or the reranker.
  - It gates two metrics because a change can keep nDCG@10 flat while losing recall@100. That's invisible today and painful once the reranker is changed.
  - `device="cpu"` makes local and CI runs compare the same way. With the first version of the pipeline, the CPU baseline matched the MPS numbers exactly, and GitHub's Linux runner matched it to +0.00 on every metric.
  - `evaluate` runs with the reranker score cache on. In CI the cache persists between runs, so only pairs whose text changed get re-scored. The cold first run scores about 9,000 pairs on a 4-vCPU runner.
  - The 0.5-point tolerance absorbs floating-point differences across machines while still catching real regressions.
  - Verified both ways: cutting `depth` from 100 to 10 failed the gate (recall@100 −11.4 points, exit code 1). Removing the query prefix *raised* SciFact scores and correctly passed.
  - Improvements never overwrite the baseline automatically. `--update` is a deliberate act that shows up in the PR diff.

### 3.15 `api.py`

```python
def create_app(searcher=None, dataset=None):
    @asynccontextmanager
    async def lifespan(app):
        state["searcher"] = searcher or HybridSearcher.open(env HSE_DATASET)
        yield
        state.clear()
```

- **Plain:** When the server starts, it loads everything once and keeps it in memory. Every request reuses it.
- **Technical:**
  - FastAPI's `lifespan` hook runs once at startup. Loading two models and the indexes per request would add seconds of latency.
  - It's an app *factory* (`create_app`) so tests can inject a fake searcher (`tests/test_api.py`) and run without models or data.

```python
@app.get("/search", response_model=SearchOut)
def search(q: str = Query(..., min_length=1, max_length=1000),
           k: int = Query(10, ge=1, le=100),
           rerank: Literal["auto", "always", "never"] = "auto"):
```

- **Technical:**
  - Input validation is declared in the signature. Empty queries, k outside 1..100, and unknown rerank modes get a 422 before any search runs. `max_length=1000` caps how much text a single request can push through the encoder.
  - The handler is a plain `def`, not `async def`, so FastAPI runs it in a threadpool and the CPU-bound search doesn't block the event loop.
  - Snippets are cut to 300 characters to keep responses small.
  - Every response includes `predicted_gain` (the router's estimate) and whether the reranker ran, plus per-stage timings.
  - `/health` reports the dataset, size, models and router threshold (`null` when there's no trained router). It returns 503 if the index isn't loaded, which is what a load balancer probe needs.

### 3.16 `cli.py`

```python
def cmd_search(args):
    from .pipeline import HybridSearcher
```

- **Technical:**
  - Every command imports its heavy dependencies inside the function. `hse --help` and `hse fix-openmp` never import torch or faiss. That's required for `fix-openmp`, which must run *before* the two libraries are ever loaded together.
  - `_config` copies only the flags you actually passed onto `Config`, so defaults stay in one place.
  - `cmd_gate` appends its table to `$GITHUB_STEP_SUMMARY` when that variable exists, so the comparison shows on the GitHub Actions run page, and it exits with code 1 on failure, which is what makes CI fail.

### 3.17 Tests (`tests/`)

| file | what it pins down |
|---|---|
| `test_bm25.py` | Stopwords + stemming, one score checked against the BM25 formula by hand, rare terms outweigh common ones, unknown terms and zero scores, save/load round trip |
| `test_fusion.py` | Docs in both lists rise to the top, exact RRF value, padding ignored, deterministic tie-breaks, weights scale votes, zero weight skips a list, margin edge cases |
| `test_metrics.py` | nDCG perfect and partial (checked by hand), zero grades ignored, recall, MRR, self-match dropping, identical-run p-value |
| `test_dense.py` | Flat equals NumPy brute force; HNSW recall rises with efSearch; binary packs 1 bit/dim and finds exact self-matches; rescoring restores exact order and handles missing IDs; save/load gives identical results |
| `test_router.py` | Each of the 8 features checked by hand; zero BM25 hits gives finite features; the threshold walk takes the highest predicted gain first; never reranks when the reranker hurts; a router trained on synthetic data finds the queries that gain; ±∞ survives JSON |
| `test_rerank.py` | With a fake model: each unique pair is scored once, the cache survives reopening, `rerank_many` splits and sorts per query, the cache key can't confuse query and passage boundaries |
| `test_data.py` | Dataset name parsing (including hyphenated names), qrels parsing, subsampling keeps every judged doc with its text |
| `test_api.py` | `/search` and `/health` with a fake searcher, input validation returns 422 |

All 37 run in under 2 seconds, with no models or downloads, so they run first in CI before the slower quality gate.

### 3.18 Project files

**`pyproject.toml`**

- Python `>=3.10,<3.13`. The upper bound is there because the native wheels (faiss, torch) lag new Python releases.
- The `hse` console script points at `cli:main`.
- On Linux, `[tool.uv.sources]` takes torch from PyTorch's CPU wheel index. Without that, CI downloads the default CUDA build: gigabytes of GPU libraries that a CPU runner can't use.
- `uv.lock` pins every transitive version, so results and CI are reproducible.

**`Makefile`** has one target per step (`setup`, `test`, `download`, `tune`, `eval`, `bench`, `route`, `report`, `gate`, `baseline`, `serve`). `setup` runs `hse fix-openmp`.

**`.github/workflows/ci.yml`**

```yaml
tests:          uv sync --locked → pytest
quality-gate:   needs: tests
                cache 1: ~/.cache/huggingface, cache/   (model weights + reranker scores)
                  restore: newest models-<hash(config.py)>-*, else newest models-*
                  save: models-<hash(config.py)>-<run id>, even if the gate fails
                cache 2: data/, artifacts/   (datasets, BM25, vectors, FAISS)
                  key = hash(uv.lock, config.py, data.py, bm25.py, embed.py, dense.py)
                hse gate --dataset scifact --tolerance 0.5 --device cpu
                upload results/gate/ as an artifact
```

- **Technical:**
  - `--locked` fails if `uv.lock` is out of date, rather than silently re-resolving to different versions.
  - The gate only runs after unit tests pass, which saves minutes on broken PRs.
  - The model cache is restored and saved as separate steps, and the save runs even when the gate fails. A plain `actions/cache` only saves on success, so a cold run that timed out or failed would throw away an hour of reranker scoring. A fresh key per run means newly scored pairs accumulate; GitHub evicts old entries when the repo's 10 GB cache limit is reached.
  - There are two caches because they go stale for different reasons. Model weights and reranker scores depend only on the models (and the text, which is in the score key), so they're restored even after code changes. Indexes depend on code, so their key includes every file that changes how artifacts are built. Editing `bm25.py` therefore invalidates the cache and rebuilds the index, instead of evaluating new code against stale indexes. Editing `fusion.py` doesn't, because fusion happens at query time.
  - `concurrency` cancels superseded runs on the same branch. `timeout-minutes: 120` covers the cold first run. Measured: 88 minutes, 78 of them spent reranking about 9,000 SciFact pairs at about 2 pairs/s on the free 4-vCPU runner. That run reproduced the Mac baseline exactly (+0.00 on all six metrics). The next run, with the cache restored, finished the gate in 39 seconds.
  - It's free: public repos get unlimited GitHub Actions minutes on standard runners.
