import re
from collections import Counter

import numpy as np
import Stemmer
from scipy import sparse

LUCENE_STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such "
    "that the their then there these they this to was will with".split()
)

_word = re.compile(r"\w+")


class Analyzer:
    def __init__(self, stem=True):
        self.stemmer = Stemmer.Stemmer("english") if stem else None

    def __call__(self, text):
        words = [w for w in _word.findall(text.lower()) if w not in LUCENE_STOPWORDS]
        if self.stemmer is not None:
            words = self.stemmer.stemWords(words)
        return words


class BM25:
    def __init__(self, k1=0.9, b=0.4):
        self.k1 = k1
        self.b = b
        self.analyzer = Analyzer()
        self.vocab = {}
        self.weights = None

    @property
    def n_docs(self):
        return self.weights.shape[0]

    def fit(self, texts):
        rows, cols, counts = [], [], []
        for i, text in enumerate(texts):
            for term, tf in Counter(self.analyzer(text)).items():
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                rows.append(i)
                counts.append(tf)

        n, v = len(texts), len(self.vocab)
        tf = sparse.csr_matrix(
            (np.asarray(counts, dtype=np.float32), (rows, cols)), shape=(n, v)
        )
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = max(doc_len.mean(), 1e-9)
        df = np.bincount(tf.indices, minlength=v)
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)

        row_of = np.repeat(np.arange(n), np.diff(tf.indptr))
        length_norm = self.k1 * (1 - self.b + self.b * doc_len / avg_len)
        tf.data = idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + length_norm[row_of])
        self.weights = tf.tocsc()
        return self

    def score(self, query):
        terms = Counter(t for t in self.analyzer(query) if t in self.vocab)
        if not terms:
            return np.zeros(self.n_docs, dtype=np.float32)
        cols = [self.vocab[t] for t in terms]
        qtf = np.fromiter(terms.values(), dtype=np.float32)
        return np.asarray(self.weights[:, cols] @ qtf).ravel()

    def search(self, query, k=100):
        scores = self.score(query)
        k = min(k, int(np.count_nonzero(scores)))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return top, scores[top]

    def search_many(self, queries, k=100):
        return [self.search(q, k) for q in queries]

    def save(self, path):
        w = self.weights
        terms = np.array(sorted(self.vocab, key=self.vocab.get), dtype=object)
        np.savez_compressed(
            path,
            data=w.data,
            indices=w.indices,
            indptr=w.indptr,
            shape=np.array(w.shape),
            terms=terms,
            params=np.array([self.k1, self.b]),
        )

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        k1, b = z["params"]
        bm = cls(float(k1), float(b))
        bm.weights = sparse.csc_matrix(
            (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
        )
        bm.vocab = {t: i for i, t in enumerate(z["terms"])}
        return bm
