from contextlib import contextmanager

import numpy as np

from .openmp import ensure_single_runtime

ensure_single_runtime()

import faiss  # noqa: E402


class DenseIndex:
    def __init__(self, index, factory):
        self.index = index
        self.factory = factory
        self.binary = isinstance(index, faiss.IndexBinary)

    @classmethod
    def build(cls, vectors, factory="Flat", train_size=100_000, seed=0):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, dim = vectors.shape
        if factory.startswith("B"):
            index = faiss.index_binary_factory(dim, factory)
            data = binarize(vectors)
        else:
            index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
            data = vectors
        if not index.is_trained:
            sample = data
            if n > train_size:
                pick = np.random.default_rng(seed).choice(n, train_size, replace=False)
                sample = data[np.sort(pick)]
            index.train(sample)
        index.add(data)
        return cls(index, factory)

    def __len__(self):
        return self.index.ntotal

    def tune(self, **params):
        for name, value in params.items():
            if self.binary:
                if name == "efSearch":
                    self.index.hnsw.efSearch = value
                elif name == "nprobe":
                    self.index.nprobe = value
            else:
                faiss.ParameterSpace().set_index_parameter(self.index, name, value)
        return self

    def search(self, queries, k):
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        if self.binary:
            dist, ids = self.index.search(binarize(queries), k)
            return -dist.astype(np.float32), ids
        return self.index.search(queries, k)

    def nbytes(self):
        if self.binary:
            return faiss.serialize_index_binary(self.index).nbytes
        return faiss.serialize_index(self.index).nbytes

    def save(self, path):
        if self.binary:
            faiss.write_index_binary(self.index, str(path))
        else:
            faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path, factory=None):
        path = str(path)
        try:
            index = faiss.read_index(path)
        except RuntimeError:
            index = faiss.read_index_binary(path)
        return cls(index, factory or "")


def binarize(vectors):
    return np.packbits(vectors > 0, axis=1)


def rescore(queries, candidates, doc_vectors, chunk=256):
    queries = np.atleast_2d(queries)
    scores = np.full(candidates.shape, -np.inf, dtype=np.float32)
    for start in range(0, len(queries), chunk):
        cand = candidates[start : start + chunk]
        valid = cand >= 0
        gathered = np.asarray(doc_vectors[np.where(valid, cand, 0).ravel()])
        gathered = gathered.reshape(*cand.shape, -1)
        s = np.einsum("qkd,qd->qk", gathered, queries[start : start + chunk])
        scores[start : start + chunk] = np.where(valid, s, -np.inf)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(scores, order, 1), np.take_along_axis(candidates, order, 1)


@contextmanager
def single_threaded():
    before = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        yield
    finally:
        faiss.omp_set_num_threads(before)
