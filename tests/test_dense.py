import numpy as np
import pytest

from hybrid_search.dense import DenseIndex, binarize, rescore


@pytest.fixture(scope="module")
def vectors():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4000, 64)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def brute_force(x, q, k):
    return np.argsort(-(q @ x.T), axis=1, kind="stable")[:, :k]


def test_flat_is_exact(vectors):
    q = vectors[:50] + 0.05
    _, ids = DenseIndex.build(vectors, "Flat").search(q, 10)
    np.testing.assert_array_equal(ids, brute_force(vectors, q, 10))


def test_hnsw_recall_improves_with_ef(vectors):
    q = vectors[:200]
    truth = brute_force(vectors, q, 10)
    index = DenseIndex.build(vectors, "HNSW16,Flat")

    def recall(ef):
        _, ids = index.tune(efSearch=ef).search(q, 10)
        return np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, truth)])

    assert recall(256) >= recall(8)
    assert recall(256) > 0.9


def test_binary_index_packs_one_bit_per_dim(vectors):
    index = DenseIndex.build(vectors, "BFlat")
    assert binarize(vectors[:1]).shape == (1, 8)
    scores, ids = index.search(vectors[:5], 3)
    assert ids[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert (scores[:, 0] == 0).all()


def test_rescore_restores_exact_order(vectors):
    q = vectors[:20] + 0.05
    truth = brute_force(vectors, q, 5)
    shuffled = np.ascontiguousarray(truth[:, ::-1])
    _, ids = rescore(q, shuffled, vectors)
    np.testing.assert_array_equal(ids, truth)


def test_rescore_handles_missing_candidates(vectors):
    cand = np.array([[3, -1, 1]])
    scores, ids = rescore(vectors[3:4], cand, vectors)
    assert ids[0, 0] == 3 and ids[0, -1] == -1 and np.isneginf(scores[0, -1])


def test_save_load(tmp_path, vectors):
    index = DenseIndex.build(vectors, "SQ8")
    index.save(tmp_path / "x.faiss")
    again = DenseIndex.load(tmp_path / "x.faiss")
    np.testing.assert_array_equal(index.search(vectors[:3], 4)[1], again.search(vectors[:3], 4)[1])
    assert again.nbytes() == index.nbytes()
