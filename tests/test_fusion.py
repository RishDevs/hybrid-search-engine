import numpy as np
import pytest

from hybrid_search.fusion import rrf, top_margin


def test_documents_found_by_both_lists_rise_to_the_top():
    ids, scores = rrf([np.array([1, 2, 3]), np.array([3, 4, 1])], k=60)
    assert set(ids[:2]) == {1, 3}
    assert scores[0] == pytest.approx(1 / 61 + 1 / 63)


def test_padding_ids_are_ignored_and_limit_applies():
    ids, _ = rrf([np.array([5, -1, -1]), np.array([7, 5])], k=60, limit=1)
    assert ids.tolist() == [5]


def test_ties_break_on_doc_id_for_stable_output():
    ids, _ = rrf([np.array([9]), np.array([4])], k=60)
    assert ids.tolist() == [4, 9]


def test_margin():
    assert top_margin(np.array([0.04, 0.03])) == pytest.approx(0.25)
    assert top_margin(np.array([0.5])) == 1.0
    assert top_margin(np.array([])) == 0.0


def test_weights_scale_each_list():
    ids, scores = rrf([np.array([1]), np.array([2])], k=60, weights=[0.5, 1.0])
    assert ids.tolist() == [2, 1]
    assert scores[1] == pytest.approx(0.5 / 61)


def test_zero_weight_list_is_skipped():
    ids, _ = rrf([np.array([1, 2]), np.array([3])], weights=[0.0, 1.0])
    assert ids.tolist() == [3]
