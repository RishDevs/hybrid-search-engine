from hybrid_search import data


def test_parse_name():
    assert data.parse_name("quora-200k") == ("quora", 200_000)
    assert data.parse_name("trec-covid") == ("trec-covid", None)
    assert data.parse_name("fiqa") == ("fiqa", None)


def test_read_qrels(tmp_path):
    f = tmp_path / "test.tsv"
    f.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t2\nq2\td3\t1\n")
    assert data.read_qrels(f) == {"q1": {"d1": 1, "d2": 2}, "q2": {"d3": 1}}


def test_subsample_keeps_every_judged_doc():
    ids = [f"d{i}" for i in range(100)]
    ds = data.Dataset("toy", ids, [f"text {i}" for i in ids], {"q": "x"}, {"q": {"d7": 1, "d93": 1}})
    small = data.subsample(ds, 10, seed=1)
    assert len(small) == 10
    assert {"d7", "d93"} <= set(small.doc_ids)
    assert small.texts[small.doc_ids.index("d7")] == "text d7"
