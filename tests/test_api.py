from fastapi.testclient import TestClient

from hybrid_search.api import create_app
from hybrid_search.config import Config
from hybrid_search.pipeline import Hit, SearchResult
from hybrid_search.router import MarginRouter


class FakeSearcher:
    def __init__(self):
        self.cfg = Config()
        self.reranker = None
        self.router = MarginRouter.always()
        self.calls = []

        class DS:
            name = "toy"

            def __len__(self):
                return 2

        self.ds = DS()

    def search(self, q, k=10, rerank="auto"):
        self.calls.append((q, k, rerank))
        hits = [Hit("d1", 0.9, "first doc " * 50), Hit("d2", 0.5, "second")][:k]
        return SearchResult(hits, False, 0.4, {"total": 1.5})


def test_search_endpoint():
    fake = FakeSearcher()
    with TestClient(create_app(fake)) as client:
        r = client.get("/search", params={"q": "hello", "k": 1, "rerank": "never"})
        assert r.status_code == 200
        body = r.json()
        assert body["hits"][0]["doc_id"] == "d1"
        assert len(body["hits"][0]["snippet"]) == 300
        assert fake.calls == [("hello", 1, "never")]

        assert client.get("/health").json()["dataset"] == "toy"


def test_rejects_bad_params():
    with TestClient(create_app(FakeSearcher())) as client:
        assert client.get("/search", params={"q": ""}).status_code == 422
        assert client.get("/search", params={"q": "x", "rerank": "sometimes"}).status_code == 422
        assert client.get("/search", params={"q": "x", "k": 0}).status_code == 422
