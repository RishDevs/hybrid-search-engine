import hashlib
import sqlite3

import numpy as np

from .config import CACHE_DIR, slug
from .embed import pick_device


class ScoreCache:
    def __init__(self, model_name, folder=None):
        folder = folder or CACHE_DIR / "ce"
        folder.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(folder / f"{slug(model_name)}.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS s (k TEXT PRIMARY KEY, v REAL)")

    @staticmethod
    def key(query, passage):
        return hashlib.sha1(f"{query}\x1f{passage}".encode()).hexdigest()

    def get_many(self, keys):
        found = {}
        for start in range(0, len(keys), 900):
            chunk = keys[start : start + 900]
            marks = ",".join("?" * len(chunk))
            found.update(self.db.execute(f"SELECT k, v FROM s WHERE k IN ({marks})", chunk).fetchall())
        return found

    def put_many(self, items):
        self.db.executemany("INSERT OR REPLACE INTO s VALUES (?, ?)", items)
        self.db.commit()


class Reranker:
    def __init__(self, model_name, device=None, batch_size=32, max_length=512, cache=False):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.batch_size = batch_size
        self.device = pick_device(device)
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)
        self.cache = ScoreCache(model_name) if cache else None

    def _predict(self, pairs, progress=False):
        if not pairs:
            return np.empty(0, dtype=np.float32)
        if self.cache is None:
            return np.asarray(
                self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=progress),
                dtype=np.float32,
            )
        keys = [ScoreCache.key(q, p) for q, p in pairs]
        known = self.cache.get_many(list(set(keys)))
        missing = {}
        for i, k in enumerate(keys):
            if k not in known and k not in missing:
                missing[k] = i
        if missing:
            fresh = self.model.predict(
                [pairs[i] for i in missing.values()], batch_size=self.batch_size, show_progress_bar=progress
            )
            new = dict(zip(missing, map(float, fresh)))
            self.cache.put_many(new.items())
            known.update(new)
        return np.array([known[k] for k in keys], dtype=np.float32)

    def score(self, query, passages):
        return self._predict([(query, p) for p in passages])

    def rerank(self, query, doc_ids, texts):
        scores = self.score(query, [texts[i] for i in doc_ids])
        order = np.argsort(-scores, kind="stable")
        return np.asarray(doc_ids)[order], scores[order]

    def rerank_many(self, queries, candidate_lists, texts, progress=False):
        pairs, bounds = [], [0]
        for q, cands in zip(queries, candidate_lists):
            pairs.extend((q, texts[i]) for i in cands)
            bounds.append(len(pairs))
        flat = self._predict(pairs, progress)
        out = []
        for i, cands in enumerate(candidate_lists):
            s = flat[bounds[i] : bounds[i + 1]]
            order = np.argsort(-s, kind="stable")
            out.append((np.asarray(cands)[order], s[order]))
        return out
