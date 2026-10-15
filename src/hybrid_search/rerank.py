import numpy as np

from .embed import pick_device


class Reranker:
    def __init__(self, model_name, device=None, batch_size=64, max_length=512):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.batch_size = batch_size
        self.device = pick_device(device)
        self.model = CrossEncoder(model_name, device=self.device, max_length=max_length)

    def score(self, query, passages):
        if not passages:
            return np.empty(0, dtype=np.float32)
        pairs = [(query, p) for p in passages]
        return np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
            dtype=np.float32,
        )

    def rerank(self, query, doc_ids, texts):
        scores = self.score(query, [texts[i] for i in doc_ids])
        order = np.argsort(-scores, kind="stable")
        return np.asarray(doc_ids)[order], scores[order]

    def rerank_many(self, queries, candidate_lists, texts, progress=False):
        pairs, bounds = [], [0]
        for q, cands in zip(queries, candidate_lists):
            pairs.extend((q, texts[i]) for i in cands)
            bounds.append(len(pairs))
        flat = np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=progress),
            dtype=np.float32,
        ) if pairs else np.empty(0, dtype=np.float32)
        out = []
        for i, cands in enumerate(candidate_lists):
            s = flat[bounds[i] : bounds[i + 1]]
            order = np.argsort(-s, kind="stable")
            out.append((np.asarray(cands)[order], s[order]))
        return out
