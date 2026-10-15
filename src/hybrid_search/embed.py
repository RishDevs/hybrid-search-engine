import hashlib
import json
from pathlib import Path

import numpy as np


def pick_device(requested=None):
    if requested:
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder:
    def __init__(self, model_name, query_prefix="", device=None, batch_size=64):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self.device = pick_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)

    @property
    def dim(self):
        return self.model.get_embedding_dimension()

    def _encode(self, texts, progress=False):
        vecs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return np.ascontiguousarray(vecs, dtype=np.float32)

    def encode_docs(self, texts, progress=False):
        return self._encode(texts, progress)

    def encode_queries(self, texts):
        return self._encode([self.query_prefix + t for t in texts])


def corpus_fingerprint(doc_ids, model_name):
    h = hashlib.sha1(model_name.encode())
    for d in doc_ids:
        h.update(d.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def cached_doc_vectors(encoder, ds, folder):
    folder = Path(folder)
    vec_path, meta_path = folder / "doc_vectors.npy", folder / "doc_vectors.json"
    fp = corpus_fingerprint(ds.doc_ids, encoder.name)
    if vec_path.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") == fp:
            return np.load(vec_path, mmap_mode="r")
    folder.mkdir(parents=True, exist_ok=True)
    vecs = encoder.encode_docs(ds.texts, progress=True)
    np.save(vec_path, vecs)
    meta_path.write_text(json.dumps({"fingerprint": fp, "model": encoder.name, "shape": vecs.shape}))
    return np.load(vec_path, mmap_mode="r")
