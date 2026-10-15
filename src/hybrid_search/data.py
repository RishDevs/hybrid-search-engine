import csv
import json
import random
import re
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"


@dataclass
class Dataset:
    name: str
    doc_ids: list
    texts: list
    queries: dict
    qrels: dict

    def __len__(self):
        return len(self.doc_ids)


def parse_name(name):
    m = re.fullmatch(r"(.+)-(\d+)k", name)
    if m:
        return m[1], int(m[2]) * 1000
    return name, None


def download(name, root=DATA_DIR):
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{name}.zip"
    print(f"downloading {name} ...")
    with urllib.request.urlopen(BEIR_URL.format(name)) as resp, open(archive, "wb") as out:
        shutil.copyfileobj(resp, out)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    archive.unlink()
    return target


def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load(name, split="test", root=DATA_DIR, seed=13):
    base, size = parse_name(name)
    folder = download(base, root)

    qrels = read_qrels(Path(folder) / "qrels" / f"{split}.tsv")
    queries = {}
    for row in _read_jsonl(folder / "queries.jsonl"):
        if row["_id"] in qrels:
            queries[row["_id"]] = row["text"]
    qrels = {q: rel for q, rel in qrels.items() if q in queries}

    doc_ids, texts = [], []
    for row in _read_jsonl(folder / "corpus.jsonl"):
        title = (row.get("title") or "").strip()
        body = (row.get("text") or "").strip()
        doc_ids.append(row["_id"])
        texts.append(f"{title}. {body}" if title else body)

    ds = Dataset(name, doc_ids, texts, queries, qrels)
    if size and size < len(ds):
        ds = subsample(ds, size, seed)
    return ds


def subsample(ds, size, seed=13):
    judged = {d for rel in ds.qrels.values() for d in rel}
    keep = [i for i, d in enumerate(ds.doc_ids) if d in judged]
    if len(keep) > size:
        raise ValueError(f"{len(keep)} judged docs don't fit in a {size}-doc sample")
    rest = [i for i, d in enumerate(ds.doc_ids) if d not in judged]
    random.Random(seed).shuffle(rest)
    chosen = sorted(keep + rest[: size - len(keep)])
    return Dataset(
        ds.name,
        [ds.doc_ids[i] for i in chosen],
        [ds.texts[i] for i in chosen],
        ds.queries,
        ds.qrels,
    )
