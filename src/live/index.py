"""Corpus loading, chunking and retrieval.

Backends:
  minilm : sentence-transformers all-MiniLM-L6-v2 + ChromaDB (what the paper reports)
  tfidf  : scikit-learn TF-IDF (offline; for plumbing tests only)

Documents are exposed to every system through opaque ids (D001...) so filenames such as
'fastapi_events_v0.90.md' cannot leak version/answer cues. Version metadata
(doc_version, status) is exposed only through Chunk.meta, which only E4 reads.
"""
import csv
import random
from dataclasses import dataclass
from pathlib import Path

CHUNK_TOKENS, CHUNK_OVERLAP = 500, 50   # whitespace-delimited tokens


@dataclass
class Chunk:
    doc_id: str
    chunk_no: int
    filename: str
    text: str
    meta: dict
    score: float = 0.0

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{self.chunk_no}"


def load_manifest(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_corpus(questions: list, manifest: list, processed_dir: Path) -> None:
    """Fail loudly if the corpus is not the one the benchmark refers to."""
    needed = {s for q in questions for s in q["target_sources"]}
    by_name = {m["filename"]: m for m in manifest}
    problems = []
    for name in sorted(needed):
        m = by_name.get(name)
        p = processed_dir / name
        if m is None:
            problems.append(f"{name}: not in manifest")
        elif (m.get("confirmed", "").strip().lower() != "yes"):
            problems.append(f"{name}: manifest row not confirmed (confirmed=yes required)")
        if not p.exists() or p.stat().st_size < 200:
            problems.append(f"{name}: missing or near-empty in {processed_dir}")
    if problems:
        shown = "\n  ".join(problems[:25])
        raise SystemExit(f"Corpus validation failed ({len(problems)} problems):\n  {shown}"
                         + ("\n  ..." if len(problems) > 25 else ""))


def chunk_text(text: str):
    words, step = text.split(), CHUNK_TOKENS - CHUNK_OVERLAP
    for i in range(0, max(len(words), 1), step):
        piece = words[i:i + CHUNK_TOKENS]
        if piece:
            yield " ".join(piece)
        if i + CHUNK_TOKENS >= len(words):
            break


class Index:
    def __init__(self, manifest: list, processed_dir: Path, backend: str = "minilm"):
        self.backend = backend
        names = sorted(m["filename"] for m in manifest if (processed_dir / m["filename"]).exists())
        ids = [f"D{i + 1:03d}" for i in range(len(names))]
        random.Random(0).shuffle(ids)                       # ids carry no ordering information
        self.doc_id = dict(zip(names, ids))
        self.filename = {v: k for k, v in self.doc_id.items()}
        mrow = {m["filename"]: m for m in manifest}
        self.chunks: list = []
        for n in names:
            m = mrow[n]
            meta = {"library": m.get("library", ""), "doc_version": m.get("doc_version", ""),
                    "status": m.get("status", "")}
            body = (processed_dir / n).read_text(encoding="utf-8", errors="ignore")
            for k, piece in enumerate(chunk_text(body)):
                self.chunks.append(Chunk(self.doc_id[n], k, n, piece, meta))
        texts = [c.text for c in self.chunks]
        if backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
            self._mat = self._vec.fit_transform(texts)
        elif backend == "minilm":
            import chromadb
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            emb = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self._chroma = chromadb.EphemeralClient()
            self._col = self._chroma.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})
            self._col.add(ids=[str(i) for i in range(len(texts))], embeddings=emb.tolist(),
                          documents=texts)
        else:
            raise ValueError(backend)

    def search(self, query: str, k: int = 5) -> list:
        if self.backend == "tfidf":
            sims = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
            order = sims.argsort()[::-1][:k]
            out = []
            for i in order:
                c = self.chunks[i]
                out.append(Chunk(c.doc_id, c.chunk_no, c.filename, c.text, c.meta, float(sims[i])))
            return out
        q = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        r = self._col.query(query_embeddings=q.tolist(), n_results=k)
        out = []
        for idx, dist in zip(r["ids"][0], r["distances"][0]):
            c = self.chunks[int(idx)]
            out.append(Chunk(c.doc_id, c.chunk_no, c.filename, c.text, c.meta, 1.0 - float(dist)))
        return out
