import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_hybrid_search.models import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, index_path: str):
        self._index_path = Path(index_path)
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunk_ids = [c.chunk_id for c in chunks]
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return ranked[:k]

    def save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunk_ids": self._chunk_ids}, f)

    def load(self) -> bool:
        if not self._index_path.exists():
            return False
        with open(self._index_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunk_ids = data["chunk_ids"]
        return True

    def ping(self) -> None:
        """Cheap availability check for readiness probes.

        Only stats the index directory -- no disk read of the (potentially
        large) pickle file itself. Raises if the directory backing the
        index is unreachable (e.g. an unmounted/permission-denied data_dir).
        """
        self._index_path.parent.stat()
