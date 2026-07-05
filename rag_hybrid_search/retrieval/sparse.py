from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.storage.base import ChunkStore
from rag_hybrid_search.storage.bm25_index import BM25Index


class SparseRetriever:
    def __init__(self, chunk_store: ChunkStore, bm25_index: BM25Index):
        self._chunk_store = chunk_store
        self._bm25_index = bm25_index

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        raw_results = self._bm25_index.search(query, k)

        results = []
        for chunk_id, score in raw_results:
            chunk = self._chunk_store.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=None,
                    bm25_score=score,
                    rrf_score=0.0,
                    rerank_score=None,
                    final_rank=0,
                )
            )
        return results
