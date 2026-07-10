import time

from rag_hybrid_search.models import RetrievedChunk
from rag_hybrid_search.providers.base import EmbeddingProvider
from rag_hybrid_search.storage.base import ChunkStore, VectorStore


class DenseRetriever:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunk_store: ChunkStore,
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunk_store = chunk_store

    def search(self, query: str, k: int, trace=None) -> list[RetrievedChunk]:
        start = time.perf_counter()
        query_embedding = self._embedding_provider.embed([query], input_type="query")[0]
        embed_latency_ms = (time.perf_counter() - start) * 1000
        if trace is not None:
            trace.log_query_embedding(
                provider=type(self._embedding_provider).__name__,
                model=self._embedding_provider.model_name,
                dim=self._embedding_provider.dimension,
                vector=query_embedding,
                latency_ms=embed_latency_ms,
            )
        raw_results = self._vector_store.query(query_embedding, k)

        results = []
        for chunk_id, score in raw_results:
            chunk = self._chunk_store.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=score,
                    bm25_score=None,
                    rrf_score=0.0,
                    rerank_score=None,
                    final_rank=0,
                )
            )
        return results
