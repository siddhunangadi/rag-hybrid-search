import chromadb

from rag_hybrid_search.models import EmbeddingRecord
from rag_hybrid_search.storage.base import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, data_dir: str, collection_name: str = "chunks"):
        self._client = chromadb.PersistentClient(path=data_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunk_id: str, embedding_record: EmbeddingRecord) -> None:
        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding_record.embedding],
            metadatas=[
                {
                    "embedding_model": embedding_record.embedding_model,
                    "embedding_dimension": embedding_record.embedding_dimension,
                    "provider": embedding_record.provider,
                    "created_at": embedding_record.created_at.isoformat(),
                }
            ],
        )

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        result = self._collection.query(query_embeddings=[embedding], n_results=k)
        ids = result["ids"][0]
        distances = result["distances"][0]
        # Chroma cosine space returns distance = 1 - cosine_similarity.
        return [(chunk_id, 1.0 - dist) for chunk_id, dist in zip(ids, distances)]

    def delete(self, chunk_ids: list[str]) -> None:
        self._collection.delete(ids=chunk_ids)
