"""Shared Pinecone index connection, used by both PineconeVectorStore and
PineconeChunkStore so they operate against one connection per index, not one
each -- vector and chunk-metadata storage are two ABCs/two classes here, but
one real Pinecone index underneath (see the migration spec for why they
aren't merged into a single class despite sharing storage).
"""
from typing import Optional

from pinecone import Pinecone


class PineconeConnection:
    def __init__(self, api_key: str, index_name: str, environment: Optional[str] = None):
        self._client = Pinecone(api_key=api_key)
        self.index = self._client.Index(index_name)
