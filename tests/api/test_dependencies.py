from unittest.mock import MagicMock, patch

from api.dependencies import build_container
from rag_hybrid_search.config import Settings


def test_build_container_wires_pinecone_backend(tmp_path):
    settings = Settings(
        data_dir=str(tmp_path),
        pinecone_api_key="k", pinecone_index_name="idx",
    )
    with patch("api.dependencies.PineconeConnection") as mock_client_cls, \
         patch("api.dependencies.PineconeVectorStore") as mock_vs_cls, \
         patch("api.dependencies.PineconeChunkStore") as mock_cs_cls:
        mock_client_cls.return_value = MagicMock()
        mock_vs_cls.return_value = MagicMock()
        mock_cs_cls.return_value = MagicMock()
        container = build_container(settings)
        mock_client_cls.assert_called_once_with(
            api_key="k", index_name="idx", environment=None,
        )
        mock_vs_cls.assert_called_once_with(mock_client_cls.return_value)
        mock_cs_cls.assert_called_once_with(
            mock_client_cls.return_value, embedding_dimension=8,
        )
        assert container.index_manager.vector_store is mock_vs_cls.return_value
        assert container.chunk_store is mock_cs_cls.return_value
