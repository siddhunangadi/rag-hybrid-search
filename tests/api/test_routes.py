from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from tests.fakes import FakePineconeIndex


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build a fresh app per test, pointed at an isolated tmp_path data_dir.

    No real API keys are set in this sandbox, so the app is expected to fall
    back to MockProvider (generation) and FakeEmbeddingProvider (embedding).
    Pinecone (the only supported vector/chunk store) is faked out via
    FakePineconeIndex -- a working in-memory index -- so these tests stay
    hermetic and don't need a real Pinecone account.
    """
    monkeypatch.delenv("RAG_NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("RAG_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_PINECONE_API_KEY", "fake-key")
    monkeypatch.setenv("RAG_PINECONE_INDEX_NAME", "fake-index")

    with patch("rag_hybrid_search.storage.pinecone_connection.Pinecone") as mock_pc_cls:
        mock_pc_cls.return_value.Index = MagicMock(return_value=FakePineconeIndex())
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client


def test_health_reports_mock_and_fake_fallback(client, tmp_path):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["generation_provider"] == "mock"
    assert body["embedding_provider"] == "fake"
    assert body["data_dir"] == str(tmp_path / "data")


def test_liveness_reports_alive_with_no_dependencies(client):
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_reports_ready_when_dependencies_are_up(client):
    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    names = {c["name"] for c in body["checks"]}
    assert names == {"pinecone", "audit", "bm25", "embedding_provider"}
    assert all(c["ok"] for c in body["checks"])


def test_diagnostics_reports_operational_state(client):
    response = client.get("/diagnostics")

    assert response.status_code == 200
    body = response.json()
    assert body["build"]["name"] == "rag-hybrid-search"
    assert body["providers"]["generation"] == "mock"
    assert body["providers"]["embedding"] == "fake"
    assert body["ingestion_stats"] == {"total_documents": 0, "total_chunks": 0}
    assert body["audit_stats"]["total_events"] >= 0
    assert body["config"]["gemini_api_key"] in (True, False)
    assert body["metrics"]["request_count"] >= 0


def test_version_reads_installed_package_metadata(client):
    response = client.get("/version")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "rag-hybrid-search"
    assert body["version"]


def test_index_ingests_a_markdown_document(client):
    response = client.post(
        "/index",
        json={
            "documents": [
                {
                    "filename": "leave-policy.md",
                    "content": "Employees get 20 days of paid annual leave per year.",
                }
            ]
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["filename"] == "leave-policy.md"
    assert results[0]["status"] == "ready", results[0]
    assert results[0]["error"] is None


def test_index_reports_per_document_failure_without_500(client):
    response = client.post(
        "/index",
        json={
            "documents": [
                {"filename": "good.md", "content": "Valid markdown content here."},
                {"filename": "bad.unsupported", "content": "no loader for this extension"},
            ]
        },
    )

    assert response.status_code == 200
    results = response.json()["results"]
    statuses = {r["filename"]: r for r in results}
    assert statuses["good.md"]["status"] == "ready"
    assert statuses["bad.unsupported"]["status"] == "failed"
    assert statuses["bad.unsupported"]["error"] is not None


def test_index_filename_with_path_traversal_cannot_escape_uploads_dir(client, tmp_path):
    response = client.post(
        "/index",
        json={
            "documents": [
                {"filename": "../../evil.md", "content": "attacker controlled content"},
            ]
        },
    )

    assert response.status_code == 200
    # data_dir is tmp_path/"data", uploads live under it; "../../evil.md"
    # from the uploads dir would land directly in tmp_path if unsanitized.
    assert not (tmp_path / "evil.md").exists()
    escaped = [p for p in tmp_path.rglob("evil.md") if "uploads" not in p.parts]
    assert escaped == []


def test_upload_rejects_file_exceeding_max_size(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("RAG_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_PINECONE_API_KEY", "fake-key")
    monkeypatch.setenv("RAG_PINECONE_INDEX_NAME", "fake-index")
    monkeypatch.setenv("RAG_MAX_UPLOAD_SIZE_BYTES", "10")

    with patch("rag_hybrid_search.storage.pinecone_connection.Pinecone") as mock_pc_cls:
        mock_pc_cls.return_value.Index = MagicMock(return_value=FakePineconeIndex())
        app = create_app()
        with TestClient(app) as sized_client:
            response = sized_client.post(
                "/upload",
                files=[("files", ("big.txt", b"x" * 100, "text/plain"))],
            )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["status"] == "failed"
    assert "exceeds max upload size" in results[0]["error"]


def test_upload_rejects_content_not_matching_pdf_magic_bytes(client):
    response = client.post(
        "/upload",
        files=[("files", ("fake.pdf", b"not actually a pdf", "application/pdf"))],
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["status"] == "failed"
    assert "does not match expected format" in results[0]["error"]


def test_upload_filename_with_path_traversal_cannot_escape_uploads_dir(client, tmp_path):
    response = client.post(
        "/upload",
        files=[("files", ("../../evil.txt", b"attacker controlled content", "text/plain"))],
    )

    assert response.status_code == 200
    assert not (tmp_path / "evil.txt").exists()
    escaped = [p for p in tmp_path.rglob("evil.txt") if "uploads" not in p.parts]
    assert escaped == []


def test_index_rejects_empty_document_list(client):
    response = client.post("/index", json={"documents": []})

    assert response.status_code == 422


def test_answer_degrades_gracefully_with_mock_provider(client):
    client.post(
        "/index",
        json={
            "documents": [
                {
                    "filename": "leave-policy.md",
                    "content": "Employees get 20 days of paid annual leave per year.",
                }
            ]
        },
    )

    response = client.post(
        "/answer",
        json={"question": "How many days of paid leave do employees get?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "citations" in body
    assert "confidence" in body
    assert "verification" in body


def test_answer_rejects_blank_question(client):
    response = client.post("/answer", json={"question": "   "})

    assert response.status_code == 400


def test_answer_returns_generic_message_on_unexpected_error(client):
    with patch(
        "rag_hybrid_search.retrieval.retriever.HybridRetriever.retrieve",
        side_effect=RuntimeError("secret internal detail: /etc/pinecone-key.txt"),
    ):
        response = client.post("/answer", json={"question": "does this leak internals?"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail == "internal error answering question"
    assert "secret internal detail" not in detail


def test_documents_reports_empty_corpus(client):
    response = client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body == {"total_documents": 0, "total_chunks": 0, "documents": []}


def test_documents_reports_indexed_corpus(client):
    client.post(
        "/index",
        json={
            "documents": [
                {"filename": "leave-policy.md", "content": "Employees get 20 days of paid annual leave per year."},
            ]
        },
    )

    response = client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["total_documents"] == 1
    assert body["total_chunks"] >= 1
    assert body["documents"][0]["filename"] == "leave-policy.md"
    assert body["documents"][0]["chunk_count"] == body["total_chunks"]
    assert body["documents"][0]["document_id"]


def test_docs_endpoint_is_available(client):
    response = client.get("/docs")

    assert response.status_code == 200


def test_cors_header_present_only_for_allowlisted_origin(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("RAG_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_PINECONE_API_KEY", "fake-key")
    monkeypatch.setenv("RAG_PINECONE_INDEX_NAME", "fake-index")
    monkeypatch.setenv("RAG_CORS_ALLOW_ORIGINS", "https://allowed.example.com")

    with patch("rag_hybrid_search.storage.pinecone_connection.Pinecone") as mock_pc_cls:
        mock_pc_cls.return_value.Index = MagicMock(return_value=FakePineconeIndex())
        app = create_app()
        with TestClient(app) as cors_client:
            allowed = cors_client.get("/health", headers={"Origin": "https://allowed.example.com"})
            blocked = cors_client.get("/health", headers={"Origin": "https://evil.example.com"})

    assert allowed.headers.get("access-control-allow-origin") == "https://allowed.example.com"
    assert "access-control-allow-origin" not in blocked.headers


_GDPR_TEXT = """Article 5

1. Personal data shall be processed lawfully, fairly and in a transparent manner.

Article 17

1. The data subject shall have the right to obtain from the controller the erasure of personal data.
"""


def test_index_with_document_type_regulation_uses_clause_chunker(client):
    response = client.post(
        "/index",
        json={
            "documents": [
                {"filename": "gdpr.txt", "content": _GDPR_TEXT, "document_type": "regulation"},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "ready"

    documents_response = client.get("/documents")
    assert documents_response.json()["total_chunks"] >= 2


def test_index_without_document_type_defaults_to_general_chunker(client):
    response = client.post(
        "/index",
        json={
            "documents": [
                {"filename": "leave-policy.md", "content": "Employees get 20 days of paid annual leave per year."},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "ready"


def test_answer_show_article_returns_only_matching_article_via_structured_citations(client):
    client.post(
        "/index",
        json={
            "documents": [
                {"filename": "gdpr.txt", "content": _GDPR_TEXT, "document_type": "regulation"},
            ]
        },
    )

    response = client.post(
        "/answer",
        json={"question": "Show Article 17", "verify": False},
    )

    assert response.status_code == 200
    body = response.json()
    structured_citations = body["structured_citations"]
    assert len(structured_citations) >= 1
    assert all(c["article"] == "17" for c in structured_citations)


def test_debug_retrieval_exposes_fusion_and_rerank_as_separate_stages(monkeypatch, tmp_path):
    """The debug panel must let you see fusion order and rerank order
    separately -- otherwise there's no way to prove reranking actually
    changed anything versus silently passing candidates through."""
    monkeypatch.delenv("RAG_NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("RAG_GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_DEBUG_TOKEN", "test-token")
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAG_PINECONE_API_KEY", "fake-key")
    monkeypatch.setenv("RAG_PINECONE_INDEX_NAME", "fake-index")
    from api.main import create_app

    with patch("rag_hybrid_search.storage.pinecone_connection.Pinecone") as mock_pc_cls:
        mock_pc_cls.return_value.Index = MagicMock(return_value=FakePineconeIndex())
        app = create_app()
        with TestClient(app) as debug_client:
            debug_client.post(
                "/index",
                json={"documents": [{"filename": "leave.txt", "content": "Employees get 20 days of paid annual leave per year."}]},
            )

            response = debug_client.get(
                "/debug/retrieval",
                params={"query": "How many days of leave?"},
                headers={"X-Debug-Token": "test-token"},
            )

            assert response.status_code == 200
            body = response.json()
            assert "rrf_results" in body
            assert "rerank_results" in body
            assert len(body["rrf_results"]) > 0
            assert len(body["rerank_results"]) > 0
