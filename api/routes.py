"""Route handlers for the RAG hybrid search API.

Handlers only translate HTTP <-> pipeline calls; business logic lives in
``rag_pipeline`` and ``rag_hybrid_search``.
"""

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from api.dependencies import Container, get_container
from api.schemas import (
    AnswerRequest,
    DocumentSummary,
    DocumentsResponse,
    DocumentTypeParam,
    HealthResponse,
    IndexDocument,
    IndexRequest,
    IndexResponse,
    IndexResult,
    VersionResponse,
)
from rag_hybrid_search.compliance.clause_chunker import ClauseChunker
from rag_hybrid_search.ingestion.chunkers.base import Chunker
from rag_hybrid_search.ingestion.loaders.base import Loader
from rag_hybrid_search.ingestion.loaders.csv_loader import CsvLoader
from rag_hybrid_search.ingestion.loaders.docx_loader import DocxLoader
from rag_hybrid_search.ingestion.loaders.html import HtmlLoader
from rag_hybrid_search.ingestion.loaders.markdown import MarkdownLoader
from rag_hybrid_search.ingestion.loaders.pdf import PdfLoader
from rag_hybrid_search.ingestion.loaders.text import TextLoader
from rag_hybrid_search.ingestion.loaders.xlsx_loader import XlsxLoader
from rag_hybrid_search.models import IndexStatus
from rag_pipeline.models import RagAnswer

_PACKAGE_NAME = "rag-hybrid-search"
_FALLBACK_VERSION = "0.1.0"

_LOADERS_BY_SUFFIX: dict[str, Loader] = {
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
    ".html": HtmlLoader(),
    ".htm": HtmlLoader(),
    ".txt": TextLoader(),
    ".pdf": PdfLoader(),
    ".csv": CsvLoader(),
    ".xlsx": XlsxLoader(),
    ".docx": DocxLoader(),
}

router = APIRouter()


def _loader_for_filename(filename: str) -> Loader:
    """Pick a loader by file extension; raise ValueError for unsupported types."""
    suffix = Path(filename).suffix.lower()
    loader = _LOADERS_BY_SUFFIX.get(suffix)
    if loader is None:
        supported = ", ".join(sorted(_LOADERS_BY_SUFFIX))
        raise ValueError(f"unsupported file extension {suffix!r}; supported: {supported}")
    return loader


def _chunker_for_document_type(document_type: DocumentTypeParam, document_title: str) -> Chunker | None:
    """Pick a compliance-aware chunker for regulation-like documents, else the container default.

    Returns ``None`` (meaning "use the container's default chunker") for
    ``"general"`` so existing behavior is unchanged; any other
    ``document_type`` routes through ``ClauseChunker`` for clause-aware
    chunking and legal metadata tagging.
    """
    if document_type == "general":
        return None
    return ClauseChunker(document_title=document_title, document_type=document_type)


def _ingest_one(document: IndexDocument, container: Container) -> IndexResult:
    """Write a single uploaded document to disk and ingest it, catching per-item errors."""
    try:
        loader = _loader_for_filename(document.filename)
        dest_path = container.uploads_dir / document.filename
        dest_path.write_text(document.content, encoding="utf-8")

        chunker = _chunker_for_document_type(document.document_type, document.filename)
        ingestion_pipeline = container.build_ingestion_pipeline(loader, chunker=chunker)
        status = ingestion_pipeline.ingest(str(dest_path))
        return IndexResult(
            filename=document.filename,
            status="ready" if status == IndexStatus.READY else "failed",
        )
    except Exception as e:  # noqa: BLE001 - deliberately isolate per-document failures
        return IndexResult(filename=document.filename, status="failed", error=str(e))


async def _ingest_upload(
    file: UploadFile, container: Container, document_type: DocumentTypeParam = "general"
) -> IndexResult:
    """Write an uploaded file's raw bytes to disk and ingest it, catching per-item errors."""
    filename = file.filename or "upload"
    try:
        loader = _loader_for_filename(filename)
        dest_path = container.uploads_dir / filename
        contents = await file.read()
        dest_path.write_bytes(contents)

        chunker = _chunker_for_document_type(document_type, filename)
        ingestion_pipeline = container.build_ingestion_pipeline(loader, chunker=chunker)
        status = ingestion_pipeline.ingest(str(dest_path))
        return IndexResult(
            filename=filename,
            status="ready" if status == IndexStatus.READY else "failed",
        )
    except Exception as e:  # noqa: BLE001 - deliberately isolate per-file failures
        return IndexResult(filename=filename, status="failed", error=str(e))


@router.post("/answer", response_model=RagAnswer)
async def answer(request: AnswerRequest, container: Container = Depends(get_container)) -> RagAnswer:
    """Answer a question using the grounded RAG pipeline.

    Falls back to ``MockProvider``/``FakeEmbeddingProvider`` output when no
    real API keys are configured (see api/dependencies.py docstring).

    Defined ``async`` (and not offloaded to a worker thread) so it runs on
    the same event-loop thread as app startup: the underlying sqlite-backed
    chunk store is a single connection created at startup and sqlite3
    connections cannot be used across threads.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be blank")
    try:
        return container.rag_pipeline.answer(
            request.question, max_chunks=request.max_chunks, verify=request.verify
        )
    except Exception as e:  # noqa: BLE001 - convert unexpected errors to a clean 500 body
        raise HTTPException(status_code=500, detail=f"unexpected error answering question: {e}") from e


@router.post("/index", response_model=IndexResponse)
async def index_documents(
    request: IndexRequest, container: Container = Depends(get_container)
) -> IndexResponse:
    """Ingest one or more documents. Per-document failures are reported, not raised.

    Kept ``async`` for the same single-thread sqlite-connection reason noted
    on ``answer`` above.
    """
    results = [_ingest_one(document, container) for document in request.documents]
    return IndexResponse(results=results)


@router.post("/upload", response_model=IndexResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    document_type: DocumentTypeParam = Form(default="general"),
    container: Container = Depends(get_container),
) -> IndexResponse:
    """Ingest one or more uploaded files as raw bytes (binary-safe, for pdf/xlsx/docx/etc).

    Complements ``POST /index`` (JSON, text-only): this endpoint accepts real
    file bytes via multipart/form-data so binary formats can be ingested.
    Per-file failures are reported, not raised.

    ``document_type`` applies to every file in this request (multipart form
    data has no per-file metadata slot); pass ``"regulation"`` to route all
    of them through the clause-aware compliance chunker.
    """
    results = [await _ingest_upload(file, container, document_type) for file in files]
    return IndexResponse(results=results)


@router.get("/health", response_model=HealthResponse)
async def health(container: Container = Depends(get_container)) -> HealthResponse:
    """Report which providers were selected and where data is being persisted."""
    return HealthResponse(
        status="ok",
        generation_provider=container.generation_provider_name,
        embedding_provider=container.embedding_provider_name,
        data_dir=container.settings.data_dir,
    )


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(container: Container = Depends(get_container)) -> DocumentsResponse:
    """Report how many documents/chunks are currently indexed."""
    summaries = container.chunk_store.get_document_summaries()
    documents = [
        DocumentSummary(
            document_id=s["document_id"],
            filename=Path(s["source_path"]).name if s["source_path"] else s["document_id"],
            chunk_count=s["chunk_count"],
        )
        for s in summaries
    ]
    return DocumentsResponse(
        total_documents=len(documents),
        total_chunks=sum(d.chunk_count for d in documents),
        documents=documents,
    )


@router.get("/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    """Report the installed package name and version."""
    try:
        resolved_version = package_version(_PACKAGE_NAME)
    except PackageNotFoundError:
        resolved_version = _FALLBACK_VERSION
    return VersionResponse(name=_PACKAGE_NAME, version=resolved_version)
