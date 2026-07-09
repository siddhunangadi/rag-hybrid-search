"""Route handlers for the RAG hybrid search API.

Handlers only translate HTTP <-> pipeline calls; business logic lives in
``rag_pipeline`` and ``rag_hybrid_search``.
"""

import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from api.dependencies import Container, get_container
from api.schemas import (
    AnswerRequest,
    DebugRetrievalResponse,
    DebugRetrievedChunk,
    DeleteDocumentResponse,
    DocumentSummary,
    DocumentsResponse,
    DocumentTypeParam,
    HealthResponse,
    IndexDocument,
    IndexRequest,
    IndexResponse,
    IndexResult,
    JobStatusResponse,
    UploadAcceptedResponse,
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
from rag_pipeline.context_builder import build_context
from rag_pipeline.models import RagAnswer
from rag_pipeline.prompt_builder import build_prompt

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


def _ingest_bytes(
    filename: str, contents: bytes, document_type: DocumentTypeParam, container: Container
) -> IndexResult:
    """Write raw file bytes to disk and ingest them, catching per-item errors.

    Synchronous by design so it can run either inline (blocking /upload) or
    on the background ingestion worker thread (see api/jobs.py, /upload/async).
    """
    try:
        loader = _loader_for_filename(filename)
        dest_path = container.uploads_dir / filename
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


async def _ingest_upload(
    file: UploadFile, container: Container, document_type: DocumentTypeParam = "general"
) -> IndexResult:
    """Read an uploaded file's raw bytes and ingest it inline (blocks until done)."""
    filename = file.filename or "upload"
    contents = await file.read()
    return _ingest_bytes(filename, contents, document_type, container)


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


@router.post("/answer/stream")
async def answer_stream(
    request: AnswerRequest, container: Container = Depends(get_container)
) -> StreamingResponse:
    """Stream an answer as Server-Sent Events instead of blocking until generation finishes.

    Emits ``event: delta`` frames with raw text as the LLM produces it, then
    one ``event: final`` frame with the full verified ``RagAnswer`` JSON once
    citation verification/confidence scoring (which need the complete text)
    finish. Kept synchronous internally for the same single-thread sqlite
    reason as ``/answer``; the generator only yields, it doesn't block the
    event loop between yields since each ``next()`` is driven by Starlette's
    response iteration.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be blank")

    def event_stream():
        for event_type, payload in container.rag_pipeline.answer_stream(
            request.question, max_chunks=request.max_chunks, verify=request.verify
        ):
            if event_type == "delta":
                data = json.dumps({"text": payload})
            else:
                data = payload.model_dump_json()
            yield f"event: {event_type}\ndata: {data}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/debug/retrieval", response_model=DebugRetrievalResponse)
async def debug_retrieval(
    query: str,
    container: Container = Depends(get_container),
    x_debug_token: str = Header(default=None),
) -> DebugRetrievalResponse:
    """Trace every retrieval stage for a query against already-indexed data.

    Disabled unless ``settings.debug_token`` is set (returns 404, so its
    existence isn't disclosed) -- it exposes raw indexed chunk text and full
    prompts, which is sensitive for compliance-style documents. Enable by
    setting ``RAG_DEBUG_TOKEN`` and passing the same value as the
    ``X-Debug-Token`` header.
    """
    if not container.settings.debug_token:
        raise HTTPException(status_code=404, detail="not found")
    if x_debug_token != container.settings.debug_token:
        raise HTTPException(status_code=403, detail="invalid or missing X-Debug-Token")
    if not query.strip():
        raise HTTPException(status_code=400, detail="query must not be blank")

    pipeline = container.rag_pipeline
    retriever = pipeline.retriever

    dense_results = retriever.dense_retriever.search(query, k=retriever.dense_k)
    bm25_results = retriever.sparse_retriever.search(query, k=retriever.sparse_k)
    reranked, _trace = retriever.retrieve(query)

    context = build_context(sorted(reranked, key=lambda r: r.final_rank)[:5])
    prompt = build_prompt(query, context, prompt_version=pipeline.prompt_version)
    raw_generation = pipeline.generation_provider.generate(prompt)

    def _to_debug_chunks(results, score_attr: str) -> list[DebugRetrievedChunk]:
        ranked = sorted(results, key=lambda r: getattr(r, score_attr) or 0.0, reverse=True)
        return [
            DebugRetrievedChunk(
                chunk_id=r.chunk.chunk_id,
                chunk_index=r.chunk.chunk_index,
                score=getattr(r, score_attr),
                text=r.chunk.text,
            )
            for r in ranked
        ]

    return DebugRetrievalResponse(
        dense_results=_to_debug_chunks(dense_results, "dense_score"),
        bm25_results=_to_debug_chunks(bm25_results, "bm25_score"),
        rrf_results=_to_debug_chunks(reranked, "rrf_score"),
        prompt=prompt,
        raw_generation=raw_generation,
    )


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


@router.post("/upload/async", response_model=UploadAcceptedResponse, status_code=202)
async def upload_documents_async(
    files: list[UploadFile] = File(...),
    document_type: DocumentTypeParam = Form(default="general"),
    container: Container = Depends(get_container),
) -> UploadAcceptedResponse:
    """Accept file uploads without blocking on ingestion; poll GET /jobs/{job_id} for the result.

    File bytes are read here (on the request thread, before responding) since
    ``UploadFile`` isn't safe to hand across threads; parsing, chunking,
    embedding, and indexing then run on the background ingestion worker so a
    large or slow upload can't tie up the request thread or time out the
    client (see api/jobs.py -- JobStore serializes ingestion on one worker
    thread to avoid racing the shared BM25 rebuild).
    """
    payloads = [((file.filename or "upload"), await file.read()) for file in files]

    def work() -> dict:
        results = [
            _ingest_bytes(filename, contents, document_type, container)
            for filename, contents in payloads
        ]
        return IndexResponse(results=results).model_dump()

    job_id = container.job_store.submit(work)
    return UploadAcceptedResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, container: Container = Depends(get_container)) -> JobStatusResponse:
    """Poll the status of a background ingestion job started via POST /upload/async."""
    job = container.job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return JobStatusResponse(job_id=job.job_id, status=job.state, result=job.result, error=job.error)


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


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str, container: Container = Depends(get_container)
) -> DeleteDocumentResponse:
    """Purge a document from the chunk store, vector store, and BM25 index together.

    Uses ``IndexManager.remove_document``, which rebuilds the BM25 index
    after deleting so the sparse index never drifts out of sync with the
    chunk store.
    """
    chunks = container.chunk_store.get_by_document(document_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"document not found: {document_id}")
    container.index_manager.remove_document(document_id)
    return DeleteDocumentResponse(document_id=document_id, chunks_deleted=len(chunks))


@router.get("/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    """Report the installed package name and version."""
    try:
        resolved_version = package_version(_PACKAGE_NAME)
    except PackageNotFoundError:
        resolved_version = _FALLBACK_VERSION
    return VersionResponse(name=_PACKAGE_NAME, version=resolved_version)
