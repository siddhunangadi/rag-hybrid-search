"""Route handlers for the RAG hybrid search API.

Handlers only translate HTTP <-> pipeline calls; business logic lives in
``rag_pipeline`` and ``rag_hybrid_search``.
"""

import json
import logging
import mimetypes
import time
import uuid
from datetime import date, datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from api.auth import Identity, require_role
from api.dependencies import Container, check_readiness, get_container
from api.schemas import (
    AnswerRequest,
    AuditEventsResponse,
    ComponentStatus,
    DebugRetrievalResponse,
    DebugRetrievedChunk,
    DeleteDocumentResponse,
    DiagnosticsResponse,
    DocumentSummary,
    DocumentsResponse,
    DocumentTypeParam,
    HealthResponse,
    IndexDocument,
    IndexRequest,
    IndexResponse,
    IndexResult,
    JobStatusResponse,
    LivenessResponse,
    MetricsResponse,
    ReadinessResponse,
    RiskCategoryParam,
    UploadAcceptedResponse,
    VersionResponse,
)
from rag_hybrid_search.audit import AuditEvent, EventStatus, EventType, now_utc
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
from rag_hybrid_search.models import ChunkProvenance, ContextChunk, IndexStatus
from rag_hybrid_search.retrieval.fusion import weighted_rrf
from rag_pipeline.context_builder import ContextLayout, build_context
from rag_pipeline.models import RagAnswer
from rag_pipeline.prompt_builder import build_prompt

logger = logging.getLogger(__name__)

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

# Magic-byte signatures for the binary formats we accept. Text formats
# (.md/.markdown/.html/.htm/.txt/.csv) have no reliable magic number, so
# they're instead checked for embedded NUL bytes (a strong signal of binary
# content mislabeled with a text extension).
_MAGIC_BYTES_BY_SUFFIX: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF-",),
    # .docx/.xlsx are zip containers -- PK\x03\x04 (normal) or PK\x05\x06
    # (empty archive) are both valid zip local-file-header signatures.
    ".docx": (b"PK\x03\x04", b"PK\x05\x06"),
    ".xlsx": (b"PK\x03\x04", b"PK\x05\x06"),
}
_TEXT_SUFFIXES = {".md", ".markdown", ".html", ".htm", ".txt", ".csv"}

router = APIRouter()


def _record_audit(
    container: Container,
    request: Request,
    identity: Identity,
    event_type: EventType,
    action: str,
    status: EventStatus,
    **fields,
) -> None:
    """Record one audit event, using the request's identity and request_id.

    Isolated in a try/except so a logging bug never breaks the actual
    request -- audit recording is observability, not a request-blocking
    concern.
    """
    try:
        container.audit_log.record(
            AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=now_utc(),
                request_id=getattr(request.state, "request_id", "unknown"),
                key_id=identity.key_id,
                role=identity.role,
                endpoint=request.url.path,
                action=action,
                status=status,
                **fields,
            )
        )
        container.metrics.increment("audit_events")
    except Exception:  # noqa: BLE001 - never let audit logging break a request
        logger.exception("failed to record audit event")
    if status == "failure":
        container.metrics.increment("failures")


def _safe_filename(filename: str) -> str:
    """Reduce a client-supplied filename to a bare basename.

    Uploaded filenames are attacker-controlled; joining them onto
    uploads_dir unsanitized would let "../../x" write outside the uploads
    directory. Rejects names that reduce to nothing (e.g. "..", "/").
    """
    name = Path(filename).name
    if not name or name in (".", ".."):
        raise ValueError(f"invalid filename {filename!r}")
    return name


def _validate_upload_content(filename: str, contents: bytes, max_bytes: int) -> None:
    """Reject oversized uploads and content that doesn't match its extension.

    Raises ValueError, caught by _ingest_bytes's existing per-file error
    handling -- one bad file fails just that item, not the whole batch.
    Binary formats (.pdf/.docx/.xlsx) are checked against a magic-byte
    signature; text formats are rejected if they contain a NUL byte (a
    strong signal of binary content mislabeled with a text extension).
    """
    if len(contents) > max_bytes:
        raise ValueError(f"file exceeds max upload size of {max_bytes} bytes")

    suffix = Path(filename).suffix.lower()
    guessed_mime, _ = mimetypes.guess_type(filename)
    logger.info("upload %r: suffix=%s guessed_mime=%s size=%d", filename, suffix, guessed_mime, len(contents))

    signatures = _MAGIC_BYTES_BY_SUFFIX.get(suffix)
    if signatures is not None:
        if not any(contents.startswith(sig) for sig in signatures):
            raise ValueError(f"file content does not match expected format for {suffix!r}")
    elif suffix in _TEXT_SUFFIXES and b"\x00" in contents:
        raise ValueError(f"file content is not valid text for {suffix!r}")


def _loader_for_filename(filename: str) -> Loader:
    """Pick a loader by file extension; raise ValueError for unsupported types."""
    suffix = Path(filename).suffix.lower()
    loader = _LOADERS_BY_SUFFIX.get(suffix)
    if loader is None:
        supported = ", ".join(sorted(_LOADERS_BY_SUFFIX))
        raise ValueError(f"unsupported file extension {suffix!r}; supported: {supported}")
    return loader


def _chunker_for_document_type(
    document_type: DocumentTypeParam,
    document_title: str,
    regulation: str | None = None,
    authority: str | None = None,
    jurisdiction: str | None = None,
    effective_date: date | None = None,
    risk_category: RiskCategoryParam | None = None,
) -> Chunker | None:
    """Pick a compliance-aware chunker for regulation-like documents, else the container default.

    Returns ``None`` (meaning "use the container's default chunker") for
    ``"general"`` so existing behavior is unchanged; any other
    ``document_type`` routes through ``ClauseChunker`` for clause-aware
    chunking and legal metadata tagging.
    """
    if document_type == "general":
        return None
    return ClauseChunker(
        document_title=document_title,
        document_type=document_type,
        regulation=regulation,
        authority=authority,
        jurisdiction=jurisdiction,
        effective_date=effective_date,
        risk_category=risk_category,
    )


def _ingest_one(
    document: IndexDocument, container: Container,
    existing_pairs: list | None = None, rebuild_bm25: bool = True,
    document_hashes: dict | None = None,
) -> IndexResult:
    """Ingest a single JSON-submitted text document, catching per-item errors."""
    return _ingest_bytes(
        document.filename,
        document.content.encode("utf-8"),
        document.document_type,
        container,
        regulation=document.regulation,
        authority=document.authority,
        jurisdiction=document.jurisdiction,
        effective_date=document.effective_date,
        risk_category=document.risk_category,
        existing_pairs=existing_pairs,
        rebuild_bm25=rebuild_bm25,
        document_hashes=document_hashes,
    )


def _ingest_bytes(
    filename: str,
    contents: bytes,
    document_type: DocumentTypeParam,
    container: Container,
    regulation: str | None = None,
    authority: str | None = None,
    jurisdiction: str | None = None,
    effective_date: date | None = None,
    risk_category: RiskCategoryParam | None = None,
    existing_pairs: list | None = None,
    rebuild_bm25: bool = True,
    document_hashes: dict | None = None,
) -> IndexResult:
    """Write raw file bytes to disk and ingest them, catching per-item errors.

    Synchronous by design so it can run either inline (blocking /upload) or
    on the background ingestion worker thread (see api/jobs.py, /upload/async).

    existing_pairs/rebuild_bm25/document_hashes: passed through to
    IngestionPipeline.ingest() so a multi-file batch (caller builds one
    dedup cache, one document-hash cache, and defers one BM25 rebuild for
    the whole batch) avoids the per-document full-corpus rescans profiling
    identified as the bottleneck. Each file still gets its own pipeline
    instance (loader/chunker vary per format), but all instances share the
    same underlying chunk_store/index_manager, so the caches and deferred
    rebuild are valid across them.
    """
    try:
        safe_name = _safe_filename(filename)
        _validate_upload_content(safe_name, contents, container.settings.max_upload_size_bytes)
        loader = _loader_for_filename(safe_name)
        dest_path = container.uploads_dir / safe_name
        dest_path.write_bytes(contents)

        chunker = _chunker_for_document_type(
            document_type, safe_name,
            regulation=regulation, authority=authority, jurisdiction=jurisdiction,
            effective_date=effective_date, risk_category=risk_category,
        )
        ingestion_pipeline = container.build_ingestion_pipeline(loader, chunker=chunker)
        status = ingestion_pipeline.ingest(
            str(dest_path), existing_pairs=existing_pairs, rebuild_bm25=rebuild_bm25,
            document_hashes=document_hashes,
        )
        return IndexResult(
            filename=filename,
            status="ready" if status == IndexStatus.READY else "failed",
        )
    except Exception as e:  # noqa: BLE001 - deliberately isolate per-file failures
        return IndexResult(filename=filename, status="failed", error=str(e))


async def _ingest_upload(
    file: UploadFile,
    container: Container,
    document_type: DocumentTypeParam = "general",
    regulation: str | None = None,
    authority: str | None = None,
    jurisdiction: str | None = None,
    effective_date: date | None = None,
    risk_category: RiskCategoryParam | None = None,
    existing_pairs: list | None = None,
    rebuild_bm25: bool = True,
    document_hashes: dict | None = None,
) -> IndexResult:
    """Read an uploaded file's raw bytes and ingest it inline (blocks until done)."""
    filename = file.filename or "upload"
    contents = await file.read()
    return _ingest_bytes(
        filename, contents, document_type, container,
        regulation=regulation, authority=authority, jurisdiction=jurisdiction,
        effective_date=effective_date, risk_category=risk_category,
        existing_pairs=existing_pairs, rebuild_bm25=rebuild_bm25,
        document_hashes=document_hashes,
    )


@router.post("/answer", response_model=RagAnswer)
async def answer(
    http_request: Request,
    request: AnswerRequest,
    container: Container = Depends(get_container),
    identity: Identity = Depends(require_role("admin", "reader")),
) -> RagAnswer:
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
    start = time.monotonic()
    container.metrics.increment("retrievals")
    container.metrics.increment("generations")
    try:
        result = container.rag_pipeline.answer(
            request.question, max_chunks=request.max_chunks, verify=request.verify
        )
    except Exception:  # noqa: BLE001 - convert unexpected errors to a clean 500 body
        # Full exception (which may embed provider response bodies, internal
        # paths, etc.) goes to server-side logs only; the client gets a
        # generic message so nothing sensitive leaks in the response.
        logger.exception("unexpected error answering question")
        _record_audit(
            container, http_request, identity, "query", "answer", "failure",
            query_text=request.question, retrieval_mode="hybrid",
            duration_ms=(time.monotonic() - start) * 1000,
            error="internal error answering question",
        )
        raise HTTPException(status_code=500, detail="internal error answering question") from None

    _record_audit(
        container, http_request, identity, "query", "answer",
        "failure" if result.error else "success",
        query_text=request.question, retrieval_mode="hybrid",
        duration_ms=(time.monotonic() - start) * 1000,
        retrieved_document_ids=sorted({c.document_id for c in result.structured_citations}),
        cited_regulations=sorted(
            {c.regulation for c in result.structured_citations if c.regulation}
        ),
        confidence_score=result.confidence.overall,
        error=result.error,
    )
    return result


@router.post("/answer/stream")
async def answer_stream(
    http_request: Request,
    request: AnswerRequest,
    container: Container = Depends(get_container),
    identity: Identity = Depends(require_role("admin", "reader")),
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

    container.metrics.increment("retrievals")
    container.metrics.increment("generations")

    def event_stream():
        start = time.monotonic()
        final_answer = None
        for event_type, payload in container.rag_pipeline.answer_stream(
            request.question, max_chunks=request.max_chunks, verify=request.verify
        ):
            if event_type == "delta":
                data = json.dumps({"text": payload})
            else:
                final_answer = payload
                data = payload.model_dump_json()
            yield f"event: {event_type}\ndata: {data}\n\n"

        if final_answer is not None:
            _record_audit(
                container, http_request, identity, "query", "answer_stream",
                "failure" if final_answer.error else "success",
                query_text=request.question, retrieval_mode="hybrid",
                duration_ms=(time.monotonic() - start) * 1000,
                retrieved_document_ids=sorted(
                    {c.document_id for c in final_answer.structured_citations}
                ),
                cited_regulations=sorted(
                    {c.regulation for c in final_answer.structured_citations if c.regulation}
                ),
                confidence_score=final_answer.confidence.overall,
                error=final_answer.error,
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/debug/retrieval", response_model=DebugRetrievalResponse)
async def debug_retrieval(
    query: str,
    container: Container = Depends(get_container),
    x_debug_token: str = Header(default=None),
    _identity=Depends(require_role("admin", "reader")),
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
    # Fuse explicitly (pre-rerank) so the debug panel can show reranking's
    # actual before/after effect, not just the final post-rerank order.
    fused = weighted_rrf(
        dense_results, bm25_results,
        dense_weight=retriever.dense_weight, sparse_weight=retriever.sparse_weight,
        k=retriever.rrf_k,
    )
    reranked, _trace = retriever.retrieve(query)

    top_chunks = [
        ContextChunk(
            chunk=r,
            provenance=ChunkProvenance(primary_subquery=0, all_subqueries=[0]),
        )
        for r in sorted(reranked, key=lambda r: r.final_rank)[:5]
    ]
    context = build_context(top_chunks, subqueries=[], layout=ContextLayout.FLAT)
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
        rrf_results=_to_debug_chunks(fused, "rrf_score"),
        rerank_results=_to_debug_chunks(reranked, "rerank_score"),
        prompt=prompt,
        raw_generation=raw_generation,
    )


@router.post("/index", response_model=IndexResponse)
async def index_documents(
    http_request: Request,
    request: IndexRequest,
    container: Container = Depends(get_container),
    identity: Identity = Depends(require_role("admin")),
) -> IndexResponse:
    """Ingest one or more documents. Per-document failures are reported, not raised.

    Kept ``async`` for the same single-thread sqlite-connection reason noted
    on ``answer`` above.

    Shares one dedup cache and defers the BM25 rebuild to once after the
    whole batch (see IngestionPipeline.ingest_batch docstring) instead of
    once per document, avoiding the full-corpus rescan profiling identified
    as the ingestion bottleneck.
    """
    existing_pairs = [
        (item.chunk, item.embedding) for item in container.chunk_store.all_with_embeddings()
    ] if request.documents else []
    document_hashes = {
        s["source_path"]: s["document_id"] for s in container.chunk_store.get_document_summaries()
    } if request.documents else {}
    results = [
        _ingest_one(
            document, container, existing_pairs=existing_pairs, rebuild_bm25=False,
            document_hashes=document_hashes,
        )
        for document in request.documents
    ]
    if request.documents:
        container.index_manager.rebuild_bm25_index()
    container.metrics.increment("uploads", len(request.documents))
    for document, result in zip(request.documents, results):
        _record_audit(
            container, http_request, identity, "upload", "index_documents",
            "success" if result.status == "ready" else "failure",
            document_id=result.filename,
            regulation_metadata={
                "document_type": document.document_type,
                "regulation": document.regulation,
                "authority": document.authority,
                "jurisdiction": document.jurisdiction,
                "effective_date": str(document.effective_date) if document.effective_date else None,
                "risk_category": document.risk_category,
            },
            error=result.error,
        )
    return IndexResponse(results=results)


@router.post("/upload", response_model=IndexResponse)
async def upload_documents(
    http_request: Request,
    files: list[UploadFile] = File(...),
    document_type: DocumentTypeParam = Form(default="general"),
    regulation: str | None = Form(default=None),
    authority: str | None = Form(default=None),
    jurisdiction: str | None = Form(default=None),
    effective_date: date | None = Form(default=None),
    risk_category: RiskCategoryParam | None = Form(default=None),
    container: Container = Depends(get_container),
    identity: Identity = Depends(require_role("admin")),
) -> IndexResponse:
    """Ingest one or more uploaded files as raw bytes (binary-safe, for pdf/xlsx/docx/etc).

    Complements ``POST /index`` (JSON, text-only): this endpoint accepts real
    file bytes via multipart/form-data so binary formats can be ingested.
    Per-file failures are reported, not raised.

    ``document_type`` and the regulation/authority/jurisdiction/
    effective_date/risk_category fields apply to every file in this request
    (multipart form data has no per-file metadata slot); pass
    ``"regulation"`` to route all of them through the clause-aware
    compliance chunker.

    Shares one dedup cache and defers the BM25 rebuild to once after the
    whole batch instead of once per file (see IngestionPipeline.ingest_batch
    docstring), avoiding the full-corpus rescan profiling identified as the
    ingestion bottleneck.
    """
    existing_pairs = [
        (item.chunk, item.embedding) for item in container.chunk_store.all_with_embeddings()
    ] if files else []
    document_hashes = {
        s["source_path"]: s["document_id"] for s in container.chunk_store.get_document_summaries()
    } if files else {}
    results = [
        await _ingest_upload(
            file, container, document_type,
            regulation=regulation, authority=authority, jurisdiction=jurisdiction,
            effective_date=effective_date, risk_category=risk_category,
            existing_pairs=existing_pairs, rebuild_bm25=False,
            document_hashes=document_hashes,
        )
        for file in files
    ]
    if files:
        container.index_manager.rebuild_bm25_index()
    container.metrics.increment("uploads", len(files))
    for filename_result in results:
        _record_audit(
            container, http_request, identity, "upload", "upload_documents",
            "success" if filename_result.status == "ready" else "failure",
            document_id=filename_result.filename,
            regulation_metadata={
                "document_type": document_type,
                "regulation": regulation,
                "authority": authority,
                "jurisdiction": jurisdiction,
                "effective_date": str(effective_date) if effective_date else None,
                "risk_category": risk_category,
            },
            error=filename_result.error,
        )
    return IndexResponse(results=results)


@router.post("/upload/async", response_model=UploadAcceptedResponse, status_code=202)
async def upload_documents_async(
    http_request: Request,
    files: list[UploadFile] = File(...),
    document_type: DocumentTypeParam = Form(default="general"),
    regulation: str | None = Form(default=None),
    authority: str | None = Form(default=None),
    jurisdiction: str | None = Form(default=None),
    effective_date: date | None = Form(default=None),
    risk_category: RiskCategoryParam | None = Form(default=None),
    container: Container = Depends(get_container),
    identity: Identity = Depends(require_role("admin")),
) -> UploadAcceptedResponse:
    """Accept file uploads without blocking on ingestion; poll GET /jobs/{job_id} for the result.

    File bytes are read here (on the request thread, before responding) since
    ``UploadFile`` isn't safe to hand across threads; parsing, chunking,
    embedding, and indexing then run on the background ingestion worker so a
    large or slow upload can't tie up the request thread or time out the
    client (see api/jobs.py -- JobStore serializes ingestion on one worker
    thread to avoid racing the shared BM25 rebuild).

    Shares one dedup cache and defers the BM25 rebuild to once after the
    whole batch instead of once per file (see IngestionPipeline.ingest_batch
    docstring), avoiding the full-corpus rescan profiling identified as the
    ingestion bottleneck -- the dominant cost at the 1000-file scale this
    endpoint exists for.
    """
    payloads = [((file.filename or "upload"), await file.read()) for file in files]

    def work() -> dict:
        existing_pairs = [
            (item.chunk, item.embedding) for item in container.chunk_store.all_with_embeddings()
        ] if payloads else []
        document_hashes = {
            s["source_path"]: s["document_id"] for s in container.chunk_store.get_document_summaries()
        } if payloads else {}
        results = [
            _ingest_bytes(
                filename, contents, document_type, container,
                regulation=regulation, authority=authority, jurisdiction=jurisdiction,
                effective_date=effective_date, risk_category=risk_category,
                existing_pairs=existing_pairs, rebuild_bm25=False,
                document_hashes=document_hashes,
            )
            for filename, contents in payloads
        ]
        if payloads:
            container.index_manager.rebuild_bm25_index()
        container.metrics.increment("uploads", len(payloads))
        for result in results:
            _record_audit(
                container, http_request, identity, "upload", "upload_documents_async",
                "success" if result.status == "ready" else "failure",
                document_id=result.filename,
                regulation_metadata={
                    "document_type": document_type,
                    "regulation": regulation,
                    "authority": authority,
                    "jurisdiction": jurisdiction,
                    "effective_date": str(effective_date) if effective_date else None,
                    "risk_category": risk_category,
                },
                error=result.error,
            )
        return IndexResponse(results=results).model_dump()

    job_id = container.job_store.submit(work)
    return UploadAcceptedResponse(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    container: Container = Depends(get_container),
    _identity=Depends(require_role("admin", "reader")),
) -> JobStatusResponse:
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


@router.get("/health/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    """Process-up check: no dependency calls, always 200 while the app is running.

    Distinct from /health/ready -- a load balancer/orchestrator should
    restart the process on liveness failure, but only stop routing traffic
    (not restart) on readiness failure.
    """
    return LivenessResponse()


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(container: Container = Depends(get_container)) -> ReadinessResponse:
    """Dependency check: Pinecone, embedding provider, BM25, audit log.

    Every check is a cheap metadata/stat call (see check_readiness), so this
    is safe for a frequent orchestrator probe. Returns 200 either way with
    status/checks in the body -- the caller decides what "not ready" means
    for routing, rather than this endpoint deciding via status code.
    """
    checks = check_readiness(container)
    status = "ready" if all(c["ok"] for c in checks) else "not_ready"
    return ReadinessResponse(status=status, checks=[ComponentStatus(**c) for c in checks])


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    container: Container = Depends(get_container),
    _identity=Depends(require_role("admin", "reader")),
) -> DocumentsResponse:
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
    http_request: Request,
    document_id: str,
    container: Container = Depends(get_container),
    identity: Identity = Depends(require_role("admin")),
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
    _record_audit(
        container, http_request, identity, "deletion", "delete_document", "success",
        document_id=document_id, retrieval_stats={"chunks_deleted": len(chunks)},
    )
    return DeleteDocumentResponse(document_id=document_id, chunks_deleted=len(chunks))


@router.get("/audit/events", response_model=AuditEventsResponse)
async def list_audit_events(
    container: Container = Depends(get_container),
    _identity: Identity = Depends(require_role("admin")),
    event_type: EventType | None = Query(default=None),
    key_id: str | None = Query(default=None),
    role: str | None = Query(default=None),
    document_id: str | None = Query(default=None),
    status: EventStatus | None = Query(default=None),
    start: datetime | None = Query(default=None, description="ISO-8601 lower bound (inclusive)."),
    end: datetime | None = Query(default=None, description="ISO-8601 upper bound (inclusive)."),
    sort: str = Query(default="desc", pattern="^(asc|desc)$"),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, gt=0, le=500),
) -> AuditEventsResponse:
    """List audit events, most recent first by default. Admin-only (compliance surface)."""
    events, total = container.audit_log.query(
        event_type=event_type, key_id=key_id, role=role, document_id=document_id,
        status=status, start=start, end=end, sort=sort, offset=offset, limit=limit,
    )
    return AuditEventsResponse(events=events, total=total, offset=offset, limit=limit)


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def diagnostics(
    container: Container = Depends(get_container),
    _identity: Identity = Depends(require_role("admin")),
) -> DiagnosticsResponse:
    """Aggregate operational state for on-call debugging. Admin-only: this
    exposes internal provider/config details that shouldn't be public, even
    with secrets scrubbed."""
    try:
        resolved_version = package_version(_PACKAGE_NAME)
    except PackageNotFoundError:
        resolved_version = _FALLBACK_VERSION

    summaries = container.chunk_store.get_document_summaries()
    checks = check_readiness(container)

    return DiagnosticsResponse(
        build={"name": _PACKAGE_NAME, "version": resolved_version},
        providers={
            "generation": container.generation_provider_name,
            "embedding": container.embedding_provider_name,
            "rerank_backend": container.settings.rerank_backend,
        },
        readiness=[ComponentStatus(**c) for c in checks],
        config=container.settings.safe_summary(),
        ingestion_stats={
            "total_documents": len(summaries),
            "total_chunks": sum(s["chunk_count"] for s in summaries),
        },
        audit_stats={"total_events": container.audit_log.count()},
        metrics=MetricsResponse(**container.metrics.snapshot()),
    )


@router.get("/version", response_model=VersionResponse)
async def get_version() -> VersionResponse:
    """Report the installed package name and version."""
    try:
        resolved_version = package_version(_PACKAGE_NAME)
    except PackageNotFoundError:
        resolved_version = _FALLBACK_VERSION
    return VersionResponse(name=_PACKAGE_NAME, version=resolved_version)
