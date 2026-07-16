"""FastAPI application entrypoint: wires the app-lifetime singletons and router."""

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import time

from api.dependencies import build_container, check_readiness
from api.routes import router
from rag_hybrid_search.config import Settings

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app, constructing all singletons once at startup.

    Passing ``settings`` explicitly (e.g. in tests) skips reading from the
    process environment; otherwise ``Settings()`` reads ``RAG_``-prefixed
    environment variables.
    """

    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = build_container(resolved_settings)
        app.state.container = container

        # Settings() already fails fast on missing/invalid config (see
        # config.py's model_validators); this additionally surfaces
        # Pinecone/BM25/audit reachability problems in the startup log
        # instead of silently waiting for the first request to hit them.
        for check in check_readiness(container):
            if not check["ok"]:
                logger.warning("startup readiness check failed: %s (%s)", check["name"], check["detail"])
            else:
                logger.info("startup readiness check ok: %s", check["name"])

        yield

        container.job_store.shutdown()

    app = FastAPI(
        title="rag-hybrid-search",
        description="Grounded hybrid search + retrieval-augmented generation API.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        """Attach a request_id to every request, client-supplied or generated;
        record operational metrics and one structured log line per request.

        Runs ahead of auth (api/auth.py reads request.state.request_id) so
        the id exists even for requests that get rejected by auth. Identity
        is only known after auth's dependency runs inside call_next, so it's
        read from request.state.identity (set by api/auth.get_identity)
        after the response comes back -- unauthenticated/rejected requests
        log with identity="-".
        """
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000
        response.headers["X-Request-ID"] = request_id

        container = getattr(request.app.state, "container", None)
        if container is not None:
            container.metrics.increment("total_requests")
            container.metrics.record_latency(duration_ms)
            if response.status_code >= 500:
                container.metrics.increment("failures")

        identity = getattr(request.state, "identity", None)
        logger.info(
            "request_id=%s endpoint=%s identity=%s latency_ms=%.2f status=%d",
            request_id, request.url.path,
            identity.key_id if identity else "-",
            duration_ms, response.status_code,
        )
        return response

    # No RAG_CORS_ALLOW_ORIGINS set -> no CORSMiddleware installed, same
    # same-origin-only behavior as before this setting existed.
    allowed_origins = resolved_settings.cors_allow_origins_list
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(router)

    if _FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
