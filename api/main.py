"""FastAPI application entrypoint: wires the app-lifetime singletons and router."""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.dependencies import build_container
from api.routes import router
from rag_hybrid_search.config import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app, constructing all singletons once at startup.

    Passing ``settings`` explicitly (e.g. in tests) skips reading from the
    process environment; otherwise ``Settings()`` reads ``RAG_``-prefixed
    environment variables.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = build_container(settings)
        yield

    app = FastAPI(
        title="rag-hybrid-search",
        description="Grounded hybrid search + retrieval-augmented generation API.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
