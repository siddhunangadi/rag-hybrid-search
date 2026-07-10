"""FastAPI application entrypoint: wires the app-lifetime singletons and router."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.dependencies import build_container
from api.routes import router
from rag_hybrid_search.config import Settings

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


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

    if _FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

    return app


app = create_app()
