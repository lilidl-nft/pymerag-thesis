"""
Punto de entrada principal de la aplicación FastAPI de Pymerag.

Crea la instancia de FastAPI, registra los routers de la API v1,
configura CORS, middlewares y eventos de ciclo de vida.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_admin import router as admin_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.api.routes_topics import router as topics_router
from app.core.config import settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Maneja el ciclo de vida de la aplicación.

    Al iniciar:
    - Configura logging.
    - Crea directorios necesarios.
    - Inicializa conexiones a servicios externos.

    Al detener:
    - Cierra conexiones y libera recursos.
    """
    # ── Startup ───────────────────────────────────────────────────
    setup_logging()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Pymerag API iniciando en http://%s:%s",
        settings.api_host,
        settings.api_port,
    )
    logger.info("Base de datos: %s", settings.database_url)
    logger.info("Qdrant: %s:%s", settings.qdrant_host, settings.qdrant_port)
    logger.info("LLM API: %s", settings.llm_api_base)

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    logger.info("Pymerag API detenida.")


app = FastAPI(
    title="Pymerag API",
    description="Asistente Inteligente RAG-MCP para Gestión Documental",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────
API_V1_PREFIX = "/api/v1"

app.include_router(ingest_router, prefix=API_V1_PREFIX, tags=["Ingest"])
app.include_router(query_router, prefix=API_V1_PREFIX, tags=["Query"])
app.include_router(topics_router, prefix=API_V1_PREFIX, tags=["Topics"])
app.include_router(admin_router, prefix=API_V1_PREFIX, tags=["Admin"])


@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    """Endpoint raíz — devuelve metadatos del servicio."""
    return {
        "service": "Pymerag API",
        "version": "0.1.0",
        "docs": f"{API_V1_PREFIX}/docs",
    }
