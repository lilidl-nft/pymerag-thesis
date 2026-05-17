"""
Rutas de la API para consultas RAG.

POST /query — Ejecutar una consulta RAG con generación de respuesta.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import settings
from app.models.sql import AuditLog
from app.rag.embeddings import get_embedder
from app.rag.retriever import QdrantIndexer, QdrantRetriever
from sqlmodel import Session, create_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query")

# ── Clientes ───────────────────────────────────────────────────────
_retriever: Optional[QdrantRetriever] = None


def _get_retriever() -> QdrantRetriever:
    """Retorna una instancia singleton del recuperador Qdrant."""
    global _retriever
    if _retriever is None:
        indexer = QdrantIndexer()
        embedder = get_embedder()
        _retriever = QdrantRetriever(indexer=indexer, embedder=embedder)
    return _retriever


# ── Modelos de request/response ───────────────────────────────────


class QueryRequest(BaseModel):
    """Cuerpo de la solicitud de consulta RAG.

    Attributes:
        query: Texto de la consulta en lenguaje natural.
        stream: Si es True, la respuesta se transmite como SSE.
        top_k: Número de chunks a recuperar para el contexto.
        filters: Filtros opcionales para restringir la búsqueda.
    """

    query: str = Field(..., min_length=1, description="Consulta en lenguaje natural.")
    stream: bool = Field(
        default=False,
        description="Transmitir respuesta como Server-Sent Events.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Número de chunks a recuperar.",
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Filtros opcionales (metadata, document_id, etc.).",
    )


class Source(BaseModel):
    """Fuente de información recuperada.

    Attributes:
        chunk_id: Identificador del chunk (ID de Qdrant).
        content: Contenido textual del chunk.
        metadata: Metadatos asociados al chunk.
        score: Relevancia del chunk (0.0 a 1.0).
    """

    chunk_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Respuesta a una consulta RAG.

    Attributes:
        answer: Respuesta generada por el LLM.
        sources: Lista de fuentes utilizadas para la respuesta.
        metadata: Metadatos de la consulta (latencia, etc.).
    """

    answer: str
    sources: list[Source] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Funciones auxiliares ──────────────────────────────────────────


def _build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """Construye el prompt para el LLM a partir de la consulta y los chunks.

    Args:
        query: Consulta del usuario.
        chunks: Chunks recuperados de Qdrant.

    Returns:
        Prompt formateado para el LLM.
    """
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        doc_id = chunk.get("document_id", "unknown")
        context_parts.append(f"[Fuente {i} | doc:{doc_id}]\n{content}")

    context_text = "\n\n".join(context_parts)

    return (
        "Eres un asistente de investigación especializado en análisis documental. "
        "Responde la pregunta del usuario basándote EXCLUSIVAMENTE en el contexto "
        "proporcionado. Si la información no está en el contexto, indícalo "
        "claramente. Cita las fuentes utilizadas.\n\n"
        f"## Contexto\n{context_text}\n\n"
        f"## Pregunta\n{query}\n\n"
        "## Respuesta\n"
    )


async def _generate_answer(
    prompt: str,
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """Genera una respuesta usando el servidor LLM compatible con OpenAI.

    Args:
        prompt: Prompt completo para el LLM.
        client: Cliente HTTP asíncrono (crea uno si es None).

    Returns:
        Texto de la respuesta generada.
    """
    close_client = client is None
    if client is None:
        client = httpx.AsyncClient(timeout=60.0)

    try:
        response = await client.post(
            f"{settings.llm_api_base}/chat/completions",
            json={
                "model": "deepseek-v4",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1024,
                "temperature": 0.3,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except httpx.HTTPError as exc:
        logger.warning("Error llamando al LLM: %s. Usando fallback.", exc)
        return (
            "No se pudo conectar con el LLM. "
            "A continuación se muestran los fragmentos recuperados:\n\n"
            + prompt.split("## Contexto\n")[-1].split("## Pregunta\n")[0]
        )
    except Exception as exc:
        logger.exception("Error inesperado en generación LLM: %s", exc)
        return f"[Error generando respuesta: {exc}]"
    finally:
        if close_client:
            await client.aclose()


def _audit_query(query: str, answer: str, latency: float) -> None:
    """Registra la consulta en la tabla de auditoría.

    Args:
        query: Texto de la consulta.
        answer: Respuesta generada.
        latency: Latencia de la consulta en segundos.
    """
    try:
        engine = create_engine(
            settings.database_url,
            echo=False,
            connect_args={"check_same_thread": False}
            if "sqlite" in settings.database_url
            else {},
        )
        with Session(engine) as session:
            log_entry = AuditLog(
                action="QUERY_EXECUTE",
                user_id="api",
                payload={
                    "query": query[:500],
                    "answer_preview": answer[:200],
                    "latency": latency,
                },
                timestamp=datetime.now(timezone.utc),
            )
            session.add(log_entry)
            session.commit()
    except Exception as exc:
        logger.debug("No se pudo registrar auditoría: %s", exc)


# ── Endpoints ─────────────────────────────────────────────────────


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ejecutar consulta RAG",
    description=(
        "Realiza una búsqueda híbrida (semántica + léxica) sobre el corpus "
        "indexado y genera una respuesta basada en los fragmentos recuperados."
    ),
)
async def execute_query(request: QueryRequest) -> QueryResponse:
    """Ejecuta una consulta RAG completa: recuperación + generación.

    Args:
        request: Datos de la consulta.

    Returns:
        Respuesta generada con fuentes y metadatos.
    """
    start_time = time.perf_counter()

    # ── 1. Recuperación ──────────────────────────────────────────
    retriever = _get_retriever()
    filters = request.filters if request.filters else None
    chunks = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        score_threshold=0.3,
        filters=filters,
    )

    # ── 2. Construir prompt ──────────────────────────────────────
    prompt = _build_prompt(request.query, chunks)

    # ── 3. Generar respuesta ─────────────────────────────────────
    answer = await _generate_answer(prompt)

    latency = time.perf_counter() - start_time

    # ── 4. Construir fuentes ─────────────────────────────────────
    sources = [
        Source(
            chunk_id=chunk.get("id", ""),
            content=chunk.get("content", ""),
            metadata=chunk.get("metadata", {}),
            score=chunk.get("score", 0.0),
        )
        for chunk in chunks
    ]

    # ── 5. Auditoría (fire-and-forget, no bloquea la respuesta) ──
    _audit_query(query=request.query, answer=answer, latency=latency)

    logger.info(
        "Consulta completada en %.2fs: '%s...' → %d fuentes, %d chars",
        latency,
        request.query[:50],
        len(sources),
        len(answer),
    )

    return QueryResponse(
        answer=answer,
        sources=sources,
        metadata={
            "latency": round(latency, 3),
            "sources_count": len(sources),
            "model": "deepseek-v4",
        },
    )
