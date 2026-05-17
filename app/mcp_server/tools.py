"""
Herramientas MCP (Model Context Protocol) para el servidor Pymerag.

Define 4 herramientas asíncronas que los agentes LLM pueden invocar
a través del protocolo MCP para realizar búsqueda semántica, resumen,
comparación de documentos y generación de respuestas con evidencia.

Cada herramienta utiliza los servicios backend existentes:
- QdrantRetriever: búsqueda híbrida (densa + dispersa) sobre Qdrant.
- RAGGraph: pipeline RAG completo (retrieve → rerank → generate → verify).
- LLMClient: generación de texto con el modelo de lenguaje.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.rag.embeddings import get_embedder
from app.rag.llm import get_llm_client
from app.rag.reranker import get_reranker
from app.rag.retriever import QdrantIndexer, QdrantRetriever

logger = logging.getLogger(__name__)

# ── Singletons compartidos ────────────────────────────────────────────
_retriever: QdrantRetriever | None = None


def _get_retriever() -> QdrantRetriever:
    """Retorna una instancia singleton del recuperador Qdrant."""
    global _retriever
    if _retriever is None:
        indexer = QdrantIndexer()
        embedder = get_embedder()
        _retriever = QdrantRetriever(indexer=indexer, embedder=embedder)
    return _retriever


# ═══════════════════════════════════════════════════════════════════════
# Modelos de entrada Pydantic para cada herramienta
# ═══════════════════════════════════════════════════════════════════════


class SearchInput(BaseModel):
    """Parámetros para la herramienta de búsqueda semántica.

    Attributes:
        query: Texto de la consulta en lenguaje natural.
        top_k: Número máximo de resultados a retornar (1-50).
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Consulta en lenguaje natural para buscar en el corpus.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Número máximo de fragmentos a recuperar.",
    )


class SummarizeInput(BaseModel):
    """Parámetros para la herramienta de resumen de documento.

    Attributes:
        document_id: Identificador del documento a resumir.
    """

    document_id: str = Field(
        ...,
        min_length=1,
        description="ID del documento del cual generar un resumen.",
    )


class CompareInput(BaseModel):
    """Parámetros para la herramienta de comparación de documentos.

    Attributes:
        doc_ids: Lista de identificadores de documentos a comparar (2 o más).
    """

    doc_ids: list[str] = Field(
        ...,
        min_length=2,
        description="Lista de IDs de documentos a comparar (mínimo 2).",
    )


class GenerateWithEvidenceInput(BaseModel):
    """Parámetros para la herramienta de generación con evidencia.

    Attributes:
        query: Consulta o pregunta del usuario.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Pregunta o consulta a responder con evidencia del corpus.",
    )


# ═══════════════════════════════════════════════════════════════════════
# Herramientas MCP
# ═══════════════════════════════════════════════════════════════════════


async def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Realiza una búsqueda híbrida (semántica + léxica) sobre el corpus indexado.

    Utiliza QdrantRetriever para buscar los fragmentos más relevantes
    combinando embeddings densos (BGE-M3) y dispersos (BM25 lexical).

    Args:
        query: Texto de la consulta en lenguaje natural.
        top_k: Número máximo de resultados a retornar.

    Returns:
        Lista de fragmentos recuperados, cada uno con:
        - id: Identificador del fragmento en Qdrant.
        - content: Texto del fragmento.
        - score: Relevancia (0.0 a 1.0).
        - document_id: ID del documento de origen.
        - metadata: Metadatos adicionales del fragmento.
    """
    logger.info(
        "MCP tool 'search': query='%s...' top_k=%d",
        query[:80],
        top_k,
    )

    retriever = _get_retriever()
    results = retriever.retrieve(query=query, top_k=top_k, score_threshold=0.2)

    # Formatear resultados para el agente LLM
    formatted: list[dict[str, Any]] = []
    for chunk in results:
        formatted.append(
            {
                "id": chunk.get("id", ""),
                "content": chunk.get("content", ""),
                "score": round(chunk.get("score", 0.0), 4),
                "document_id": chunk.get("document_id", ""),
                "chunk_index": chunk.get("chunk_index", -1),
                "language": chunk.get("language", ""),
                "metadata": chunk.get("metadata", {}),
            }
        )

    logger.info(
        "MCP tool 'search': %d resultados encontrados.",
        len(formatted),
    )
    return formatted


async def summarize(document_id: str) -> dict[str, Any]:
    """Genera un resumen de un documento específico usando sus fragmentos indexados.

    Recupera todos los fragmentos del documento, los ordena por chunk_index
    y genera un resumen usando el LLM configurado.

    Args:
        document_id: Identificador único del documento a resumir.

    Returns:
        Diccionario con:
        - document_id: ID del documento.
        - summary: Texto del resumen generado.
        - num_chunks: Número de fragmentos que componen el documento.
        - sources: Lista de fragmentos usados como fuentes.
    """
    logger.info(
        "MCP tool 'summarize': document_id='%s'",
        document_id,
    )

    retriever = _get_retriever()
    documents = retriever.retrieve(
        query="*",  # query comodín para recuperar por filtro
        top_k=100,  # Suficiente para documentos grandes
        score_threshold=0.0,
        filters={"document_id": document_id},
    )

    if not documents:
        logger.warning(
            "MCP tool 'summarize': documento '%s' no encontrado.",
            document_id,
        )
        return {
            "document_id": document_id,
            "summary": "",
            "num_chunks": 0,
            "sources": [],
            "error": f"Documento '{document_id}' no encontrado en el índice.",
        }

    # Ordenar fragmentos por chunk_index para preservar el orden original
    sorted_docs = sorted(
        documents,
        key=lambda d: d.get("chunk_index", 0),
    )

    # Concatenar contenido para el resumen
    full_text = "\n\n".join(
        d.get("content", "") for d in sorted_docs
    )

    # Generar resumen con el LLM
    llm = get_llm_client()
    summary_prompt = (
        "Genera un resumen conciso y estructurado del siguiente documento. "
        "Incluye los puntos principales, hallazgos clave y conclusiones. "
        "El resumen debe ser en español y no exceder 500 palabras.\n\n"
        f"Documento:\n{full_text[:8000]}"  # Truncar para no exceder contexto
    )

    try:
        summary = llm.generate(prompt=summary_prompt)
    except Exception as exc:
        logger.exception("Error generando resumen con LLM: %s", exc)
        summary = (
            f"[Error al generar el resumen: {exc}]\n\n"
            "Fragmentos del documento:\n\n" + full_text[:2000]
        )

    logger.info(
        "MCP tool 'summarize': resumen generado (%d chars) de %d fragmentos.",
        len(summary),
        len(sorted_docs),
    )

    return {
        "document_id": document_id,
        "summary": summary,
        "num_chunks": len(sorted_docs),
        "sources": [
            {
                "chunk_id": d.get("id", ""),
                "chunk_index": d.get("chunk_index", -1),
                "score": round(d.get("score", 0.0), 4),
            }
            for d in sorted_docs
        ],
    }


async def compare(doc_ids: list[str]) -> dict[str, Any]:
    """Compara dos o más documentos, identificando similitudes y diferencias.

    Recupera los fragmentos de cada documento, genera resúmenes individuales
    y luego pide al LLM que compare los contenidos.

    Args:
        doc_ids: Lista de identificadores de documentos a comparar (mínimo 2).

    Returns:
        Diccionario con:
        - documents: Lista de resúmenes individuales por documento.
        - comparison: Texto del análisis comparativo generado por el LLM.
        - doc_ids: Lista de IDs comparados.
    """
    logger.info(
        "MCP tool 'compare': %d documentos a comparar.",
        len(doc_ids),
    )

    if len(doc_ids) < 2:
        return {
            "comparison": "",
            "documents": [],
            "doc_ids": doc_ids,
            "error": "Se requieren al menos 2 documentos para comparar.",
        }

    retriever = _get_retriever()

    # Recuperar fragmentos de cada documento
    doc_summaries: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        chunks = retriever.retrieve(
            query="*",
            top_k=50,
            score_threshold=0.0,
            filters={"document_id": doc_id},
        )

        if not chunks:
            doc_summaries.append(
                {
                    "document_id": doc_id,
                    "content": f"[Documento '{doc_id}' no encontrado]",
                    "num_chunks": 0,
                }
            )
            continue

        sorted_chunks = sorted(
            chunks,
            key=lambda d: d.get("chunk_index", 0),
        )
        content = "\n\n".join(
            c.get("content", "") for c in sorted_chunks
        )
        doc_summaries.append(
            {
                "document_id": doc_id,
                "content": content[:4000],  # Truncar por doc
                "num_chunks": len(sorted_chunks),
            }
        )

    # Construir prompt de comparación para el LLM
    docs_text = ""
    for i, ds in enumerate(doc_summaries, 1):
        docs_text += (
            f"\n### Documento {i}: {ds['document_id']}\n"
            f"{ds['content']}\n"
        )

    compare_prompt = (
        "Compara los siguientes documentos e identifica:\n"
        "1. Temas y conceptos en común.\n"
        "2. Diferencias clave en contenido, enfoque o conclusiones.\n"
        "3. Relaciones o complementariedades entre los documentos.\n"
        "4. Una tabla comparativa resumen.\n\n"
        f"{docs_text}\n\n"
        "Análisis comparativo:"
    )

    llm = get_llm_client()
    try:
        comparison = llm.generate(prompt=compare_prompt, max_tokens=1500)
    except Exception as exc:
        logger.exception("Error generando comparación con LLM: %s", exc)
        comparison = f"[Error al generar la comparación: {exc}]"

    logger.info(
        "MCP tool 'compare': análisis completado (%d chars).",
        len(comparison),
    )

    return {
        "documents": [
            {
                "document_id": ds["document_id"],
                "num_chunks": ds["num_chunks"],
                "preview": ds["content"][:500] + "..."
                if len(ds["content"]) > 500
                else ds["content"],
            }
            for ds in doc_summaries
        ],
        "comparison": comparison,
        "doc_ids": doc_ids,
    }


async def generate_with_evidence(query: str) -> dict[str, Any]:
    """Genera una respuesta fundamentada con evidencia del corpus documental.

    Ejecuta el pipeline RAG completo: recuperación de fragmentos relevantes,
    re-ranking por relevancia fina, y generación de respuesta con citas
    a las fuentes. La respuesta incluye los fragmentos de evidencia.

    Args:
        query: Pregunta o consulta a responder con evidencia.

    Returns:
        Diccionario con:
        - answer: Respuesta generada por el LLM con citas a fuentes.
        - sources: Lista de fragmentos usados como evidencia.
        - query: Consulta original (eco).
    """
    logger.info(
        "MCP tool 'generate_with_evidence': query='%s...'",
        query[:80],
    )

    # Usar el grafo RAG completo para obtener respuesta con verificación de citas
    try:
        from app.rag.graph import RAGGraph

        graph = RAGGraph(
            retriever=_get_retriever(),
            reranker=get_reranker(),
            llm=get_llm_client(),
            top_k_retrieve=10,
            top_k_rerank=5,
        )
        state = graph.run(query)
        answer = state.get("answer", "")
        # Usar las sources ya enriquecidas por el grafo
        sources_raw = state.get("sources", [])

    except ImportError:
        # Fallback sin LangGraph: recuperación directa + generación simple
        logger.warning(
            "LangGraph no disponible. Usando fallback directo para "
            "generate_with_evidence."
        )
        retriever = _get_retriever()
        chunks = retriever.retrieve(query=query, top_k=5, score_threshold=0.3)

        if not chunks:
            return {
                "answer": (
                    "No se encontró evidencia suficiente en el corpus "
                    "para responder esta consulta."
                ),
                "sources": [],
                "query": query,
            }

        # Construir contexto para el LLM
        context_parts: list[str] = []
        sources_raw: list[dict[str, Any]] = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            doc_id = chunk.get("document_id", "unknown")
            context_parts.append(f"[Fuente {i} | doc:{doc_id}]\n{content}")
            sources_raw.append(
                {
                    "chunk_id": chunk.get("id", ""),
                    "document_id": doc_id,
                    "score": round(chunk.get("score", 0.0), 4),
                    "content": content[:300] + "..."
                    if len(content) > 300
                    else content,
                }
            )

        context_text = "\n\n".join(context_parts)
        llm = get_llm_client()
        try:
            answer = llm.generate(
                prompt=(
                    f"Responde la siguiente pregunta basándote EXCLUSIVAMENTE "
                    f"en el contexto proporcionado. Cita las fuentes usando "
                    f"[Fuente N].\n\n"
                    f"## Contexto\n{context_text}\n\n"
                    f"## Pregunta\n{query}"
                ),
            )
        except Exception:
            answer = "Error al generar la respuesta con el LLM."
            logger.exception("Error en fallback generate_with_evidence.")

    except Exception:
        logger.exception("Error en pipeline RAG para generate_with_evidence.")
        answer = "Error interno al procesar la consulta."
        sources_raw = []

    # Formatear fuentes para la respuesta
    sources = [
        {
            "chunk_id": s.get("chunk_id", s.get("id", "")),
            "document_id": s.get("document_id", ""),
            "score": round(s.get("score", s.get("rerank_score", 0.0)), 4),
            "preview": (
                s.get("content", "")[:300] + "..."
                if len(s.get("content", "")) > 300
                else s.get("content", "")
            ),
        }
        for s in sources_raw
    ]

    logger.info(
        "MCP tool 'generate_with_evidence': respuesta de %d chars, %d fuentes.",
        len(answer),
        len(sources),
    )

    return {
        "answer": answer,
        "sources": sources,
        "query": query,
    }
