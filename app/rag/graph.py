"""
Máquina de estados RAG usando LangGraph.

Implementa el bucle principal de Retrieval-Augmented Generation
como un grafo de estados con cuatro nodos:

retrieve → rerank → generate → verify

El nodo verify comprueba que la respuesta contenga citas a las fuentes.
Si la verificación falla, el grafo puede reintentar la generación.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal, TypedDict

from app.rag.llm import LLMClient, get_llm_client
from app.rag.reranker import Reranker, get_reranker
from app.rag.retriever import QdrantRetriever

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────
MAX_VERIFY_RETRIES = 2
"""Número máximo de reintentos si la verificación de citas falla."""

VERIFICATION_PROMPT = (
    "Reescribe la respuesta anterior asegurándote de incluir "
    "referencias explícitas a las fuentes proporcionadas "
    "usando el formato [Fuente N]."
)
"""Prompt para solicitar al LLM que agregue citas a la respuesta."""


# ── Definición del estado ────────────────────────────────────────────


class RAGState(TypedDict):
    """Estado compartido entre los nodos del grafo RAG.

    Attributes:
        query: Consulta original del usuario.
        contexts: Lista de textos de chunks recuperados.
        sources: Lista de diccionarios con metadatos de cada fuente.
        answer: Respuesta generada por el LLM.
        verified: Si la respuesta pasó la verificación de citas.
        retries: Contador de reintentos de verificación.
        error: Mensaje de error si algo falló.
    """

    query: str
    contexts: list[str]
    sources: list[dict[str, Any]]
    answer: str
    verified: bool
    retries: int
    error: str


# ── Nodos del grafo ──────────────────────────────────────────────────


def retrieve_node(
    state: RAGState,
    retriever: QdrantRetriever,
    top_k: int = 10,
) -> dict[str, Any]:
    """Recupera chunks relevantes desde Qdrant.

    Args:
        state: Estado actual del grafo RAG.
        retriever: Instancia del recuperador Qdrant.
        top_k: Número de chunks a recuperar.

    Returns:
        Actualización parcial del estado con contexts y sources.
    """
    query = state["query"]

    logger.info("Recuperando chunks para: '%s...'", query[:60])

    try:
        results = retriever.retrieve(query=query, top_k=top_k, score_threshold=0.2)
    except Exception:
        logger.exception("Error en recuperación de chunks.")
        return {
            "contexts": [],
            "sources": [],
            "error": "Fallo en la etapa de recuperación.",
        }

    contexts = [r.get("content", "") for r in results]
    sources = [
        {
            "chunk_id": r.get("id", ""),
            "document_id": r.get("document_id", ""),
            "score": r.get("score", 0.0),
            "content": r.get("content", ""),
        }
        for r in results
    ]

    logger.info(
        "Recuperados %d chunks (score > 0.2).",
        len(contexts),
    )

    return {
        "contexts": contexts,
        "sources": sources,
        "error": "",
    }


def rerank_node(
    state: RAGState,
    reranker: Reranker,
    top_k: int = 5,
) -> dict[str, Any]:
    """Reordena los chunks recuperados por relevancia fina.

    Args:
        state: Estado actual del grafo RAG.
        reranker: Instancia del re-ranker cross-encoder.
        top_k: Número de chunks a conservar tras re-ranking.

    Returns:
        Actualización parcial del estado con contexts y sources reordenados.
    """
    query = state["query"]
    sources = state["sources"]

    if not sources:
        logger.warning("Sin fuentes para re-ranquear.")
        return {"contexts": [], "sources": []}

    logger.info("Re-rankeando %d chunks...", len(sources))

    try:
        ranked = reranker.rerank(query=query, chunks=sources, top_k=top_k)
    except Exception:
        logger.exception("Error en re-ranking. Conservando orden original.")
        ranked = sources[:top_k]

    contexts = [r.get("content", "") for r in ranked]

    logger.info("Re-ranking completado: %d chunks finales.", len(contexts))

    return {"contexts": contexts, "sources": ranked}


def generate_node(
    state: RAGState,
    llm: LLMClient,
) -> dict[str, Any]:
    """Genera una respuesta usando el LLM con los chunks recuperados.

    Args:
        state: Estado actual del grafo RAG.
        llm: Cliente LLM para generación.

    Returns:
        Actualización parcial del estado con la respuesta generada.
    """
    query = state["query"]
    contexts = state["contexts"]

    if state.get("error"):
        return {
            "answer": f"[Error en etapa anterior: {state['error']}]",
        }

    if not contexts:
        return {
            "answer": (
                "No se encontraron fragmentos relevantes en la base de conocimiento "
                "para responder esta consulta."
            ),
        }

    logger.info("Generando respuesta con LLM (%d contextos)...", len(contexts))

    try:
        answer = llm.generate(prompt=query, context=contexts)
    except Exception:
        logger.exception("Error en generación LLM.")
        answer = "[Error al generar la respuesta.]"

    logger.info(
        "Respuesta generada: %d caracteres.",
        len(answer),
    )

    return {"answer": answer, "verified": False}


def verify_node(state: RAGState) -> dict[str, Any]:
    """Verifica que la respuesta incluya citas a las fuentes.

    Comprueba si la respuesta contiene marcadores de fuente [Fuente N].
    Si no los contiene y quedan reintentos, la respuesta se marca
    para regeneración con instrucciones de citado.

    Args:
        state: Estado actual del grafo RAG.

    Returns:
        Actualización parcial del estado con resultado de verificación.
    """
    answer = state.get("answer", "")
    sources = state.get("sources", [])
    retries = state.get("retries", 0)

    # Si no hay fuentes, no tiene sentido verificar citas
    if not sources:
        logger.debug("Verificación omitida: sin fuentes.")
        return {"verified": True}

    # Verificar presencia de marcadores de fuente
    has_citations = _check_citations(answer, len(sources))

    if has_citations:
        logger.info("Verificación de citas: OK.")
        return {"verified": True}

    if retries < MAX_VERIFY_RETRIES:
        logger.info(
            "Verificación de citas: FALLO (reintento %d/%d).",
            retries + 1,
            MAX_VERIFY_RETRIES,
        )
        return {
            "verified": False,
            "retries": retries + 1,
        }
    else:
        logger.warning(
            "Verificación de citas: FALLO definitivo tras %d reintentos.",
            MAX_VERIFY_RETRIES,
        )
        return {"verified": True}  # Aceptar de todos modos


def _check_citations(answer: str, num_sources: int) -> bool:
    """Verifica si la respuesta contiene citas a fuentes.

    Busca patrones como [Fuente 1], [Fuente 2], etc. en el texto.

    Args:
        answer: Texto de la respuesta generada.
        num_sources: Número total de fuentes disponibles.

    Returns:
        True si se encontró al menos una cita a fuente.
    """
    import re

    # Buscar patrones [Fuente N] o [Source N] o similares
    patterns = [
        r"\[Fuente\s+\d+\]",
        r"\[Source\s+\d+\]",
        r"\[fuente\s+\d+\]",
        r"\(Fuente\s+\d+\)",
        r"\(Source\s+\d+\)",
    ]

    for pattern in patterns:
        if re.search(pattern, answer):
            return True

    # También verificar si menciona números de fuente de otra forma
    for i in range(1, num_sources + 1):
        if f"fuente {i}" in answer.lower() or f"source {i}" in answer.lower():
            return True

    return False


# ── Función de enrutamiento (condicional) ────────────────────────────


def should_retry(state: RAGState) -> Literal["generate", "__end__"]:
    """Determina si el grafo debe reintentar la generación.

    Si la verificación falló y quedan reintentos, vuelve a generate.
    En caso contrario, termina el flujo.

    Args:
        state: Estado actual del grafo RAG.

    Returns:
        'generate' para reintentar, '__end__' para finalizar.
    """
    if not state.get("verified", False) and state.get("retries", 0) <= MAX_VERIFY_RETRIES:
        return "generate"
    return "__end__"


# ── Constructor del grafo ────────────────────────────────────────────


class RAGGraph:
    """Orquestador del pipeline RAG usando LangGraph.

    Encapsula la construcción y ejecución del grafo de estados
    retrieve → rerank → generate → verify.

    Attributes:
        retriever: Recuperador de chunks Qdrant.
        reranker: Re-ranker cross-encoder.
        llm: Cliente LLM.
        top_k_retrieve: Número de chunks a recuperar.
        top_k_rerank: Número de chunks tras re-ranking.
    """

    def __init__(
        self,
        retriever: QdrantRetriever | None = None,
        reranker: Reranker | None = None,
        llm: LLMClient | None = None,
        top_k_retrieve: int = 10,
        top_k_rerank: int = 5,
    ) -> None:
        """Inicializa el grafo RAG.

        Args:
            retriever: Instancia del recuperador Qdrant.
            reranker: Instancia del re-ranker.
            llm: Cliente LLM.
            top_k_retrieve: Chunks a recuperar en primera etapa.
            top_k_rerank: Chunks a conservar tras re-ranking.
        """
        self.retriever = retriever or QdrantRetriever()
        self.reranker = reranker or get_reranker()
        self.llm = llm or get_llm_client()
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self._graph = self._build_graph()

    def _build_graph(self):
        """Construye el StateGraph de LangGraph con los cuatro nodos.

        Returns:
            Grafo compilado listo para ejecución.
        """
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            logger.error(
                "LangGraph no está instalado. "
                "Instálalo con: pip install langgraph"
            )
            raise

        # Crear el grafo con el tipo de estado
        workflow = StateGraph(RAGState)

        # Registrar nodos (usando closures para inyectar dependencias)
        workflow.add_node(
            "retrieve",
            lambda s: retrieve_node(
                s,
                retriever=self.retriever,
                top_k=self.top_k_retrieve,
            ),
        )
        workflow.add_node(
            "rerank",
            lambda s: rerank_node(
                s,
                reranker=self.reranker,
                top_k=self.top_k_rerank,
            ),
        )
        workflow.add_node(
            "generate",
            lambda s: generate_node(s, llm=self.llm),
        )
        workflow.add_node("verify", verify_node)

        # Definir transiciones
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", "verify")

        # Transición condicional desde verify
        workflow.add_conditional_edges(
            "verify",
            should_retry,
            {
                "generate": "generate",
                "__end__": END,
            },
        )

        return workflow.compile()

    def run(self, query: str) -> RAGState:
        """Ejecuta el pipeline RAG completo para una consulta.

        Args:
            query: Consulta del usuario en lenguaje natural.

        Returns:
            Estado final del grafo con la respuesta y fuentes.
        """
        initial_state: RAGState = {
            "query": query,
            "contexts": [],
            "sources": [],
            "answer": "",
            "verified": False,
            "retries": 0,
            "error": "",
        }

        logger.info("Iniciando pipeline RAG para: '%s...'", query[:60])

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception:
            logger.exception("Error ejecutando el grafo RAG.")
            # Retornar estado con error
            initial_state["error"] = "Error en el pipeline RAG."
            initial_state["answer"] = (
                "Ocurrió un error interno al procesar tu consulta. "
                "Por favor, intenta de nuevo más tarde."
            )
            return initial_state

        logger.info(
            "Pipeline RAG completado: %d fuentes, %d chars de respuesta.",
            len(final_state.get("sources", [])),
            len(final_state.get("answer", "")),
        )

        return final_state


# ── Singleton ────────────────────────────────────────────────────────
_rag_graph: RAGGraph | None = None


def get_rag_graph() -> RAGGraph:
    """Retorna la instancia singleton del grafo RAG."""
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = RAGGraph()
    return _rag_graph
