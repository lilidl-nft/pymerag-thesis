"""
Re-ranker de chunks usando cross-encoder BGE-Reranker-v2-M3.

Aplica un modelo cross-encoder sobre los chunks recuperados por Qdrant
para reordenarlos por relevancia fina respecto a la consulta.

Incluye un fallback dummy que preserva el orden original por score
cuando FlagEmbedding no está disponible.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
"""Modelo cross-encoder para re-ranking."""

DEFAULT_TOP_K = 5
"""Número de chunks a retornar después del re-ranking."""


class Reranker:
    """Reordena chunks recuperados usando un cross-encoder.

    Utiliza bge-reranker-v2-m3 de FlagEmbedding para evaluar
    la relevancia precisa de cada chunk respecto a la consulta.
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        """Inicializa el re-ranker.

        Args:
            model_name: Nombre del modelo cross-encoder en HuggingFace Hub.
        """
        self._model_name = model_name
        self._model: Any = None
        self._loaded = False
        self._use_fallback = False

    @property
    def model(self) -> Any:
        """Instancia del modelo cross-encoder (carga lazy)."""
        if not self._loaded:
            self._load_model()
        return self._model

    @property
    def model_name(self) -> str:
        """Nombre del modelo en uso."""
        return self._model_name

    def _load_model(self) -> None:
        """Carga el modelo cross-encoder desde FlagEmbedding."""
        logger.info(
            "Cargando cross-encoder '%s'...",
            self._model_name,
        )
        try:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self._model_name,
                use_fp16=False,
            )
            self._loaded = True
            self._use_fallback = False
            logger.info("Cross-encoder cargado exitosamente.")
        except ImportError:
            logger.warning(
                "FlagEmbedding no disponible. Usando fallback de re-ranking. "
                "Instala FlagEmbedding con: pip install FlagEmbedding"
            )
            self._use_fallback = True
            self._loaded = True
        except Exception:
            logger.exception(
                "Error cargando cross-encoder '%s'. Usando fallback.",
                self._model_name,
            )
            self._use_fallback = True
            self._loaded = True

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """Reordena los chunks por relevancia a la consulta.

        Si el cross-encoder no está disponible, usa un fallback
        que mantiene el orden original (por score de recuperación).

        Args:
            query: Consulta del usuario.
            chunks: Lista de chunks recuperados, cada uno con al menos
                    'content' y opcionalmente 'score'.
            top_k: Número máximo de chunks a retornar.

        Returns:
            Lista de chunks reordenados, enriquecidos con 'rerank_score'.
            Máximo top_k elementos.
        """
        if not chunks:
            return []

        if self._use_fallback:
            return self._fallback_rerank(chunks, top_k)

        return self._cross_encode_rerank(query, chunks, top_k)

    def _cross_encode_rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Re-ranking usando el cross-encoder real.

        Args:
            query: Consulta del usuario.
            chunks: Lista de chunks a reordenar.
            top_k: Número máximo de resultados.

        Returns:
            Chunks reordenados con 'rerank_score'.
        """
        if not chunks:
            return []

        contents = [chunk.get("content", "") for chunk in chunks]

        try:
            # FlagReranker.compute_score acepta pares (query, passage)
            pairs = [[query, content] for content in contents]
            scores = self.model.compute_score(pairs, normalize=True)

            # compute_score puede retornar un solo float o una lista
            if isinstance(scores, float):
                scores = [scores]

            # Asignar scores y ordenar
            for chunk, score in zip(chunks, scores, strict=False):
                chunk["rerank_score"] = float(score)

            ranked = sorted(
                chunks,
                key=lambda c: c.get("rerank_score", 0.0),
                reverse=True,
            )
            logger.info(
                "Re-ranking completado: %d → %d chunks.",
                len(chunks),
                min(top_k, len(ranked)),
            )
            return ranked[:top_k]

        except Exception:
            logger.exception("Error en re-ranking cross-encoder. Usando fallback.")
            return self._fallback_rerank(chunks, top_k)

    @staticmethod
    def _fallback_rerank(
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fallback: ordena por el score original de recuperación.

        Args:
            chunks: Lista de chunks con 'score' opcional.
            top_k: Número máximo de resultados.

        Returns:
            Chunks ordenados por score descendente.
        """
        # Preservar score original como rerank_score si no existe
        for chunk in chunks:
            if "rerank_score" not in chunk:
                chunk["rerank_score"] = chunk.get("score", 0.0)

        ranked = sorted(
            chunks,
            key=lambda c: c.get("score", 0.0),
            reverse=True,
        )
        logger.debug(
            "Fallback re-ranking: %d → %d chunks (por score original).",
            len(chunks),
            min(top_k, len(ranked)),
        )
        return ranked[:top_k]


# ── Singleton ────────────────────────────────────────────────────────
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    """Retorna la instancia singleton del re-ranker."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
