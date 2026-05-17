"""
Módulo de embeddings híbridos usando BGE-M3.

BGE-M3 genera tres tipos de representaciones:
- Dense vectors (1024-dim): para búsqueda semántica.
- Sparse / lexical weights: para búsqueda por palabras clave (BM25-like).
- ColBERT vectors: para re-ranking fino (opcional).

Este módulo expone los vectores densos y dispersos para
la búsqueda híbrida en Qdrant.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qdrant_client.models import SparseVector

from app.core.config import settings

logger = logging.getLogger(__name__)


class HybridEmbedder:
    """Genera embeddings densos y dispersos usando BGE-M3.

    Encapsula el modelo FlagEmbedding y normaliza la salida
    para la API de Qdrant.
    """

    def __init__(self) -> None:
        """Carga el modelo BGE-M3 usando FlagEmbedding."""
        self._model: Any = None
        self._model_name = settings.embedding_model
        self._device = settings.embedding_device
        self._dense_dim = settings.embedding_dense_dim
        self._loaded = False

    @property
    def model(self) -> Any:
        """Instancia del modelo (carga lazy)."""
        if not self._loaded:
            self._load_model()
        return self._model

    @property
    def dense_dim(self) -> int:
        """Dimensionalidad del vector denso."""
        return self._dense_dim

    @property
    def model_name(self) -> str:
        """Nombre del modelo en uso."""
        return self._model_name

    def _load_model(self) -> None:
        """Carga el modelo BGE-M3 desde FlagEmbedding."""
        logger.info(
            "Cargando modelo BGE-M3 '%s' en %s...",
            self._model_name,
            self._device,
        )
        try:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self._model_name,
                use_fp16=self._device == "cuda",
                device=self._device,
            )
            self._loaded = True
            logger.info("Modelo BGE-M3 cargado exitosamente.")
        except ImportError:
            logger.warning(
                "FlagEmbedding no disponible. Usando fallback dummy. "
                "Instala FlagEmbedding con: pip install FlagEmbedding"
            )
            self._model = _DummyBGE()
            self._loaded = True

    def encode(
        self,
        texts: str | list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = True,
        batch_size: int = 32,
        max_length: int = 8192,
    ) -> dict[str, Any]:
        """Genera embeddings densos y dispersos para los textos.

        Args:
            texts: Texto(s) a codificar.
            return_dense: Si se deben retornar vectores densos.
            return_sparse: Si se deben retornar vectores dispersos.
            batch_size: Tamaño del lote para procesamiento.
            max_length: Longitud máxima de tokens.

        Returns:
            Diccionario con 'dense_vecs' (np.ndarray) y
            'lexical_weights' (list[dict]).
        """
        is_single = isinstance(texts, str)
        sentences = [texts] if is_single else texts

        result = self.model.encode(
            sentences,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        if is_single:
            if return_dense and "dense_vecs" in result:
                result["dense_vecs"] = result["dense_vecs"][0]
            if return_sparse and "lexical_weights" in result:
                result["lexical_weights"] = result["lexical_weights"][0]

        return result

    def encode_dense(self, texts: str | list[str], **kwargs: Any) -> np.ndarray:
        """Retorna solo los vectores densos.

        Args:
            texts: Texto(s) a codificar.
            **kwargs: Argumentos adicionales para encode().

        Returns:
            Array numpy de shape (n_texts, 1024) o (1024,).
        """
        result = self.encode(texts, return_dense=True, return_sparse=False, **kwargs)
        return result["dense_vecs"]

    def encode_sparse(self, texts: str | list[str], **kwargs: Any) -> list[dict]:
        """Retorna solo los pesos léxicos (sparse).

        Args:
            texts: Texto(s) a codificar.
            **kwargs: Argumentos adicionales para encode().

        Returns:
            Lista de diccionarios {token_id: weight}.
        """
        result = self.encode(
            texts, return_dense=False, return_sparse=True, **kwargs
        )
        return result["lexical_weights"]

    @staticmethod
    def to_qdrant_sparse(lexical_weights: dict[int, float]) -> SparseVector:
        """Convierte pesos léxicos de BGE-M3 al formato SparseVector de Qdrant.

        Args:
            lexical_weights: Diccionario {token_id: weight} de BGE-M3.

        Returns:
            SparseVector con índices y valores para Qdrant.
        """
        if not lexical_weights:
            return SparseVector(indices=[], values=[])

        indices = sorted(lexical_weights.keys())
        values = [lexical_weights[i] for i in indices]
        return SparseVector(indices=indices, values=values)

    @staticmethod
    def dense_to_list(dense_vec: np.ndarray) -> list[float]:
        """Convierte vector denso numpy a lista de floats.

        Args:
            dense_vec: Array numpy de 1024 dimensiones.

        Returns:
            Lista de floats.
        """
        if isinstance(dense_vec, np.ndarray):
            return dense_vec.tolist()
        return list(dense_vec)


class _DummyBGE:
    """Fallback dummy cuando FlagEmbedding no está instalado.

    Genera embeddings aleatorios para desarrollo y testing.
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        max_length: int = 8192,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert_vecs: bool = False,
    ) -> dict[str, Any]:
        """Genera embeddings dummy aleatorios."""
        n = len(sentences)
        result: dict[str, Any] = {}

        if return_dense:
            rng = np.random.default_rng(
                sum(hash(s) for s in sentences) % (2**31)
            )
            result["dense_vecs"] = rng.normal(size=(n, 1024)).astype(np.float32)

        if return_sparse:
            result["lexical_weights"] = []
            for sentence in sentences:
                words = sentence.lower().split()
                weights = {}
                for i, word in enumerate(words[:100]):
                    token_id = abs(hash(word)) % 250000
                    weights[token_id] = 1.0 / (1.0 + i * 0.1)
                result["lexical_weights"].append(weights)

        return result


# ── Singleton ───────────────────────────────────────────────────────
_embedder: HybridEmbedder | None = None


def get_embedder() -> HybridEmbedder:
    """Retorna la instancia singleton del embedder híbrido."""
    global _embedder
    if _embedder is None:
        _embedder = HybridEmbedder()
    return _embedder
