"""
Tests para el módulo de recuperación RAG (Retriever, Embeddings, Prompt).

Valida embeddings híbridos con fallback dummy, indexación en Qdrant,
búsqueda híbrida con score threshold y generación de prompts.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.rag.embeddings import (
    HybridEmbedder,
    _DummyBGE,
    get_embedder,
)
from app.rag.retriever import (
    DENSE_VECTOR_NAME,
    HYBRID_SEARCH_LIMIT,
    SPARSE_VECTOR_NAME,
    QdrantIndexer,
    QdrantRetriever,
)

# Importar la función de construcción de prompt desde routes para test unitario
from app.api.routes_query import _build_prompt


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def dummy_embedder() -> _DummyBGE:
    """Instancia del embedder dummy determinista."""
    return _DummyBGE()


@pytest.fixture
def embedder_singleton() -> HybridEmbedder:
    """Singleton del embedder (usa fallback dummy en CI)."""
    return get_embedder()


# ── Tests de _DummyBGE ──────────────────────────────────────────────


class TestDummyBGE:
    """Pruebas unitarias para el embedder fallback _DummyBGE."""

    def test_encode_dense_output_shape(self, dummy_embedder: _DummyBGE) -> None:
        """El vector denso debe tener 1024 dimensiones."""
        sentences = ["Hola mundo", "Python es genial"]
        result = dummy_embedder.encode(
            sentences,
            return_dense=True,
            return_sparse=False,
        )
        assert "dense_vecs" in result
        dense = result["dense_vecs"]
        assert dense.shape == (2, 1024)
        assert dense.dtype == np.float32

    def test_encode_sparse_output_structure(self, dummy_embedder: _DummyBGE) -> None:
        """La salida dispersa debe ser lista de diccionarios token_id: peso."""
        sentences = ["Hola mundo", "Python es genial para datos"]
        result = dummy_embedder.encode(
            sentences,
            return_dense=False,
            return_sparse=True,
        )
        assert "lexical_weights" in result
        weights = result["lexical_weights"]
        assert len(weights) == 2
        for w in weights:
            assert isinstance(w, dict)
            # Cada diccionario asigna token_id (int) a peso (float)
            for token_id, weight in w.items():
                assert isinstance(token_id, int)
                assert isinstance(weight, float)

    def test_encode_deterministic(self, dummy_embedder: _DummyBGE) -> None:
        """Misma entrada debe producir misma salida (determinismo)."""
        sentences = ["Texto de prueba"]
        r1 = dummy_embedder.encode(sentences, return_dense=True, return_sparse=True)
        r2 = dummy_embedder.encode(sentences, return_dense=True, return_sparse=True)
        np.testing.assert_array_equal(r1["dense_vecs"], r2["dense_vecs"])
        assert r1["lexical_weights"] == r2["lexical_weights"]

    def test_different_inputs_different_dense(
        self, dummy_embedder: _DummyBGE
    ) -> None:
        """Entradas diferentes producen vectores densos diferentes."""
        r1 = dummy_embedder.encode(["Texto A"], return_dense=True, return_sparse=False)
        r2 = dummy_embedder.encode(["Texto B"], return_dense=True, return_sparse=False)
        assert not np.array_equal(r1["dense_vecs"], r2["dense_vecs"])

    def test_lexical_weights_decay_with_position(
        self, dummy_embedder: _DummyBGE
    ) -> None:
        """Los pesos léxicos deben decaer con la posición de la palabra."""
        result = dummy_embedder.encode(
            ["primera segunda tercera cuarta quinta"],
            return_dense=False,
            return_sparse=True,
        )
        weights = result["lexical_weights"][0]
        values = list(weights.values())
        # Los pesos deben ser decrecientes
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


# ── Tests de HybridEmbedder ─────────────────────────────────────────


class TestHybridEmbedder:
    """Pruebas para el HybridEmbedder (con fallback dummy en CI)."""

    def test_singleton_same_instance(self) -> None:
        """get_embedder() debe retornar siempre la misma instancia."""
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2

    def test_dense_dim_property(self, embedder_singleton: HybridEmbedder) -> None:
        """dense_dim debe ser 1024 (BGE-M3)."""
        assert embedder_singleton.dense_dim == 1024

    def test_model_name_property(self, embedder_singleton: HybridEmbedder) -> None:
        """model_name debe ser el configurado."""
        assert isinstance(embedder_singleton.model_name, str)
        assert len(embedder_singleton.model_name) > 0

    def test_encode_dense_returns_numpy(
        self, embedder_singleton: HybridEmbedder
    ) -> None:
        """encode_dense retorna np.ndarray."""
        text = "Texto de prueba para embedding"
        result = embedder_singleton.encode_dense(text)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)

    def test_encode_dense_batch_returns_2d(
        self, embedder_singleton: HybridEmbedder
    ) -> None:
        """encode_dense con lista retorna array 2D (n_textos, 1024)."""
        texts = ["Primer texto", "Segundo texto", "Tercer texto"]
        result = embedder_singleton.encode_dense(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 1024)

    def test_encode_sparse_returns_list_of_dicts(
        self, embedder_singleton: HybridEmbedder
    ) -> None:
        """encode_sparse retorna lista de diccionarios."""
        text = "Texto con palabras clave importantes"
        result = embedder_singleton.encode_sparse(text)
        assert isinstance(result, dict)

    def test_encode_hybrid_returns_both(
        self, embedder_singleton: HybridEmbedder
    ) -> None:
        """encode() con ambas flags retorna dense y sparse."""
        result = embedder_singleton.encode(
            "Consulta híbrida",
            return_dense=True,
            return_sparse=True,
        )
        assert "dense_vecs" in result
        assert "lexical_weights" in result

    def test_dense_to_list_conversion(self, embedder_singleton: HybridEmbedder) -> None:
        """dense_to_list convierte np.ndarray a lista de floats."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = embedder_singleton.dense_to_list(arr)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]

    def test_dense_to_list_handles_list_input(
        self, embedder_singleton: HybridEmbedder
    ) -> None:
        """dense_to_list también acepta listas de entrada."""
        result = embedder_singleton.dense_to_list([4.0, 5.0])
        assert result == [4.0, 5.0]

    def test_to_qdrant_sparse_empty(self, embedder_singleton: HybridEmbedder) -> None:
        """to_qdrant_sparse con dict vacío produce SparseVector vacío."""
        result = embedder_singleton.to_qdrant_sparse({})
        assert result.indices == []
        assert result.values == []

    def test_to_qdrant_sparse_sorted(self, embedder_singleton: HybridEmbedder) -> None:
        """to_qdrant_sparse ordena los índices."""
        weights = {5: 0.8, 1: 0.9, 10: 0.3}
        result = embedder_singleton.to_qdrant_sparse(weights)
        assert result.indices == [1, 5, 10]
        assert result.values == [0.9, 0.8, 0.3]

    def test_single_text_encode_returns_not_list(
        self, embedder_singleton: HybridEmbedder
    ) -> None:
        """Codificar un solo string retorna embedding único, no lista."""
        result = embedder_singleton.encode(
            "Texto único",
            return_dense=True,
            return_sparse=True,
        )
        dense = result.get("dense_vecs")
        if dense is not None:
            assert dense.ndim == 1  # debe ser (1024,) no (1, 1024)
        sparse = result.get("lexical_weights")
        if sparse is not None:
            assert isinstance(sparse, dict)  # dict único, no lista de dicts


# ── Tests de QdrantIndexer ──────────────────────────────────────────


class TestQdrantIndexer:
    """Pruebas para el indexador Qdrant con cliente mockeado."""

    def test_create_client_default(self) -> None:
        """Crea un cliente Qdrant con configuración por defecto."""
        with patch("app.rag.retriever.QdrantClient") as mock:
            QdrantIndexer._create_client()
            mock.assert_called_once()

    def test_collection_not_exists_initially(
        self, mock_indexer: MagicMock
    ) -> None:
        """Al inicio la colección no debe existir."""
        # El mock empieza vacío
        from app.rag.retriever import QdrantIndexer
        assert not mock_indexer.collection_exists()

    def test_create_collection(self, mock_indexer: MagicMock) -> None:
        """Crear colección debe invocar create_collection en Qdrant."""
        mock_indexer.create_collection()
        assert mock_indexer.collection_exists()

    def test_create_collection_idempotent(self, mock_indexer: MagicMock) -> None:
        """Crear colección dos veces no debe fallar."""
        mock_indexer.create_collection()
        mock_indexer.create_collection()
        assert mock_indexer.collection_exists()

    def test_create_collection_force(self, mock_indexer: MagicMock) -> None:
        """force=True recrea la colección aunque ya exista."""
        mock_indexer.create_collection()
        mock_indexer.create_collection(force=True)
        assert mock_indexer.collection_exists()

    def test_index_chunks_empty_list(self, mock_indexer: MagicMock) -> None:
        """Indexar lista vacía retorna lista vacía."""
        ids = mock_indexer.index_chunks([])
        assert ids == []

    def test_index_chunks_returns_ids(
        self, mock_indexer: MagicMock, sample_chunk_dicts: list[dict[str, Any]]
    ) -> None:
        """Indexar chunks retorna lista de IDs."""
        ids = mock_indexer.index_chunks(sample_chunk_dicts)
        assert len(ids) == len(sample_chunk_dicts)
        for qid in ids:
            assert isinstance(qid, str)
            assert len(qid) > 0

    def test_index_chunks_creates_collection_if_needed(
        self, mock_indexer: MagicMock, sample_chunk_dicts: list[dict[str, Any]]
    ) -> None:
        """Indexar chunks crea la colección automáticamente."""
        assert not mock_indexer.collection_exists()
        mock_indexer.index_chunks(sample_chunk_dicts)
        assert mock_indexer.collection_exists()

    def test_delete_by_document(
        self, mock_indexer: MagicMock, sample_chunk_dicts: list[dict[str, Any]]
    ) -> None:
        """Eliminar por documento retorna conteo de puntos eliminados."""
        mock_indexer.index_chunks(sample_chunk_dicts)
        deleted = mock_indexer.delete_by_document("test_doc_123")
        assert isinstance(deleted, int)


# ── Tests de QdrantRetriever ────────────────────────────────────────


class TestQdrantRetriever:
    """Pruebas para el recuperador híbrido QdrantRetriever."""

    @pytest.fixture
    def retriever(self, mock_indexer: MagicMock) -> QdrantRetriever:
        """Recuperador con indexador mockeado."""
        from app.rag.embeddings import get_embedder
        return QdrantRetriever(indexer=mock_indexer, embedder=get_embedder())

    def test_retrieve_empty_collection(
        self, retriever: QdrantRetriever
    ) -> None:
        """Consulta sobre colección vacía retorna lista vacía."""
        results = retriever.retrieve("consulta de prueba")
        assert results == []

    def test_retrieve_after_index(
        self,
        mock_indexer: MagicMock,
        sample_chunk_dicts: list[dict[str, Any]],
    ) -> None:
        """Consulta tras indexar chunks retorna resultados."""
        from app.rag.embeddings import get_embedder

        mock_indexer.index_chunks(sample_chunk_dicts)
        retriever = QdrantRetriever(indexer=mock_indexer, embedder=get_embedder())
        results = retriever.retrieve("Python y machine learning", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_retrieve_result_structure(
        self,
        mock_indexer: MagicMock,
        sample_chunk_dicts: list[dict[str, Any]],
    ) -> None:
        """Cada resultado debe tener la estructura esperada."""
        from app.rag.embeddings import get_embedder

        mock_indexer.index_chunks(sample_chunk_dicts)
        retriever = QdrantRetriever(indexer=mock_indexer, embedder=get_embedder())
        results = retriever.retrieve("Python", top_k=2)

        for r in results:
            assert "id" in r
            assert "content" in r
            assert "score" in r
            assert "document_id" in r
            assert "chunk_index" in r
            assert "language" in r
            assert "metadata" in r
            assert isinstance(r["score"], float)

    def test_retrieve_score_threshold(
        self,
        mock_indexer: MagicMock,
        sample_chunk_dicts: list[dict[str, Any]],
    ) -> None:
        """Score threshold alto (>= 0.8) debe filtrar resultados."""
        from app.rag.embeddings import get_embedder

        mock_indexer.index_chunks(sample_chunk_dicts)
        retriever = QdrantRetriever(indexer=mock_indexer, embedder=get_embedder())

        # Con threshold bajo, retorna resultados
        results_low = retriever.retrieve("Python", top_k=5, score_threshold=0.1)
        # Con threshold muy alto, podría no retornar nada
        results_high = retriever.retrieve("Python", top_k=5, score_threshold=0.99)

        # Verificar que el threshold alto no produce más resultados que el bajo
        assert len(results_high) <= len(results_low)

    def test_retrieve_with_filters(
        self,
        mock_indexer: MagicMock,
        sample_chunk_dicts: list[dict[str, Any]],
    ) -> None:
        """Filtros se aplican en la búsqueda."""
        from app.rag.embeddings import get_embedder

        mock_indexer.index_chunks(sample_chunk_dicts)
        retriever = QdrantRetriever(indexer=mock_indexer, embedder=get_embedder())

        results = retriever.retrieve(
            "Python",
            top_k=3,
            filters={"document_id": "test_doc"},
        )
        # El mock retorna resultados independientemente del filtro
        assert isinstance(results, list)

    def test_collection_name_property(self, retriever: QdrantRetriever) -> None:
        """collection_name debe ser el configurado en settings."""
        from app.core.config import settings
        assert retriever.collection_name == settings.qdrant_collection

    def test_build_filter_single_value(self) -> None:
        """_build_filter con valor único produce MatchValue."""
        from qdrant_client import models

        qfilter = QdrantRetriever._build_filter({"doc_id": "123"})
        assert isinstance(qfilter, models.Filter)
        assert len(qfilter.must) == 1

    def test_build_filter_list_value(self) -> None:
        """_build_filter con lista produce MatchAny."""
        from qdrant_client import models

        qfilter = QdrantRetriever._build_filter({"doc_id": ["123", "456"]})
        assert isinstance(qfilter, models.Filter)
        assert len(qfilter.must) == 1

    def test_build_filter_multiple_keys(self) -> None:
        """_build_filter con múltiples claves produce múltiples condiciones."""
        from qdrant_client import models

        qfilter = QdrantRetriever._build_filter({
            "doc_id": "123",
            "language": "es",
        })
        assert len(qfilter.must) == 2

    def test_retrieve_fusion_weights(
        self,
        mock_indexer: MagicMock,
        sample_chunk_dicts: list[dict[str, Any]],
    ) -> None:
        """Pesos personalizados de fusión se aceptan sin error."""
        from app.rag.embeddings import get_embedder

        mock_indexer.index_chunks(sample_chunk_dicts)
        retriever = QdrantRetriever(indexer=mock_indexer, embedder=get_embedder())

        results = retriever.retrieve(
            "consulta",
            top_k=2,
            dense_weight=0.8,
            sparse_weight=0.2,
        )
        assert isinstance(results, list)


# ── Tests de _build_prompt ──────────────────────────────────────────


class TestBuildPrompt:
    """Pruebas para la construcción de prompts del LLM."""

    def test_build_prompt_includes_query(self) -> None:
        """El prompt debe incluir la consulta del usuario."""
        chunks = [
            {"content": "Contenido relevante sobre IA", "document_id": "doc1"},
        ]
        prompt = _build_prompt("¿Qué es IA?", chunks)
        assert "¿Qué es IA?" in prompt

    def test_build_prompt_includes_context(self) -> None:
        """El prompt debe incluir los chunks como contexto."""
        chunks = [
            {"content": "La IA es inteligencia artificial.", "document_id": "doc1"},
            {"content": "Machine learning es una rama de la IA.", "document_id": "doc2"},
        ]
        prompt = _build_prompt("¿Qué es IA?", chunks)
        assert "La IA es inteligencia artificial" in prompt
        assert "Machine learning es una rama de la IA" in prompt

    def test_build_prompt_includes_source_labels(self) -> None:
        """Cada fuente debe estar etiquetada con su número y doc_id."""
        chunks = [
            {"content": "Contenido A", "document_id": "abc123"},
        ]
        prompt = _build_prompt("pregunta", chunks)
        assert "Fuente 1" in prompt
        assert "doc:abc123" in prompt

    def test_build_prompt_empty_chunks(self) -> None:
        """Prompt con lista vacía de chunks aún es válido."""
        prompt = _build_prompt("pregunta sin contexto", [])
        assert "pregunta sin contexto" in prompt
        assert "## Contexto\n" in prompt

    def test_build_prompt_structure_sections(self) -> None:
        """El prompt debe tener secciones Contexto, Pregunta y Respuesta."""
        chunks = [{"content": "contenido", "document_id": "1"}]
        prompt = _build_prompt("test", chunks)
        assert "## Contexto" in prompt
        assert "## Pregunta" in prompt
        assert "## Respuesta" in prompt
