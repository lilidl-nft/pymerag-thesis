"""
Fixtures compartidos para la suite de tests de Pymerag.

Provee datos sintéticos, clientes mock y configuraciones aisladas
para que todos los tests sean deterministas y reproducibles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Parchear config antes de importar cualquier módulo de app ──────
# Para evitar que se intenten conectar a Qdrant/LLM reales durante los tests.


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirige toda la configuración a valores de testing."""
    monkeypatch.setattr(
        "app.core.config.settings.database_url",
        f"sqlite:///{tmp_path}/test.db",
    )
    monkeypatch.setattr(
        "app.core.config.settings.qdrant_host", "localhost"
    )
    monkeypatch.setattr(
        "app.core.config.settings.qdrant_port", 6333
    )
    monkeypatch.setattr(
        "app.core.config.settings.qdrant_collection", "test_collection"
    )
    monkeypatch.setattr(
        "app.core.config.settings.llm_api_base", "http://mock-llm:8080/v1"
    )
    monkeypatch.setattr(
        "app.core.config.settings.chunk_size", 128
    )
    monkeypatch.setattr(
        "app.core.config.settings.chunk_overlap", 16
    )
    monkeypatch.setattr(
        "app.core.config.settings.embedding_dense_dim", 1024
    )
    monkeypatch.setattr(
        "app.core.config.settings.embedding_device", "cpu"
    )
    monkeypatch.setattr(
        "app.core.config.settings.data_dir", tmp_path / "data"
    )
    monkeypatch.setattr(
        "app.core.config.settings.upload_dir", tmp_path / "data/uploads"
    )


# ── Datos sintéticos ────────────────────────────────────────────────


SAMPLE_DOC_TEXT = (
    "Python es un lenguaje de programación interpretado de alto nivel. "
    "Fue creado por Guido van Rossum y lanzado por primera vez en 1991. "
    "Python se caracteriza por su sintaxis clara y legible, lo que lo hace "
    "ideal para principiantes. El ecosistema de Python incluye bibliotecas "
    "para ciencia de datos como NumPy, Pandas y Scikit-learn.\n\n"
    "En el campo de la inteligencia artificial, Python se ha convertido en "
    "el lenguaje dominante. Frameworks como PyTorch y TensorFlow permiten "
    "entrenar redes neuronales profundas. Los modelos de lenguaje grande "
    "(LLMs) como GPT, LLaMA y DeepSeek han revolucionado el procesamiento "
    "del lenguaje natural.\n\n"
    "La arquitectura RAG (Retrieval-Augmented Generation) combina modelos "
    "de lenguaje con búsqueda en bases de conocimiento externas. Esto "
    "permite generar respuestas fundamentadas en documentos reales, "
    "reduciendo las alucinaciones y mejorando la precisión factual.\n\n"
    "Qdrant es una base de datos vectorial de alto rendimiento escrita en "
    "Rust. Soporta búsqueda híbrida combinando vectores densos y dispersos, "
    "lo que la hace ideal para pipelines RAG que requieren tanto búsqueda "
    "semántica como léxica."
)

SAMPLE_METADATA = {
    "title": "Introducción a Python y RAG",
    "author": "Equipo Pymerag",
    "language": "es",
}


@pytest.fixture
def sample_text() -> str:
    """Texto de ejemplo en español para tests de chunking."""
    return SAMPLE_DOC_TEXT


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Metadatos de ejemplo para documentos."""
    return dict(SAMPLE_METADATA)


@pytest.fixture
def sample_chunk_dicts() -> list[dict[str, Any]]:
    """Lista de diccionarios de chunks sintéticos."""
    return [
        {
            "content": "Python es un lenguaje de programación interpretado.",
            "start_index": 0,
            "end_index": 53,
            "language": "es",
            "chunk_index": 0,
            "metadata": {"title": "Test Doc"},
        },
        {
            "content": "El ecosistema de Python incluye NumPy y Pandas.",
            "start_index": 54,
            "end_index": 104,
            "language": "es",
            "chunk_index": 1,
            "metadata": {"title": "Test Doc"},
        },
        {
            "content": "RAG combina LLMs con búsqueda en bases de conocimiento.",
            "start_index": 105,
            "end_index": 170,
            "language": "es",
            "chunk_index": 2,
            "metadata": {"title": "Test Doc"},
        },
    ]


@pytest.fixture
def sample_temp_file(tmp_path: Path) -> Path:
    """Crea un archivo temporal .txt con contenido de prueba."""
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text(SAMPLE_DOC_TEXT, encoding="utf-8")
    return file_path


@pytest.fixture
def sample_temp_dir(tmp_path: Path) -> Path:
    """Crea un directorio temporal con archivos .txt de prueba."""
    dir_path = tmp_path / "test_docs"
    dir_path.mkdir()
    (dir_path / "doc1.txt").write_text("Documento uno. Contenido de prueba.")
    (dir_path / "doc2.txt").write_text("Documento dos. Más contenido interesante.")
    # Agregar un archivo no soportado que debería ser ignorado
    (dir_path / "image.png").write_text("fake png")
    return dir_path


# ── Mock de QdrantClient ────────────────────────────────────────────


class FakeQdrantCollections:
    """Colecciones falsas para mock de Qdrant."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.collections = [
            MagicMock(name=n) for n in (names or [])
        ]
        for c, n in zip(self.collections, names or []):
            c.name = n


class FakePointStruct:
    """Versión simplificada de models.PointStruct para assertions."""

    def __init__(self, id: Any, vector: Any, payload: Any) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class FakeScoredPoint:
    """Versión simplificada de ScoredPoint para assertions."""

    def __init__(self, id: Any, score: float, payload: dict | None = None) -> None:
        self.id = id
        self.score = score
        self.payload = payload or {}


class FakeQueryResult:
    """Resultado falso de query_points."""

    def __init__(self, points: list[FakeScoredPoint]) -> None:
        self.points = points


@pytest.fixture
def mock_qdrant_client() -> Generator[MagicMock, None, None]:
    """Cliente Qdrant mockeado con comportamientos predecibles."""
    with patch("app.rag.retriever.QdrantClient", autospec=True) as mock_cls:
        client = MagicMock()

        # Control de colecciones
        _collections: list[str] = []

        def _get_collections():
            return FakeQdrantCollections(_collections)

        def _create_collection(collection_name, vectors_config, sparse_vectors_config):
            if collection_name not in _collections:
                _collections.append(collection_name)

        def _delete_collection(collection_name):
            if collection_name in _collections:
                _collections.remove(collection_name)

        client.get_collections.side_effect = _get_collections
        client.create_collection.side_effect = _create_collection
        client.delete_collection.side_effect = _delete_collection

        # Almacén de puntos
        _points: list[FakePointStruct] = []

        def _upsert(collection_name, points, wait=True):
            for p in points:
                # Remover puntos existentes con el mismo ID
                nonlocal _points
                _points = [ep for ep in _points if ep.id != p.id]
                _points.append(FakePointStruct(p.id, p.vector, p.payload))

        client.upsert.side_effect = _upsert

        def _query_points(
            collection_name,
            prefetch=None,
            query=None,
            with_payload=True,
            limit=5,
            score_threshold=None,
        ):
            # Búsqueda simple: retornar los puntos más parecidos por score fijo
            results = []
            for i, p in enumerate(_points[:limit]):
                score = 0.95 - (i * 0.1)
                if score_threshold is not None and score < score_threshold:
                    continue
                results.append(FakeScoredPoint(p.id, score, p.payload))
            return FakeQueryResult(results)

        client.query_points.side_effect = _query_points

        class FakeDeleteResult:
            status = MagicMock(completed_count=len(_points))

        def _delete(collection_name, points_selector):
            nonlocal _points
            count = len(_points)
            _points = []
            result = FakeDeleteResult()
            result.status.completed_count = count
            return result

        client.delete.side_effect = _delete

        mock_cls.return_value = client
        yield client


# ── Mock de QdrantIndexer ───────────────────────────────────────────


@pytest.fixture
def mock_indexer(mock_qdrant_client: MagicMock) -> Generator[MagicMock, None, None]:
    """Indexador Qdrant con cliente mockeado."""
    with patch(
        "app.rag.retriever.QdrantIndexer._create_client",
        return_value=mock_qdrant_client,
    ):
        from app.rag.retriever import QdrantIndexer

        indexer = QdrantIndexer()
        yield indexer


# ── Mock de httpx AsyncClient ───────────────────────────────────────


@pytest.fixture
def mock_httpx_client() -> Generator[MagicMock, None, None]:
    """Cliente HTTP asíncrono mockeado para el endpoint de LLM."""
    with patch("httpx.AsyncClient", autospec=True) as mock_cls:
        client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Python es un lenguaje de programación de alto nivel creado por Guido van Rossum."
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        async def _post(*args, **kwargs):
            return mock_response

        client.post.side_effect = _post

        async def _aclose():
            pass

        client.aclose.side_effect = _aclose

        mock_cls.return_value = client
        yield client


# ── TestClient de FastAPI ───────────────────────────────────────────


@pytest.fixture
def test_app(
    _patch_settings: None,  # autouse fixture
    mock_qdrant_client: MagicMock,
) -> Generator[TestClient, None, None]:
    """Cliente de test de FastAPI con servicios externos mockeados."""
    from app.main import app

    with TestClient(app) as client:
        yield client


# ── Helper para cargar el golden set ─────────────────────────────────


@pytest.fixture
def golden_set_path() -> Path:
    """Ruta al archivo de golden set para evaluación RAGAS."""
    from app.core.config import settings
    return settings.data_dir.parent / "data" / "golden_set.jsonl"


@pytest.fixture
def load_golden_set(golden_set_path: Path) -> list[dict[str, Any]]:
    """Carga el golden set como lista de diccionarios."""
    if not golden_set_path.exists():
        return []
    items: list[dict[str, Any]] = []
    with open(golden_set_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
