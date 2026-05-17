"""
Cliente y motor de búsqueda híbrida sobre Qdrant.

Implementa la colección Pymerag con índices denso + disperso
y expone una interfaz de recuperación para el pipeline RAG.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
)

from app.core.config import settings
from app.rag.embeddings import HybridEmbedder, get_embedder

logger = logging.getLogger(__name__)


# ── Constantes de Qdrant ───────────────────────────────────────────
DENSE_VECTOR_NAME = "dense"
"""Nombre del vector denso en la colección Qdrant."""

SPARSE_VECTOR_NAME = "sparse"
"""Nombre del vector disperso en la colección Qdrant."""

HYBRID_SEARCH_LIMIT = 100
"""Límite de candidatos en la etapa de pre-búsqueda híbrida."""


class QdrantIndexer:
    """Administra la colección Qdrant y la indexación de chunks.

    Responsabilidades:
    - Crear la colección con la configuración de vectores híbridos.
    - Insertar puntos (chunks) con embeddings densos y dispersos.
    - Eliminar chunks por documento.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: HybridEmbedder | None = None,
    ) -> None:
        """Inicializa el indexador.

        Args:
            client: Cliente Qdrant (crea uno nuevo si es None).
            embedder: Instancia del embedder híbrido.
        """
        self._client = client or self._create_client()
        self._embedder = embedder or get_embedder()
        self._collection = settings.qdrant_collection

    @property
    def client(self) -> QdrantClient:
        """Cliente Qdrant subyacente."""
        return self._client

    @staticmethod
    def _create_client() -> QdrantClient:
        """Crea un cliente Qdrant según la configuración."""
        if settings.qdrant_prefer_grpc:
            return QdrantClient(
                host=settings.qdrant_host,
                grpc_port=settings.qdrant_grpc_port,
                api_key=settings.qdrant_api_key,
                prefer_grpc=True,
            )
        return QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
        )

    def collection_exists(self) -> bool:
        """Verifica si la colección ya fue creada."""
        try:
            collections = self._client.get_collections()
            return any(c.name == self._collection for c in collections.collections)
        except Exception:
            return False

    def create_collection(self, force: bool = False) -> None:
        """Crea la colección Qdrant con configuración híbrida.

        Args:
            force: Si es True, elimina y recrea la colección si ya existe.
        """
        if self.collection_exists():
            if force:
                logger.info(
                    "Eliminando colección existente '%s'...", self._collection
                )
                self._client.delete_collection(self._collection)
            else:
                logger.info(
                    "Colección '%s' ya existe. Usar force=True para recrear.",
                    self._collection,
                )
                return

        logger.info(
            "Creando colección '%s' con vectores denso (%d-dim) + disperso...",
            self._collection,
            self._embedder.dense_dim,
        )

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=self._embedder.dense_dim,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )
        logger.info("Colección '%s' creada exitosamente.", self._collection)

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 64,
    ) -> list[str]:
        """Indexa una lista de chunks en Qdrant.

        Cada chunk debe tener al menos la clave 'content'.
        Se generan embeddings densos y dispersos automáticamente.

        Args:
            chunks: Lista de diccionarios con 'content' y metadatos opcionales.
            batch_size: Tamaño del lote para generar embeddings.

        Returns:
            Lista de IDs de los puntos creados en Qdrant.
        """
        if not chunks:
            return []

        self.create_collection()

        texts = [chunk["content"] for chunk in chunks]
        qdrant_ids: list[str] = []

        logger.info("Generando embeddings para %d chunks...", len(texts))

        # Generar embeddings por lotes para no saturar memoria
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]

            embeddings = self._embedder.encode(
                batch_texts,
                return_dense=True,
                return_sparse=True,
                batch_size=len(batch_texts),
            )

            dense_vecs = embeddings.get("dense_vecs")
            lexical_weights = embeddings.get("lexical_weights")

            points = []
            for j, chunk in enumerate(batch_chunks):
                qdrant_id = str(uuid.uuid4())
                qdrant_ids.append(qdrant_id)
                chunk["qdrant_id"] = qdrant_id

                vector_payload: dict[str, Any] = {}

                if dense_vecs is not None:
                    j_idx = j if len(batch_texts) > 1 else 0
                    vector_payload[DENSE_VECTOR_NAME] = (
                        self._embedder.dense_to_list(dense_vecs[j_idx])
                    )

                if lexical_weights is not None:
                    j_idx = j if len(batch_texts) > 1 else 0
                    vector_payload[SPARSE_VECTOR_NAME] = (
                        self._embedder.to_qdrant_sparse(lexical_weights[j_idx])
                    )

                # Payload: información que Qdrant almacena junto al vector
                payload = {
                    "content": chunk["content"],
                    "document_id": chunk.get("document_id", ""),
                    "chunk_index": chunk.get("chunk_index", j),
                    "language": chunk.get("language", ""),
                }
                # Incluir metadatos adicionales aplanados
                metadata = chunk.get("metadata", {})
                for k, v in metadata.items():
                    if k not in payload:
                        payload[k] = v

                points.append(
                    models.PointStruct(
                        id=qdrant_id,
                        vector=vector_payload,
                        payload=payload,
                    )
                )

            self._client.upsert(
                collection_name=self._collection,
                points=points,
                wait=True,
            )

        logger.info(
            "%d chunks indexados en la colección '%s'.",
            len(qdrant_ids),
            self._collection,
        )
        return qdrant_ids

    def delete_by_document(self, document_id: str) -> int:
        """Elimina todos los chunks asociados a un documento.

        Args:
            document_id: ID del documento cuyos chunks se eliminarán.

        Returns:
            Número de puntos eliminados.
        """
        result = self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        deleted = result.status.completed_count if result.status else 0
        logger.info(
            "Eliminados %d chunks del documento '%s'.",
            deleted,
            document_id,
        )
        return deleted


class QdrantRetriever:
    """Motor de recuperación híbrida (dense + sparse) sobre Qdrant.

    Combina búsqueda semántica (dense) con búsqueda léxica (sparse)
    usando el mecanismo de fusión de Qdrant para obtener resultados
    más precisos.
    """

    def __init__(
        self,
        indexer: QdrantIndexer | None = None,
        embedder: HybridEmbedder | None = None,
    ) -> None:
        """Inicializa el recuperador.

        Args:
            indexer: Indexador de Qdrant (crea uno si es None).
            embedder: Instancia del embedder híbrido.
        """
        self._indexer = indexer or QdrantIndexer()
        self._embedder = embedder or get_embedder()

    @property
    def client(self) -> QdrantClient:
        """Cliente Qdrant subyacente."""
        return self._indexer.client

    @property
    def collection_name(self) -> str:
        """Nombre de la colección Qdrant."""
        return settings.qdrant_collection

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.5,
        filters: dict[str, Any] | None = None,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Recupera los chunks más relevantes para una consulta.

        Usa búsqueda híbrida densa + dispersa con fusión por
        Reciprocal Rank Fusion (RRF).

        Args:
            query: Consulta en lenguaje natural.
            top_k: Número máximo de resultados a retornar.
            score_threshold: Umbral mínimo de score para filtrar resultados.
            filters: Filtros opcionales (ej. {"document_id": "..."}).
            dense_weight: Peso del vector denso en la fusión (0-1).
            sparse_weight: Peso del vector disperso en la fusión (0-1).

        Returns:
            Lista de resultados, cada uno con 'id', 'content', 'score',
            'document_id', 'chunk_index', y metadatos.
        """
        if not self._indexer.collection_exists():
            logger.warning(
                "La colección '%s' no existe. Retornando lista vacía.",
                self.collection_name,
            )
            return []

        # Generar embedding de la consulta
        query_embedding = self._embedder.encode(
            query,
            return_dense=True,
            return_sparse=True,
        )

        dense_vec = query_embedding.get("dense_vecs")
        lexical_weights = query_embedding.get("lexical_weights")

        if dense_vec is None and lexical_weights is None:
            logger.error("No se pudo generar embedding para la consulta.")
            return []

        # Construir filtro de Qdrant si se proveyeron filtros
        qdrant_filter = self._build_filter(filters) if filters else None

        # Construir prefetch: búsqueda densa y dispersa en paralelo
        prefetch: list[models.Prefetch] = []

        if dense_vec is not None:
            dense_vector_list = self._embedder.dense_to_list(dense_vec)
            prefetch.append(
                models.Prefetch(
                    query=dense_vector_list,
                    using=DENSE_VECTOR_NAME,
                    limit=HYBRID_SEARCH_LIMIT,
                    filter=qdrant_filter,
                )
            )

        if lexical_weights is not None:
            sparse_vec = self._embedder.to_qdrant_sparse(lexical_weights)
            # Solo incluir si tiene índices válidos
            if sparse_vec.indices:
                prefetch.append(
                    models.Prefetch(
                        query=sparse_vec,
                        using=SPARSE_VECTOR_NAME,
                        limit=HYBRID_SEARCH_LIMIT,
                        filter=qdrant_filter,
                    )
                )

        if not prefetch:
            logger.warning("No hay vectores de consulta válidos.")
            return []

        # Ejecutar búsqueda híbrida con fusión
        fusion_mode = (
            models.FusionMode.RRF
            if len(prefetch) > 1
            else None
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(
                fusion=fusion_mode,
            ) if fusion_mode else None,
            with_payload=True,
            limit=top_k,
            score_threshold=score_threshold if fusion_mode is None else None,
        )

        # Procesar resultados
        output: list[dict[str, Any]] = []
        for point in results.points:
            payload = point.payload or {}
            output.append(
                {
                    "id": point.id,
                    "content": payload.get("content", ""),
                    "score": point.score or 0.0,
                    "document_id": payload.get("document_id", ""),
                    "chunk_index": payload.get("chunk_index", -1),
                    "language": payload.get("language", ""),
                    "metadata": {
                        k: v
                        for k, v in payload.items()
                        if k
                        not in {
                            "content",
                            "document_id",
                            "chunk_index",
                            "language",
                        }
                    },
                }
            )

        logger.info(
            "Búsqueda completada: '%s...' → %d resultados (score > %.2f)",
            query[:50],
            len(output),
            score_threshold,
        )
        return output

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> models.Filter:
        """Construye un filtro de Qdrant a partir de un diccionario.

        Args:
            filters: Diccionario clave-valor.

        Returns:
            Objeto Filter de Qdrant.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )
        return models.Filter(must=conditions)
