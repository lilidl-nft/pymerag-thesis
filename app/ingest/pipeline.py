"""
Pipeline de ingesta de documentos de Pymerag.

Orquesta el flujo completo: extracción → limpieza → chunking →
embedding → persistencia en Qdrant y PostgreSQL.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings
from app.ingest.chunker import TextChunk, TextChunker
from app.ingest.extractors import DoclingExtractor
from app.models.sql import Chunk, Document
from app.rag.embeddings import HybridEmbedder, get_embedder
from app.rag.retriever import QdrantIndexer

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Pipeline completo de ingesta de documentos.

    Flujo:
    1. Extraer texto y metadatos con Docling.
    2. Limpiar el texto extraído.
    3. Detectar idioma y dividir en chunks.
    4. Generar embeddings híbridos (dense + sparse).
    5. Indexar chunks en Qdrant.
    6. Persistir metadatos en PostgreSQL.
    """

    def __init__(
        self,
        indexer: QdrantIndexer | None = None,
        embedder: HybridEmbedder | None = None,
        extractor: DoclingExtractor | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        """Inicializa el pipeline de ingesta.

        Args:
            indexer: Indexador de Qdrant.
            embedder: Generador de embeddings híbridos.
            extractor: Extractor de documentos Docling.
            chunker: Segmentador de texto.
        """
        self._indexer = indexer or QdrantIndexer()
        self._embedder = embedder or get_embedder()
        self._extractor = extractor or DoclingExtractor()
        self._chunker = chunker or TextChunker()

        # Configurar la base de datos relacional
        self._engine = create_engine(
            settings.database_url,
            echo=False,
            connect_args={"check_same_thread": False}
            if "sqlite" in settings.database_url
            else {},
        )
        SQLModel.metadata.create_all(self._engine)

    @property
    def indexer(self) -> QdrantIndexer:
        """Indexador Qdrant subyacente."""
        return self._indexer

    def ingest_file(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ingiere un único documento.

        Args:
            file_path: Ruta al archivo a procesar.
            metadata: Metadatos adicionales provistos por el usuario.

        Returns:
            Resumen con doc_id, número de chunks creados y estado.
        """
        path = Path(file_path)
        job_id = str(uuid.uuid4())
        logger.info("[%s] Iniciando ingesta de '%s'...", job_id, path.name)

        # ── 1. Extracción ──────────────────────────────────────
        try:
            raw_text = self._extractor.extract(path)
            doc_meta = self._extractor.extract_metadata(path)
        except Exception as e:
            logger.error("[%s] Error en extracción: %s", job_id, e)
            return {"job_id": job_id, "status": "failed", "error": str(e)}

        if not raw_text.strip():
            logger.warning("[%s] El documento no produjo texto.", job_id)
            return {"job_id": job_id, "status": "empty", "chunks": 0}

        # ── 2. Metadatos combinados ────────────────────────────
        combined_meta = {**doc_meta, **(metadata or {})}

        # ── 3. Chunking ────────────────────────────────────────
        text_chunks: list[TextChunk] = self._chunker.chunk(
            raw_text, metadata=combined_meta
        )

        if not text_chunks:
            logger.warning("[%s] No se generaron chunks.", job_id)
            return {"job_id": job_id, "status": "empty", "chunks": 0}

        # ── 4. Preparar datos para indexación ──────────────────
        chunk_dicts: list[dict[str, Any]] = []
        for tc in text_chunks:
            chunk_dicts.append(
                {
                    "content": tc.content,
                    "start_index": tc.start_index,
                    "end_index": tc.end_index,
                    "language": tc.language,
                    "chunk_index": tc.metadata.get("chunk_index", 0),
                    "metadata": tc.metadata,
                }
            )

        # ── 5. Indexar en Qdrant ───────────────────────────────
        try:
            qdrant_ids = self._indexer.index_chunks(chunk_dicts)
        except Exception as e:
            logger.error("[%s] Error indexando en Qdrant: %s", job_id, e)
            return {"job_id": job_id, "status": "failed", "error": str(e)}

        # ── 6. Persistir en PostgreSQL ─────────────────────────
        doc_id = self._persist_document(
            path=path,
            raw_text=raw_text,
            metadata=combined_meta,
        )
        self._persist_chunks(
            document_id=doc_id,
            chunk_dicts=chunk_dicts,
            qdrant_ids=qdrant_ids,
        )

        logger.info(
            "[%s] Ingesta completada: doc='%s', %d chunks.",
            job_id,
            doc_id,
            len(qdrant_ids),
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "doc_id": doc_id,
            "chunks": len(qdrant_ids),
            "file_name": path.name,
            "file_type": path.suffix.lower().lstrip("."),
        }

    def _persist_document(
        self,
        path: Path,
        raw_text: str,
        metadata: dict[str, Any],
    ) -> str:
        """Guarda los metadatos del documento en PostgreSQL.

        Args:
            path: Ruta al archivo original.
            raw_text: Texto extraído completo.
            metadata: Metadatos combinados.

        Returns:
            ID del documento creado.
        """
        doc = Document(
            name=path.name,
            path=str(path.absolute()),
            file_type=path.suffix.lower().lstrip("."),
            metadata_=metadata,
            created_at=datetime.now(UTC),
        )

        with Session(self._engine) as session:
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc.id

    def _persist_chunks(
        self,
        document_id: str,
        chunk_dicts: list[dict[str, Any]],
        qdrant_ids: list[str],
    ) -> None:
        """Guarda la metadata de los chunks en PostgreSQL.

        Args:
            document_id: ID del documento padre.
            chunk_dicts: Lista de diccionarios de chunk.
            qdrant_ids: IDs correspondientes en Qdrant.
        """
        with Session(self._engine) as session:
            for _i, (cd, qid) in enumerate(zip(chunk_dicts, qdrant_ids, strict=True)):
                chunk = Chunk(
                    document_id=document_id,
                    content=cd["content"],
                    qdrant_id=qid,
                    start_index=cd.get("start_index"),
                    end_index=cd.get("end_index"),
                    embedding_model=self._embedder.model_name,
                )
                session.add(chunk)
            session.commit()

    def ingest_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Ingiere todos los documentos soportados en un directorio.

        Args:
            dir_path: Ruta al directorio.
            recursive: Si se deben procesar subdirectorios.
            metadata: Metadatos adicionales para todos los archivos.

        Returns:
            Lista de resultados individuales por archivo.
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"No es un directorio: {dir_path}")

        supported = self._extractor.supported_extensions()
        pattern = "**/*" if recursive else "*"
        results: list[dict[str, Any]] = []

        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                result = self.ingest_file(file_path, metadata=metadata)
                results.append(result)

        logger.info(
            "Directorio procesado: %d archivos ingeridos de '%s'.",
            len(results),
            path,
        )
        return results
