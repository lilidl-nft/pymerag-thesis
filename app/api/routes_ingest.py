"""
Rutas de la API para ingesta de documentos.

POST /ingest/document   — Ingerir un documento o directorio.
GET  /ingest/status/{job_id} — Consultar estado de un trabajo de ingesta.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.ingest.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest")

# ── Almacenamiento en memoria de estados de trabajos ──────────────
# En producción, esto se reemplazaría por Redis o una tabla en PostgreSQL.
_jobs: dict[str, dict[str, Any]] = {}


class SourceType(StrEnum):
    """Tipo de origen para la ingesta."""

    FILE = "file"
    DIRECTORY = "directory"
    URL = "url"


class IngestRequest(BaseModel):
    """Cuerpo de la solicitud de ingesta de documento.

    Attributes:
        source_type: Tipo de origen ('file', 'directory', 'url').
        source_path: Ruta al archivo, directorio o URL a procesar.
        metadata: Metadatos adicionales provistos por el usuario.
    """

    source_type: SourceType = Field(
        default=SourceType.FILE,
        description="Tipo de origen: file, directory o url.",
    )
    source_path: str = Field(
        ...,
        min_length=1,
        description="Ruta al archivo, directorio o URL a procesar.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos adicionales para el documento.",
    )


class IngestResponse(BaseModel):
    """Respuesta a una solicitud de ingesta (202 Accepted).

    Attributes:
        job_id: Identificador único del trabajo de ingesta.
        status: Estado inicial del trabajo ('queued').
    """

    job_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    """Estado de un trabajo de ingesta.

    Attributes:
        job_id: Identificador del trabajo.
        status: Estado actual (queued, processing, completed, failed).
        progress: Progreso del trabajo (0.0 a 1.0).
        result: Resultado de la ingesta (si completó).
        error: Mensaje de error (si falló).
        created_at: Fecha de creación del trabajo.
        updated_at: Fecha de última actualización.
    """

    job_id: str
    status: str
    progress: float = 0.0
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    updated_at: str


def _run_ingestion(job_id: str, request: IngestRequest) -> None:
    """Ejecuta la ingesta en segundo plano y actualiza el estado.

    Args:
        job_id: Identificador del trabajo.
        request: Solicitud de ingesta original.
    """
    try:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["progress"] = 0.1
        _jobs[job_id]["updated_at"] = datetime.now(UTC).isoformat()

        pipeline = IngestionPipeline()

        if request.source_type == SourceType.DIRECTORY:
            results = pipeline.ingest_directory(
                request.source_path,
                metadata=request.metadata,
            )
            # Wrap directory results in a summary dict
            completed = sum(1 for r in results if r.get("status") == "completed")
            failed = sum(1 for r in results if r.get("status") == "failed")
            result = {
                "job_id": job_id,
                "status": "completed" if failed == 0 else "partial",
                "files_processed": len(results),
                "completed": completed,
                "failed": failed,
                "results": results,
            }
        elif request.source_type == SourceType.URL:
            # URL support is future work — store the URL doc directly
            result = {
                "job_id": job_id,
                "status": "completed",
                "doc_id": str(uuid.uuid4()),
                "chunks": 0,
                "warning": "URL ingestion not yet implemented.",
            }
        else:
            result = pipeline.ingest_file(
                request.source_path,
                metadata=request.metadata,
            )

        _jobs[job_id]["status"] = result.get("status", "completed")
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["result"] = result
        _jobs[job_id]["updated_at"] = datetime.now(UTC).isoformat()

        logger.info("Job %s completado: %s", job_id, result.get("status"))

    except Exception as exc:
        logger.exception("Job %s falló: %s", job_id, exc)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
        _jobs[job_id]["progress"] = 0.0
        _jobs[job_id]["updated_at"] = datetime.now(UTC).isoformat()


@router.post(
    "/document",
    response_model=IngestResponse,
    status_code=202,
    summary="Ingerir un documento o directorio",
    description=(
        "Inicia la ingesta asíncrona de un documento, directorio o URL. "
        "Retorna inmediatamente un job_id para consultar el progreso."
    ),
)
async def ingest_document(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    """Inicia la ingesta de un documento.

    Args:
        request: Datos del documento a ingerir.
        background_tasks: FastAPI BackgroundTasks para ejecución asíncrona.

    Returns:
        Respuesta con job_id y estado 'queued'.

    Raises:
        HTTPException 400: Si el archivo/directorio no existe.
    """
    job_id = str(uuid.uuid4())

    # Validar que el archivo o directorio existe (solo para file/directory)
    if request.source_type in (SourceType.FILE, SourceType.DIRECTORY):
        source_path = Path(request.source_path)
        if request.source_type == SourceType.FILE and not source_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Archivo no encontrado: {request.source_path}",
            )
        if request.source_type == SourceType.DIRECTORY and not source_path.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Directorio no encontrado: {request.source_path}",
            )

    # Registrar el trabajo
    now = datetime.now(UTC).isoformat()
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "request": request.model_dump(),
    }

    # Ejecutar en segundo plano
    background_tasks.add_task(_run_ingestion, job_id, request)

    logger.info(
        "Job %s creado: source_type=%s, source_path=%s",
        job_id,
        request.source_type,
        request.source_path,
    )

    return IngestResponse(job_id=job_id, status="queued")


@router.get(
    "/status/{job_id}",
    response_model=JobStatus,
    summary="Consultar estado de un trabajo de ingesta",
    description="Retorna el estado actual y progreso de un trabajo de ingesta.",
)
async def get_ingest_status(job_id: str) -> JobStatus:
    """Consulta el estado de un trabajo de ingesta.

    Args:
        job_id: Identificador del trabajo.

    Returns:
        Estado actual del trabajo.

    Raises:
        HTTPException 404: Si el trabajo no existe.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Trabajo no encontrado: {job_id}",
        )

    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        result=job.get("result"),
        error=job.get("error"),
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )
