"""
Rutas de administración y monitoreo de la API.

GET /admin/health — Verificar salud del sistema y servicios.
GET /admin/audit  — Consultar registros de auditoría.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import httpx
from fastapi import APIRouter, Query
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sqlmodel import Session, create_engine, select

from app.core.config import settings
from app.models.sql import AuditLog

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")


# ── Modelos ────────────────────────────────────────────────────────


class ServiceStatus(BaseModel):
    """Estado de un servicio externo.

    Attributes:
        status: 'up' si el servicio responde, 'down' en caso contrario.
        latency_ms: Latencia de la verificación en milisegundos.
        error: Mensaje de error si el servicio no responde.
    """

    status: str
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud.

    Attributes:
        status: Estado general ('ok' o 'degraded').
        services: Estado de cada servicio externo.
    """

    status: str
    services: dict[str, ServiceStatus]
    timestamp: str


class AuditEntry(BaseModel):
    """Entrada de auditoría individual.

    Attributes:
        id: Identificador único del registro.
        timestamp: Fecha y hora de la acción.
        action: Tipo de acción registrada.
        target_id: ID del recurso afectado.
        user_id: Usuario o proceso que ejecutó la acción.
    """

    id: str
    timestamp: str
    action: str
    target_id: str | None = None
    user_id: str | None = None


# ── Endpoints ──────────────────────────────────────────────────────


async def _check_qdrant() -> ServiceStatus:
    """Verifica la conectividad con Qdrant.

    Returns:
        Estado del servicio Qdrant.
    """
    import time

    start = time.perf_counter()
    try:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            timeout=5,
        )
        client.get_collections()
        latency = (time.perf_counter() - start) * 1000
        return ServiceStatus(status="up", latency_ms=round(latency, 2))
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("Qdrant health check failed: %s", exc)
        return ServiceStatus(
            status="down",
            latency_ms=round(latency, 2),
            error=str(exc),
        )


async def _check_llm() -> ServiceStatus:
    """Verifica la conectividad con el servidor LLM.

    Returns:
        Estado del servicio LLM.
    """
    import time

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{settings.llm_api_base}/models",
            )
            if response.status_code == 200:
                latency = (time.perf_counter() - start) * 1000
                return ServiceStatus(status="up", latency_ms=round(latency, 2))
            else:
                latency = (time.perf_counter() - start) * 1000
                return ServiceStatus(
                    status="down",
                    latency_ms=round(latency, 2),
                    error=f"HTTP {response.status_code}",
                )
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("LLM health check failed: %s", exc)
        return ServiceStatus(
            status="down",
            latency_ms=round(latency, 2),
            error=str(exc),
        )


async def _check_db() -> ServiceStatus:
    """Verifica la conectividad con la base de datos.

    Returns:
        Estado del servicio de base de datos.
    """
    import time

    start = time.perf_counter()
    try:
        engine = create_engine(
            settings.database_url,
            echo=False,
            connect_args={"check_same_thread": False}
            if "sqlite" in settings.database_url
            else {},
        )
        with Session(engine) as session:
            session.exec(select(AuditLog).limit(1)).all()
        latency = (time.perf_counter() - start) * 1000
        return ServiceStatus(status="up", latency_ms=round(latency, 2))
    except Exception as exc:
        latency = (time.perf_counter() - start) * 1000
        logger.warning("DB health check failed: %s", exc)
        return ServiceStatus(
            status="down",
            latency_ms=round(latency, 2),
            error=str(exc),
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar salud del sistema",
    description=(
        "Verifica el estado de todos los servicios externos (Qdrant, LLM, DB) "
        "y retorna su estado actual."
    ),
)
async def health_check() -> HealthResponse:
    """Endpoint de health check para monitoreo y balanceadores de carga.

    Returns:
        Estado general y por servicio.
    """
    qdrant_status = await _check_qdrant()
    llm_status = await _check_llm()
    db_status = await _check_db()

    services = {
        "qdrant": qdrant_status,
        "llm": llm_status,
        "db": db_status,
    }

    all_up = all(s.status == "up" for s in services.values())
    overall = "ok" if all_up else "degraded"

    return HealthResponse(
        status=overall,
        services=services,
        timestamp=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/audit",
    response_model=list[AuditEntry],
    summary="Consultar registros de auditoría",
    description="Retorna los registros de auditoría más recientes del sistema.",
)
async def get_audit_logs(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Número máximo de registros a retornar.",
    ),
    action: str | None = Query(
        default=None,
        description="Filtrar por tipo de acción (ej. 'QUERY_EXECUTE').",
    ),
) -> list[AuditEntry]:
    """Obtiene los logs de auditoría del sistema.

    Args:
        limit: Número máximo de entradas a retornar.
        action: Filtrar por tipo de acción.

    Returns:
        Lista de entradas de auditoría.
    """
    engine = create_engine(
        settings.database_url,
        echo=False,
        connect_args={"check_same_thread": False}
        if "sqlite" in settings.database_url
        else {},
    )

    with Session(engine) as session:
        stmt = select(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit)  # type: ignore[union-attr]
        if action:
            stmt = stmt.where(AuditLog.action == action)  # type: ignore[union-attr]
        logs = session.exec(stmt).all()

        return [
            AuditEntry(
                id=log.id,
                timestamp=log.timestamp.isoformat(),
                action=log.action,
                target_id=log.target_id,
                user_id=log.user_id,
            )
            for log in logs
        ]
