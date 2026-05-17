"""
Rutas de la API para consulta de tópicos.

GET /topics — Obtener el panorama de tópicos del corpus.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.topics.service import get_topic_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/topics")


class TopicItem(BaseModel):
    """Ítem individual de tópico.

    Attributes:
        id: Identificador único del tópico.
        name: Nombre o etiqueta del tópico.
        description: Descripción del tópico.
        count: Cantidad de chunks asociados al tópico.
    """

    id: str
    name: str
    description: str = ""
    count: int = 0


class TopicsResponse(BaseModel):
    """Respuesta con el panorama de tópicos.

    Attributes:
        topics: Lista de tópicos descubiertos.
    """

    topics: list[TopicItem]


@router.get(
    "",
    response_model=TopicsResponse,
    summary="Consultar panorama de tópicos",
    description=(
        "Retorna la lista de tópicos descubiertos por el pipeline BERTopic "
        "junto con estadísticas básicas de cada uno."
    ),
)
async def list_topics() -> TopicsResponse:
    """Obtiene todos los tópicos actualmente conocidos.

    Returns:
        Lista de tópicos con id, nombre, descripción y conteo.
    """
    service = get_topic_service()
    raw_topics = service.list_topics()

    topics = [
        TopicItem(
            id=t["id"],
            name=t["name"],
            description=t["description"],
            count=t["count"],
        )
        for t in raw_topics
    ]

    logger.info("Consulta de tópicos: %d encontrados.", len(topics))

    return TopicsResponse(topics=topics)
