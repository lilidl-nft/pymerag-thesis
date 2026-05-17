"""
Servicio de tópicos — descubre y consulta la taxonomía del corpus.

Orquesta el pipeline BERTopic para descubrimiento no supervisado
de tópicos y expone consultas sobre el panorama de tópicos actual.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sqlmodel import Session, create_engine, select

from app.core.config import settings
from app.models.sql import Topic

logger = logging.getLogger(__name__)


class TopicService:
    """Gestiona el descubrimiento y consulta de tópicos del corpus.

    Responsabilidades:
    - Entrenar el modelo BERTopic sobre los chunks indexados.
    - Persistir los tópicos descubiertos en PostgreSQL.
    - Consultar el panorama de tópicos actual.
    """

    def __init__(self) -> None:
        """Inicializa el servicio de tópicos."""
        self._engine = create_engine(
            settings.database_url,
            echo=False,
            connect_args={"check_same_thread": False}
            if "sqlite" in settings.database_url
            else {},
        )

    def list_topics(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retorna los tópicos actualmente conocidos.

        Args:
            limit: Número máximo de tópicos a retornar.

        Returns:
            Lista de diccionarios con id, name, description y count.
        """
        with Session(self._engine) as session:
            topics = session.exec(select(Topic).limit(limit)).all()
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "description": t.description or "",
                    "count": len(t.representative_chunks),
                }
                for t in topics
            ]

    def get_topic(self, topic_id: str) -> Optional[dict[str, Any]]:
        """Obtiene un tópico específico por ID.

        Args:
            topic_id: Identificador del tópico.

        Returns:
            Diccionario con los datos del tópico o None si no existe.
        """
        with Session(self._engine) as session:
            topic = session.get(Topic, topic_id)
            if topic is None:
                return None
            return {
                "id": topic.id,
                "name": topic.name,
                "description": topic.description or "",
                "representative_chunks": topic.representative_chunks,
                "count": len(topic.representative_chunks),
            }


# ── Singleton ───────────────────────────────────────────────────────
_topic_service: Optional[TopicService] = None


def get_topic_service() -> TopicService:
    """Retorna la instancia singleton del servicio de tópicos."""
    global _topic_service
    if _topic_service is None:
        _topic_service = TopicService()
    return _topic_service
