"""
Modelos de datos SQLModel para la metadata relacional de Pymerag.

Define las tablas Document, Chunk, Topic y AuditLog según
el esquema aprobado en Sprint 0.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON
from sqlmodel import Column, Field, Relationship, SQLModel


def utcnow() -> datetime:
    """Retorna el timestamp UTC actual."""
    return datetime.now(UTC)


def new_uuid() -> str:
    """Genera un UUID v4 como string."""
    return str(uuid.uuid4())


class Document(SQLModel, table=True):
    """Metadatos de un documento ingerido en el sistema."""

    __tablename__ = "documents"

    id: str = Field(default_factory=new_uuid, primary_key=True)
    """Identificador único del documento."""

    name: str = Field(index=True)
    """Nombre de archivo o título descriptivo."""

    path: str
    """Ruta original o URL de origen del documento."""

    file_type: str | None = Field(default=None)
    """Tipo de archivo: 'pdf', 'docx', 'pptx', 'txt', etc."""

    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, name="metadata"),
    )
    """Metadatos personalizados extraídos por Docling o provistos por el usuario."""

    created_at: datetime = Field(default_factory=utcnow)
    """Fecha y hora de ingesta del documento."""

    updated_at: datetime | None = Field(default=None)
    """Fecha de última modificación del documento."""

    # ── Relaciones ───────────────────────────────────────────────
    chunks: list["Chunk"] = Relationship(back_populates="document")  # noqa: UP037


class Chunk(SQLModel, table=True):
    """Fragmento (chunk) de texto indexado en Qdrant."""

    __tablename__ = "chunks"

    id: str = Field(default_factory=new_uuid, primary_key=True)
    """Identificador único del chunk."""

    document_id: str = Field(foreign_key="documents.id", index=True)
    """Referencia al documento padre."""

    content: str
    """Contenido textual del chunk."""

    qdrant_id: str = Field(unique=True, index=True)
    """Identificador del punto correspondiente en Qdrant."""

    start_index: int | None = Field(default=None)
    """Offset de inicio del chunk en el texto original."""

    end_index: int | None = Field(default=None)
    """Offset de fin del chunk en el texto original."""

    embedding_model: str | None = Field(default=None)
    """Nombre del modelo usado para generar el embedding de este chunk."""

    # ── Relaciones ───────────────────────────────────────────────
    document: "Document | None" = Relationship(back_populates="chunks")  # noqa: UP037


class Topic(SQLModel, table=True):
    """Tópico descubierto por el pipeline BERTopic."""

    __tablename__ = "topics"

    id: str = Field(default_factory=new_uuid, primary_key=True)
    """Identificador único del tópico."""

    name: str = Field(index=True)
    """Etiqueta o nombre del tópico."""

    description: str | None = Field(default=None)
    """Descripción automática o manual del tópico."""

    representative_chunks: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
    )
    """Lista de IDs de chunks que representan este tópico."""


class AuditLog(SQLModel, table=True):
    """Registro de auditoría de acciones críticas del sistema."""

    __tablename__ = "audit_logs"

    id: str = Field(default_factory=new_uuid, primary_key=True)
    """Identificador único del registro."""

    timestamp: datetime = Field(default_factory=utcnow)
    """Momento en que ocurrió la acción."""

    user_id: str | None = Field(default=None)
    """Identificador del usuario o proceso del sistema."""

    action: str = Field(index=True)
    """Acción registrada (INGEST_START, QUERY_EXECUTE, etc.)."""

    target_id: str | None = Field(default=None)
    """ID del recurso afectado por la acción."""

    payload: dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON),
    )
    """Instantánea del request/response asociado a la acción."""
