"""
Estrategia de segmentación (chunking) del pipeline de ingesta.

Implementa chunking con solapamiento, limpieza de texto
y detección de idioma.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from langdetect import detect, DetectorFactory

from app.core.config import settings

# Hacer langdetect determinista para tests reproducibles
DetectorFactory.seed = 42

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Representa un fragmento de texto listo para ser indexado."""

    content: str
    """Contenido textual del chunk."""

    start_index: int
    """Offset de inicio en el texto original."""

    end_index: int
    """Offset de fin en el texto original."""

    language: Optional[str] = None
    """Código ISO 639-1 del idioma detectado."""

    metadata: dict = field(default_factory=dict)
    """Metadatos adicionales del chunk."""


def clean_text(text: str) -> str:
    """Limpia el texto eliminando ruido común de documentos.

    Args:
        text: Texto crudo extraído por Docling.

    Returns:
        Texto limpio y normalizado.
    """
    if not text:
        return ""

    # Normalizar saltos de línea múltiples
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Colapsar espacios múltiples (respetando saltos de línea)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Eliminar espacios al inicio/fin de cada línea
    text = re.sub(r"^[ \t]+|[ \t]+$", "", text, flags=re.MULTILINE)

    # Eliminar líneas completamente vacías al inicio/fin
    text = text.strip()

    return text


def detect_language(text: str, min_length: int = 50) -> Optional[str]:
    """Detecta el idioma principal de un texto.

    Args:
        text: Texto a analizar.
        min_length: Longitud mínima para una detección confiable.

    Returns:
        Código ISO 639-1 del idioma, o None si no se pudo determinar.
    """
    if len(text.strip()) < min_length:
        return None

    try:
        lang = detect(text)
        logger.debug("Idioma detectado: %s", lang)
        return lang
    except Exception:
        logger.debug("No se pudo detectar el idioma del fragmento")
        return None


class TextChunker:
    """Divide texto largo en fragmentos manejables para indexación.

    Utiliza un enfoque de ventana deslizante con solapamiento
    para preservar contexto entre fragmentos adyacentes.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Inicializa el chunker.

        Args:
            chunk_size: Tamaño máximo de cada chunk en caracteres.
            chunk_overlap: Cantidad de caracteres que solapan entre chunks.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"El solapamiento ({self.chunk_overlap}) debe ser menor "
                f"que el tamaño del chunk ({self.chunk_size})"
            )

    def chunk(self, text: str, metadata: dict | None = None) -> list[TextChunk]:
        """Divide un texto en chunks solapados.

        La estrategia intenta cortar en límites de oración o párrafo
        para preservar la coherencia semántica.

        Args:
            text: Texto a dividir.
            metadata: Metadatos adicionales a propagar a cada chunk.

        Returns:
            Lista de TextChunk listos para embedding e indexación.
        """
        if not text or not text.strip():
            return []

        cleaned = clean_text(text)
        chunks: list[TextChunk] = []
        meta = metadata or {}

        start = 0
        chunk_index = 0

        while start < len(cleaned):
            end = min(start + self.chunk_size, len(cleaned))

            # Intentar cortar en un límite natural (oración o párrafo)
            if end < len(cleaned):
                # Buscar el último salto de párrafo dentro de la ventana
                last_break = cleaned.rfind("\n\n", start, end)
                if last_break > start + self.chunk_size // 2:
                    end = last_break + 2
                else:
                    # Buscar el último punto seguido de espacio o salto
                    last_sentence = max(
                        cleaned.rfind(". ", start, end),
                        cleaned.rfind(".\n", start, end),
                        cleaned.rfind("? ", start, end),
                        cleaned.rfind("!\n", start, end),
                    )
                    if last_sentence > start + self.chunk_size // 4:
                        end = last_sentence + 1

            chunk_text = cleaned[start:end].strip()
            if chunk_text:
                language = detect_language(chunk_text)
                chunks.append(
                    TextChunk(
                        content=chunk_text,
                        start_index=start,
                        end_index=end,
                        language=language,
                        metadata={
                            **meta,
                            "chunk_index": chunk_index,
                        },
                    )
                )
                chunk_index += 1

            # Avanzar con solapamiento
            start = end - self.chunk_overlap if end < len(cleaned) else len(cleaned)

        logger.info(
            "Texto dividido en %d chunks (tamaño=%d, solapamiento=%d)",
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks
