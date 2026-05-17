"""
Extractores de documentos basados en Docling.

Soporta PDF, DOCX, PPTX y archivos de texto plano.
Utiliza Docling para preservar la estructura jerárquica de los documentos.
"""

from __future__ import annotations

import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Formatos soportados ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS: dict[str, InputFormat] = {
    ".pdf": InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".pptx": InputFormat.PPTX,
    ".txt": InputFormat.ASCIIDOC,  # Docling lo trata como texto genérico
    ".md": InputFormat.MD,
    ".html": InputFormat.HTML,
}

SUPPORTED_MIME: dict[str, InputFormat] = {
    "application/pdf": InputFormat.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": InputFormat.DOCX,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": InputFormat.PPTX,
    "text/plain": InputFormat.ASCIIDOC,
    "text/markdown": InputFormat.MD,
    "text/html": InputFormat.HTML,
}


class DoclingExtractor:
    """Extrae texto estructurado de documentos usando Docling.

    Docling preserva la jerarquía del documento (títulos, secciones, tablas)
    y exporta el contenido en formato Markdown.
    """

    def __init__(self) -> None:
        """Inicializa el convertidor de Docling con configuración de pipeline."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=settings.docling_num_threads,
            device=AcceleratorDevice.CPU,
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            },
        )

    @staticmethod
    def supported_extensions() -> set[str]:
        """Extensiones de archivo que este extractor puede procesar."""
        return set(SUPPORTED_EXTENSIONS.keys())

    def extract(self, file_path: str | Path) -> str:
        """Extrae el contenido textual completo de un documento.

        Args:
            file_path: Ruta al archivo a procesar.

        Returns:
            Texto del documento en formato Markdown.

        Raises:
            ValueError: Si el formato del archivo no es soportado.
            FileNotFoundError: Si el archivo no existe.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Formato no soportado: '{ext}'. "
                f"Formatos soportados: {sorted(SUPPORTED_EXTENSIONS.keys())}"
            )

        logger.info("Extrayendo contenido de '%s' con Docling...", path.name)

        result = self._converter.convert(str(path))

        if settings.docling_export_format == "text":
            text = result.document.export_to_text()
        else:
            text = result.document.export_to_markdown()

        if not text or not text.strip():
            logger.warning("Docling no produjo texto para '%s'", path.name)
            return ""

        logger.info(
            "Extracción completada: %d caracteres de '%s'",
            len(text),
            path.name,
        )
        return text

    def extract_metadata(self, file_path: str | Path) -> dict:
        """Extrae metadatos del documento usando Docling.

        Args:
            file_path: Ruta al archivo.

        Returns:
            Diccionario con metadatos extraídos (título, autor, fecha, etc.).
        """
        path = Path(file_path)
        result = self._converter.convert(str(path))
        doc = result.document

        metadata: dict = {}
        if doc.name:
            metadata["title"] = doc.name
        if hasattr(doc, "origin") and doc.origin:
            origin = doc.origin
            if hasattr(origin, "filename"):
                metadata["filename"] = origin.filename

        return metadata
