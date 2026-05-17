"""
Configuración del sistema de logging de Pymerag.

Provee una configuración unificada con soporte para salida
a consola (desarrollo) y JSON estructurado (producción).
"""

from __future__ import annotations

import logging
import sys


def setup_logging(
    level: str | None = None,
    json_format: bool = False,
) -> None:
    """Configura el logging global para la aplicación.

    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR).
               Por defecto usa INFO.
        json_format: Si es True, emite logs en formato JSON estructurado
                     (recomendado para producción con Langfuse/Datadog).
    """
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)

    if json_format:
        try:
            from pythonjsonlogger import jsonlogger
        except ImportError:
            json_format = False

    if json_format:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Limpiar handlers existentes para evitar duplicados
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Silenciar loggers ruidosos de terceros
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info("Logging configurado: nivel=%s, json=%s", log_level, json_format)
