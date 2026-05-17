"""
Servidor MCP (Model Context Protocol) para Pymerag.

Expone las herramientas de búsqueda, resumen, comparación y generación
con evidencia a través del protocolo MCP, permitiendo que agentes LLM
externos (Claude Desktop, Continue, etc.) interactúen con el corpus
documental indexado.

Soporta dos modos de transporte:
- HTTP/SSE en puerto 8765 (por defecto) para conexiones remotas.
- stdio para desarrollo local e integración con clientes MCP de escritorio.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from fastmcp import FastMCP

from app.core.config import settings
from app.mcp_server import tools

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────
MCP_SERVER_NAME = "pymerag-mcp"
"""Nombre del servidor MCP, usado en la negociación del protocolo."""

DEFAULT_HTTP_PORT = 8765
"""Puerto por defecto para el transporte HTTP/SSE."""

DEFAULT_HTTP_HOST = "0.0.0.0"
"""Host por defecto para el transporte HTTP/SSE."""


# ═══════════════════════════════════════════════════════════════════════
# Constructor del servidor MCP
# ═══════════════════════════════════════════════════════════════════════


def create_mcp_server() -> FastMCP:
    """Crea y configura la instancia del servidor FastMCP con todas las herramientas.

    Registra las 4 herramientas definidas en app.mcp_server.tools:
    - search: Búsqueda híbrida sobre el corpus.
    - summarize: Resumen de un documento específico.
    - compare: Comparación entre múltiples documentos.
    - generate_with_evidence: Generación de respuesta con evidencia y citas.

    Returns:
        Instancia de FastMCP configurada y lista para ejecutar.
    """
    mcp = FastMCP(
        name=MCP_SERVER_NAME,
        instructions=(
            "Pymerag MCP Server — Asistente Inteligente para Gestión Documental. "
            "Proporciona herramientas para buscar, resumir, comparar y generar "
            "respuestas con evidencia a partir de un corpus documental indexado "
            "con búsqueda híbrida (BGE-M3 + BM25) en Qdrant."
        ),
    )

    # ── Registrar herramientas ──────────────────────────────────────
    _register_tools(mcp)

    logger.info(
        "Servidor MCP '%s' creado con %d herramientas registradas.",
        MCP_SERVER_NAME,
        len(mcp.list_tools()),
    )

    return mcp


def _register_tools(mcp: FastMCP) -> None:
    """Registra las herramientas MCP en el servidor.

    Cada herramienta se registra como un tool de FastMCP usando
    el decorador mcp.tool(). Las funciones son las definidas en
    app.mcp_server.tools.

    Args:
        mcp: Instancia del servidor FastMCP.
    """
    # Tool 1: Búsqueda híbrida
    mcp.add_tool(
        tools.search,
    )

    # Tool 2: Resumen de documento
    mcp.add_tool(
        tools.summarize,
    )

    # Tool 3: Comparación de documentos
    mcp.add_tool(
        tools.compare,
    )

    # Tool 4: Generación con evidencia
    mcp.add_tool(
        tools.generate_with_evidence,
    )

    logger.debug("4 herramientas MCP registradas en el servidor.")


# ═══════════════════════════════════════════════════════════════════════
# Puntos de entrada (entry points)
# ═══════════════════════════════════════════════════════════════════════


def run_http(
    host: str = DEFAULT_HTTP_HOST,
    port: int = DEFAULT_HTTP_PORT,
    transport: str = "sse",
) -> None:
    """Inicia el servidor MCP en modo HTTP/SSE.

    Args:
        host: Dirección de escucha (0.0.0.0 para aceptar conexiones externas).
        port: Puerto TCP donde escucha el servidor.
        transport: Modo de transporte HTTP ('sse', 'http', 'streamable-http').
    """
    mcp = create_mcp_server()

    logger.info(
        "Iniciando servidor MCP en modo %s: http://%s:%d",
        transport,
        host,
        port,
    )

    mcp.run_http_async(
        host=host,
        port=port,
        transport=transport,
        show_banner=True,
    )


def run_stdio() -> None:
    """Inicia el servidor MCP en modo stdio para desarrollo local.

    Útil para integración con clientes MCP de escritorio (Claude Desktop,
    Continue, etc.) que se comunican a través de entrada/salida estándar.
    """
    mcp = create_mcp_server()

    logger.info("Iniciando servidor MCP en modo stdio.")

    mcp.run_stdio_async(
        show_banner=True,
    )


def main() -> None:
    """Punto de entrada CLI del servidor MCP.

    Soporta los argumentos:
    - --transport: Modo de transporte ('http' o 'stdio').
    - --host: Host para modo HTTP (default: 0.0.0.0).
    - --port: Puerto para modo HTTP (default: 8765).

    Ejemplos:
        python -m app.mcp_server.server --transport http --port 8765
        python -m app.mcp_server.server --transport stdio
    """
    parser = argparse.ArgumentParser(
        description="Pymerag MCP Server — Servidor de herramientas MCP",
    )
    parser.add_argument(
        "--transport",
        choices=["http", "stdio"],
        default="http",
        help="Modo de transporte: http (SSE) o stdio (default: http).",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HTTP_HOST,
        help=f"Host para modo HTTP (default: {DEFAULT_HTTP_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help=f"Puerto para modo HTTP (default: {DEFAULT_HTTP_PORT}).",
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        default=True,
        help="Usar transporte SSE en modo HTTP (default).",
    )
    parser.add_argument(
        "--streamable-http",
        action="store_true",
        default=False,
        help="Usar transporte streamable-http en modo HTTP.",
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        run_stdio()
    else:
        transport_mode = "streamable-http" if args.streamable_http else "sse"
        run_http(
            host=args.host,
            port=args.port,
            transport=transport_mode,
        )


# ── Singleton del servidor (para importación programática) ───────────
_mcp_server: FastMCP | None = None


def get_mcp_server() -> FastMCP:
    """Retorna la instancia singleton del servidor MCP.

    Útil para integrar el servidor MCP dentro de la aplicación FastAPI
    principal o para testing.

    Returns:
        Instancia configurada de FastMCP.
    """
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = create_mcp_server()
    return _mcp_server


if __name__ == "__main__":
    main()
