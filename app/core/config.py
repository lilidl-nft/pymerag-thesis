"""
Configuración central de Pymerag.

Carga variables de entorno y expone una instancia singleton de Settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración global de la aplicación Pymerag."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Base de datos ──────────────────────────────────────────────
    database_url: str = "sqlite:///pymerag.db"
    """URL de conexión a PostgreSQL (o SQLite para desarrollo)."""

    # ── Qdrant ─────────────────────────────────────────────────────
    qdrant_host: str = "localhost"
    """Host del servidor Qdrant."""

    qdrant_port: int = 6333
    """Puerto REST de Qdrant."""

    qdrant_grpc_port: int = 6334
    """Puerto gRPC de Qdrant."""

    qdrant_api_key: str | None = None
    """Clave API para Qdrant Cloud (opcional)."""

    qdrant_collection: str = "pymerag_chunks"
    """Nombre de la colección en Qdrant."""

    qdrant_prefer_grpc: bool = False
    """Preferir gRPC sobre HTTP para Qdrant."""

    # ── Embeddings ─────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    """Modelo de embeddings (BGE-M3 para búsqueda híbrida)."""

    embedding_device: str = "cpu"
    """Dispositivo para inferencia de embeddings ('cpu' o 'cuda')."""

    embedding_dense_dim: int = 1024
    """Dimensionalidad del vector denso de BGE-M3."""

    # ── Chunking ───────────────────────────────────────────────────
    chunk_size: int = 512
    """Tamaño máximo de cada chunk en tokens."""

    chunk_overlap: int = 64
    """Solapamiento entre chunks consecutivos en tokens."""

    # ── Docling ────────────────────────────────────────────────────
    docling_num_threads: int = 4
    """Número de hilos para Docling."""

    docling_export_format: str = "markdown"
    """Formato de exportación de Docling ('markdown' o 'text')."""

    # ── LLM / llama.cpp ────────────────────────────────────────────
    llm_model_path: str | None = None
    """Ruta al modelo GGUF para llama.cpp (None = usar API externa)."""

    llm_api_base: str = "http://localhost:8080/v1"
    """URL base de la API compatible con OpenAI (llama.cpp server)."""

    # ── Langfuse (observabilidad) ──────────────────────────────────
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"

    # ── API Server ─────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    """Host donde escucha el servidor FastAPI."""

    api_port: int = 8000
    """Puerto del servidor FastAPI."""

    api_debug: bool = False
    """Modo debug de FastAPI (deshabilitar en producción)."""

    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://localhost:8000",
    ]
    """Orígenes permitidos para CORS."""

    # ── Seguridad / JWT ──────────────────────────────────────────────
    jwt_secret: str = "pymerag-secret-change-in-production"
    """Clave secreta para firmar tokens JWT."""

    jwt_algorithm: str = "HS256"
    """Algoritmo de firma para JWT."""

    jwt_expire_minutes: int = 60
    """Tiempo de expiración de los tokens JWT en minutos."""

    # ── Rate Limiting ───────────────────────────────────────────────
    rate_limit_requests: int = 100
    """Número máximo de solicitudes por ventana de rate limiting."""

    rate_limit_window: int = 60
    """Ventana de tiempo para rate limiting en segundos."""

    # ── Directorios ────────────────────────────────────────────────
    data_dir: Path = Path("data")
    """Directorio raíz para datos y corpus."""

    upload_dir: Path = Path("data/uploads")
    """Directorio para archivos subidos temporalmente."""


# ── Singleton ───────────────────────────────────────────────────────
settings = Settings()
