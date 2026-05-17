"""
Pymerag Frontend — Interfaz Streamlit para el Asistente RAG-MCP.

Módulos de navegación:
- Chat: Consultas conversacionales con fuentes citables e historial.
- Documentos: Ingesta de documentos (archivo/directorio) y monitoreo de trabajos.
- Tópicos: Visualización del panorama temático del corpus indexado.
- Admin: Health check de servicios y registros de auditoría.

API Contract: docs/sprint0_designs/api_contract.md
Base URL: http://localhost:8000/api/v1

Usage:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import datetime
import json
import os
import time
from collections.abc import Callable
from typing import Any

import httpx
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

API_BASE_URL: str = os.getenv("PYMERAG_API_URL", "http://localhost:8000/api/v1")
"""Base URL for the Pymerag REST API. Override with PYMERAG_API_URL env var."""

REQUEST_TIMEOUT: float = 120.0
"""HTTP request timeout in seconds."""

# ═══════════════════════════════════════════════════════════════════════════════
# API Client
# ═══════════════════════════════════════════════════════════════════════════════


class APIClient:
    """Thin HTTP client wrapper for the Pymerag REST API.

    All business logic lives in the backend; this client only handles
    serialization, transport, and error mapping.
    """

    def __init__(self, base_url: str = API_BASE_URL, timeout: float = REQUEST_TIMEOUT) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 5) -> dict[str, Any]:
        """POST /query — execute a RAG query.

        Args:
            query_text: Natural language query.
            top_k: Number of chunks to retrieve.

        Returns:
            Response dict with 'answer', 'sources', and 'metadata'.
        """
        payload: dict[str, Any] = {
            "query": query_text,
            "top_k": top_k,
            "stream": False,
            "filters": {},
        }
        r = self.client.post(f"{self._base}/query", json=payload)
        r.raise_for_status()
        return r.json()

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_document(
        self,
        source_path: str,
        source_type: str = "file",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /ingest/document — trigger async document ingestion.

        Args:
            source_path: File path, directory path, or URL.
            source_type: One of 'file', 'directory', 'url'.
            metadata: Optional user-provided metadata.

        Returns:
            Response dict with 'job_id' and 'status'.
        """
        payload: dict[str, Any] = {
            "source_type": source_type,
            "source_path": source_path,
            "metadata": metadata or {},
        }
        r = self.client.post(f"{self._base}/ingest/document", json=payload)
        r.raise_for_status()
        return r.json()

    def ingest_status(self, job_id: str) -> dict[str, Any]:
        """GET /ingest/status/{job_id} — check ingestion job progress.

        Args:
            job_id: Job identifier returned by ingest_document.

        Returns:
            Response dict with 'status', 'progress', 'result', 'error', etc.
        """
        r = self.client.get(f"{self._base}/ingest/status/{job_id}")
        r.raise_for_status()
        return r.json()

    # ── Topics ─────────────────────────────────────────────────────────────

    def list_topics(self) -> dict[str, Any]:
        """GET /topics — retrieve the topic landscape.

        Returns:
            Response dict with 'topics' list.
        """
        r = self.client.get(f"{self._base}/topics")
        r.raise_for_status()
        return r.json()

    # ── Admin ──────────────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """GET /admin/health — check system and service health.

        Returns:
            Response dict with 'status', 'services', 'timestamp'.
        """
        r = self.client.get(f"{self._base}/admin/health")
        r.raise_for_status()
        return r.json()

    def audit_logs(self, limit: int = 50, action: str | None = None) -> list[dict[str, Any]]:
        """GET /admin/audit — retrieve audit log entries.

        Args:
            limit: Maximum entries to return.
            action: Optional filter by action type.

        Returns:
            List of audit entry dicts.
        """
        params: dict[str, Any] = {"limit": limit}
        if action:
            params["action"] = action
        r = self.client.get(f"{self._base}/admin/audit", params=params)
        r.raise_for_status()
        return r.json()


def get_api_client() -> APIClient:
    """Return a cached APIClient instance stored in session state."""
    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient()
    return st.session_state.api_client


# ═══════════════════════════════════════════════════════════════════════════════
# UI Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render expandable source citations with scores and content previews.

    Args:
        sources: List of source dicts from the query response.
    """
    if not sources:
        return

    with st.expander(f"📚 Fuentes consultadas ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            chunk_id = src.get("chunk_id", "?")
            score = src.get("score", 0.0)
            content = src.get("content", "")
            metadata = src.get("metadata", {})

            score_color = (
                "green" if score >= 0.8 else "orange" if score >= 0.5 else "red"
            )
            st.markdown(
                f"**Fuente {i}** — Score: :{score_color}[{score:.3f}] — "
                f"`{chunk_id[:12]}…`"
            )
            if metadata:
                st.caption(f"Metadatos: {json.dumps(metadata, ensure_ascii=False)}")
            with st.expander("Ver contenido", expanded=False):
                st.text(content[:2000])
            st.divider()


def render_status_badge(status: str) -> str:
    """Return a colored badge string for a job/service status."""
    mapping = {
        "queued": "🔵 En cola",
        "processing": "🟡 Procesando",
        "completed": "🟢 Completado",
        "failed": "🔴 Fallido",
        "partial": "🟠 Parcial",
        "up": "🟢 Activo",
        "down": "🔴 Inactivo",
        "ok": "🟢 OK",
        "degraded": "🟠 Degradado",
    }
    return mapping.get(status, f"⚪ {status}")


def render_progress_bar(progress: float, status: str) -> None:
    """Render a progress bar with status label.

    Args:
        progress: Float between 0.0 and 1.0.
        status: Current job status string.
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(min(max(progress, 0.0), 1.0))
    with col2:
        st.caption(render_status_badge(status))


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Chat
# ═══════════════════════════════════════════════════════════════════════════════


def chat_page() -> None:
    """Conversational RAG interface with source citations and chat history."""
    st.title("💬 Chat — Asistente RAG")
    st.caption("Consultá tu corpus documental con lenguaje natural.")

    # ── Sidebar settings ──────────────────────────────────────────────────
    with st.sidebar:
        st.subheader("⚙️ Configuración de consulta")
        top_k = st.slider(
            "Fragmentos a recuperar (top_k)",
            min_value=1,
            max_value=20,
            value=5,
            help="Cantidad de chunks del corpus que se usarán como contexto.",
        )
        if st.button("🗑️ Limpiar historial", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Initialize session state ──────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Render message history ────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])
            if msg.get("metadata"):
                latency = msg["metadata"].get("latency")
                if latency is not None:
                    st.caption(f"⏱️ Latencia: {latency:.2f}s")

    # ── Chat input ────────────────────────────────────────────────────────
    if prompt := st.chat_input("Escribí tu pregunta sobre los documentos…"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call API
        api = get_api_client()
        with st.chat_message("assistant"):
            with st.spinner("🔍 Buscando en el corpus y generando respuesta…"):
                try:
                    response = api.query(query_text=prompt, top_k=top_k)
                    answer = response.get("answer", "Sin respuesta.")
                    sources = response.get("sources", [])
                    metadata = response.get("metadata", {})

                    st.markdown(answer)
                    render_sources(sources)
                    latency = metadata.get("latency")
                    if latency is not None:
                        st.caption(f"⏱️ Latencia: {latency:.2f}s")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "metadata": metadata,
                    })
                except httpx.HTTPError as exc:
                    error_msg = f"❌ Error de conexión con la API: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })
                except Exception as exc:
                    error_msg = f"❌ Error inesperado: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Documentos (Ingestion)
# ═══════════════════════════════════════════════════════════════════════════════


def documents_page() -> None:
    """Document ingestion and job monitoring interface."""
    st.title("📄 Documentos — Ingesta y Monitoreo")
    st.caption("Subí documentos al corpus o monitoreá trabajos de ingesta activos.")

    tab_upload, tab_monitor = st.tabs(["📤 Subir documento", "🔍 Monitorear trabajo"])

    # ── Tab: Upload ───────────────────────────────────────────────────────
    with tab_upload:
        st.subheader("Ingesta de documentos")

        source_type = st.radio(
            "Tipo de origen",
            options=["file", "directory"],
            format_func=lambda x: "📄 Archivo" if x == "file" else "📁 Directorio",
            horizontal=True,
        )

        source_path_input: str = ""
        if source_type == "file":
            uploaded_file = st.file_uploader(
                "Seleccioná un archivo",
                type=None,
                help="Formatos soportados: PDF, DOCX, TXT, MD, HTML, etc.",
            )
            source_path_input = st.text_input(
                "O especificá una ruta absoluta en el servidor",
                placeholder="/ruta/al/documento.pdf",
            )
            source_path = source_path_input if source_path_input else None
            if uploaded_file is not None:
                st.info(
                    "📌 Archivos subidos desde el navegador deben ser guardados "
                    "en el servidor antes de la ingesta. Usá la ruta del servidor "
                    "para archivos ya disponibles en el filesystem."
                )
        else:
            source_path = st.text_input(
                "Ruta del directorio en el servidor",
                placeholder="/ruta/al/directorio/",
                help="Directorio con documentos a ingerir (PDF, DOCX, TXT, etc.).",
            )
            uploaded_file = None

        with st.expander("📋 Metadatos adicionales (opcional)", expanded=False):
            metadata_json = st.text_area(
                "Metadatos en formato JSON",
                value="{}",
                height=100,
                help='Ejemplo: {"autor": "Juan Pérez", "categoria": "legal"}',
            )

        if st.button("🚀 Iniciar ingesta", type="primary", use_container_width=True):
            # Resolve source path
            resolved_path: str | None = None
            if source_type == "file" and source_path_input:
                resolved_path = source_path_input
            elif source_type == "file" and uploaded_file is not None:
                st.error(
                    "La ingesta por subida directa no está implementada en esta versión. "
                    "Usá la ruta del servidor para archivos ya disponibles."
                )
                return
            elif source_type == "directory":
                resolved_path = source_path

            if not resolved_path:
                st.warning("⚠️ Especificá una ruta de origen válida.")
                return

            # Parse metadata
            try:
                metadata = json.loads(metadata_json) if metadata_json.strip() else {}
            except json.JSONDecodeError:
                st.error("❌ El JSON de metadatos no es válido.")
                return

            # Call API
            api = get_api_client()
            with st.spinner("📨 Enviando solicitud de ingesta…"):
                try:
                    response = api.ingest_document(
                        source_path=resolved_path,
                        source_type=source_type,
                        metadata=metadata,
                    )
                    job_id = response.get("job_id", "?")
                    st.success(f"✅ Trabajo creado: `{job_id}`")
                    st.info(
                        "⏳ La ingesta se ejecuta en segundo plano. "
                        "Usá la pestaña 'Monitorear trabajo' para seguir el progreso."
                    )
                    # Store last job for convenience
                    st.session_state.last_job_id = job_id
                except httpx.HTTPError as exc:
                    st.error(f"❌ Error de conexión: {exc}")
                except Exception as exc:
                    st.error(f"❌ Error: {exc}")

    # ── Tab: Monitor ──────────────────────────────────────────────────────
    with tab_monitor:
        st.subheader("Estado de trabajos de ingesta")

        job_id_default = st.session_state.get("last_job_id", "")
        job_id = st.text_input(
            "ID del trabajo (job_id)",
            value=job_id_default,
            placeholder="UUID del trabajo de ingesta",
        )

        col_check, col_auto = st.columns([1, 2])
        with col_check:
            check_clicked = st.button("🔍 Consultar estado", use_container_width=True)
        with col_auto:
            auto_refresh = st.checkbox("Auto-refresh (cada 2s)", value=False)

        if check_clicked or (auto_refresh and job_id):
            if not job_id:
                st.warning("⚠️ Ingresá un job_id para consultar.")
            else:
                api = get_api_client()
                try:
                    status_data = api.ingest_status(job_id)
                    st.json(status_data)

                    # Visual progress
                    progress = status_data.get("progress", 0.0)
                    status = status_data.get("status", "unknown")
                    render_progress_bar(progress, status)

                    # Show result details
                    if status in ("completed", "partial"):
                        result = status_data.get("result")
                        if result:
                            st.success(f"✅ {result.get('files_processed', '?')} archivos procesados")  # noqa: E501
                    elif status == "failed":
                        error = status_data.get("error", "Error desconocido")
                        st.error(f"❌ {error}")

                except httpx.HTTPError as exc:
                    st.error(f"❌ Error: {exc}")
                except Exception as exc:
                    st.error(f"❌ Error inesperado: {exc}")

            if auto_refresh:
                time.sleep(2)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Tópicos
# ═══════════════════════════════════════════════════════════════════════════════


def topics_page() -> None:
    """Topic landscape visualization.

    Displays topics discovered by BERTopic from the indexed corpus,
    along with their descriptions and document counts.
    """
    st.title("🏷️ Tópicos — Panorama Temático")
    st.caption("Tópicos descubiertos automáticamente en el corpus indexado.")

    api = get_api_client()

    if st.button("🔄 Cargar tópicos", use_container_width=True):
        st.rerun()

    with st.spinner("📊 Consultando tópicos…"):
        try:
            data = api.list_topics()
            topics: list[dict[str, Any]] = data.get("topics", [])
        except httpx.HTTPError as exc:
            st.error(f"❌ Error de conexión con la API: {exc}")
            return
        except Exception as exc:
            st.error(f"❌ Error inesperado: {exc}")
            return

    if not topics:
        st.info("ℹ️ No hay tópicos disponibles. Ingerí documentos primero.")
        return

    st.metric("Tópicos descubiertos", len(topics))

    # ── Sort by count descending ──────────────────────────────────────────
    topics_sorted = sorted(topics, key=lambda t: t.get("count", 0), reverse=True)

    # ── Bar chart ─────────────────────────────────────────────────────────
    chart_data = {
        t["name"]: t.get("count", 0)
        for t in topics_sorted[:20]  # top 20
    }
    if chart_data:
        st.subheader("Distribución de documentos por tópico")
        st.bar_chart(chart_data)

    # ── Detailed table ────────────────────────────────────────────────────
    st.subheader("Listado detallado")
    for topic in topics_sorted:
        name = topic.get("name", "Sin nombre")
        description = topic.get("description", "")
        count = topic.get("count", 0)
        topic_id = topic.get("id", "")

        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"### {name}")
                if description:
                    st.caption(description)
                st.caption(f"ID: `{topic_id}`")
            with cols[1]:
                st.metric("Documentos", count)


# ═══════════════════════════════════════════════════════════════════════════════
# Page: Admin
# ═══════════════════════════════════════════════════════════════════════════════


def admin_page() -> None:
    """Admin panel: health check dashboard and audit log viewer."""
    st.title("⚙️ Administración — Salud y Auditoría")
    st.caption("Monitoreo del sistema, estado de servicios y registros de actividad.")

    tab_health, tab_audit = st.tabs(["🏥 Salud del sistema", "📋 Registros de auditoría"])

    # ── Tab: Health ───────────────────────────────────────────────────────
    with tab_health:
        st.subheader("Estado de los servicios")

        if st.button("🔄 Refrescar", key="health_refresh"):
            st.rerun()

        api = get_api_client()
        with st.spinner("🔍 Verificando servicios…"):
            try:
                health = api.health()
            except httpx.HTTPError as exc:
                st.error(f"❌ No se pudo conectar con la API: {exc}")
                return
            except Exception as exc:
                st.error(f"❌ Error inesperado: {exc}")
                return

        overall = health.get("status", "unknown")
        timestamp = health.get("timestamp", "")
        services = health.get("services", {})

        # Overall status
        overall_badge = render_status_badge(overall)
        st.markdown(f"### Estado general: {overall_badge}")
        if timestamp:
            st.caption(f"Última verificación: {timestamp}")

        st.divider()

        # Per-service cards
        for svc_name, svc_data in services.items():
            svc_status = svc_data.get("status", "unknown")
            svc_latency = svc_data.get("latency_ms")
            svc_error = svc_data.get("error")

            emoji = {"qdrant": "🗄️", "llm": "🧠", "db": "🗃️"}.get(svc_name, "🔌")
            name_display = {"qdrant": "Qdrant (Vector DB)", "llm": "LLM Server", "db": "Base de datos"}.get(  # noqa: E501
                svc_name, svc_name
            )

            with st.container(border=True):
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    st.markdown(f"{emoji} **{name_display}**")
                with cols[1]:
                    st.markdown(render_status_badge(svc_status))
                with cols[2]:
                    if svc_latency is not None:
                        st.metric("Latencia", f"{svc_latency:.1f} ms")
                if svc_error:
                    st.error(f"Error: {svc_error}")

    # ── Tab: Audit ────────────────────────────────────────────────────────
    with tab_audit:
        st.subheader("Registros de actividad")

        col1, col2 = st.columns([1, 3])
        with col1:
            limit = st.number_input("Límite", min_value=1, max_value=500, value=50, step=10)
        with col2:
            action_filter = st.text_input(
                "Filtrar por acción (opcional)",
                placeholder="QUERY_EXECUTE, INGEST_START, …",
            )

        if st.button("🔍 Cargar registros", key="audit_load"):
            st.rerun()

        api = get_api_client()
        with st.spinner("📋 Cargando auditoría…"):
            try:
                logs = api.audit_logs(
                    limit=limit,
                    action=action_filter if action_filter else None,
                )
            except httpx.HTTPError as exc:
                st.error(f"❌ Error de conexión: {exc}")
                return
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                return

        if not logs:
            st.info("ℹ️ No hay registros de auditoría disponibles.")
            return

        st.metric("Registros encontrados", len(logs))

        # Table
        rows = []
        for entry in logs:
            ts = entry.get("timestamp", "")
            # Truncate ISO timestamp for display
            if ts and len(ts) > 19:
                ts = ts[:19].replace("T", " ")
            rows.append({
                "Timestamp": ts,
                "Acción": entry.get("action", ""),
                "Usuario": entry.get("user_id", ""),
                "Target ID": entry.get("target_id", "") or "—",
                "ID": entry.get("id", ""),
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Timestamp", width="medium"),
                "Acción": st.column_config.TextColumn("Acción", width="medium"),
                "Usuario": st.column_config.TextColumn("Usuario", width="small"),
                "Target ID": st.column_config.TextColumn("Target ID", width="small"),
                "ID": st.column_config.TextColumn("ID", width="small"),
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

PAGES: dict[str, Callable[[], None]] = {
    "💬 Chat": chat_page,
    "📄 Documentos": documents_page,
    "🏷️ Tópicos": topics_page,
    "⚙️ Admin": admin_page,
}


def main() -> None:
    """Configure the Streamlit app and render the selected page."""
    st.set_page_config(
        page_title="Pymerag — Asistente RAG-MCP",
        page_icon="🦜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/streamlit/streamlit/develop/lib/streamlit/static/favicon.png",
            width=48,
        )
        st.title("Pymerag")
        st.caption("Asistente Inteligente RAG-MCP")

        st.divider()

        selected_page = st.radio(
            "Navegación",
            options=list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.divider()

        # API status indicator (quick timeout to avoid blocking UI)
        st.caption(f"API: `{API_BASE_URL}`")
        api_ok = False
        try:
            # Use a separate short-lived client for the health check
            quick_client = httpx.Client(timeout=3.0)
            r = quick_client.get(f"{API_BASE_URL.rstrip('/')}/admin/health")
            api_ok = r.json().get("status") == "ok"
            quick_client.close()
        except Exception:
            api_ok = False

        if api_ok:
            st.success("🟢 API conectada")
        else:
            st.error("🔴 API no disponible")

        st.divider()
        st.caption(f"Pymerag v0.1.0 — {datetime.datetime.now():%Y-%m-%d %H:%M}")

    # ── Render page ───────────────────────────────────────────────────────
    page_fn = PAGES.get(selected_page, chat_page)
    page_fn()


if __name__ == "__main__":
    main()
