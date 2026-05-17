"""
Tests para los endpoints de la API REST de Pymerag.

Valida ingesta, consulta RAG, health check y auditoría
usando el TestClient de FastAPI con servicios mockeados.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# ── Fixtures locales ────────────────────────────────────────────────


@pytest.fixture
def client(test_app: TestClient) -> TestClient:
    """Cliente de test de FastAPI (alias del fixture de conftest)."""
    return test_app


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> str:
    """Crea un archivo PDF de muestra (con contenido mínimo válido)."""
    pdf_path = tmp_path / "test_doc.pdf"
    # Escribir un PDF mínimo para que Docling pueda procesarlo
    # Un PDF mínimo con texto "Hello PDF"
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"  # noqa: E501
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF"
    )
    pdf_path.write_bytes(minimal_pdf)
    return str(pdf_path)


@pytest.fixture
def sample_txt_path(tmp_path: Path) -> str:
    """Crea un archivo de texto de muestra para pruebas de ingesta."""
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text(
        "Python es un lenguaje de programación de alto nivel. "
        "Fue creado por Guido van Rossum y lanzado en 1991. "
        "Python se caracteriza por su sintaxis clara y legible, "
        "lo que lo hace ideal para principiantes y expertos.",
        encoding="utf-8",
    )
    return str(txt_path)


# ── Tests del endpoint raíz ─────────────────────────────────────────


class TestRootEndpoint:
    """Pruebas para el endpoint raíz (GET /)."""

    def test_root_returns_service_info(self, client: TestClient) -> None:
        """El endpoint raíz debe retornar metadatos del servicio."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Pymerag API"
        assert "version" in data
        assert "docs" in data


# ── Tests de ingesta ────────────────────────────────────────────────


class TestIngestAPI:
    """Pruebas para los endpoints de ingesta de documentos."""

    API_PREFIX = "/api/v1"

    def test_ingest_file_accepted(
        self, client: TestClient, sample_txt_path: str
    ) -> None:
        """POST /ingest/document con archivo válido retorna 202 Accepted."""
        response = client.post(
            f"{self.API_PREFIX}/ingest/document",
            json={
                "source_type": "file",
                "source_path": sample_txt_path,
                "metadata": {"project": "pymerag-test"},
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_ingest_file_not_found(self, client: TestClient) -> None:
        """POST /ingest/document con archivo inexistente retorna 400."""
        response = client.post(
            f"{self.API_PREFIX}/ingest/document",
            json={
                "source_type": "file",
                "source_path": "/nonexistent/file.pdf",
            },
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "no encontrado" in detail.lower() or "not found" in detail.lower()

    def test_ingest_directory_not_found(self, client: TestClient) -> None:
        """POST /ingest/document con directorio inexistente retorna 400."""
        response = client.post(
            f"{self.API_PREFIX}/ingest/document",
            json={
                "source_type": "directory",
                "source_path": "/nonexistent/dir",
            },
        )
        assert response.status_code == 400

    def test_ingest_empty_query_returns_422(self, client: TestClient) -> None:
        """POST /ingest/document sin source_path debe retornar 422."""
        response = client.post(
            f"{self.API_PREFIX}/ingest/document",
            json={"source_type": "file"},
        )
        assert response.status_code == 422

    def test_ingest_url_type_accepted(self, client: TestClient) -> None:
        """POST /ingest/document con tipo URL es aceptado sin validar existencia."""
        response = client.post(
            f"{self.API_PREFIX}/ingest/document",
            json={
                "source_type": "url",
                "source_path": "https://example.com/doc.pdf",
            },
        )
        assert response.status_code == 202

    def test_get_ingest_status_queued(
        self, client: TestClient, sample_txt_path: str
    ) -> None:
        """GET /ingest/status/{job_id} retorna estado inicial 'queued'."""
        # Primero crear un trabajo
        ingest_resp = client.post(
            f"{self.API_PREFIX}/ingest/document",
            json={
                "source_type": "file",
                "source_path": sample_txt_path,
            },
        )
        job_id = ingest_resp.json()["job_id"]

        # Consultar estado
        response = client.get(f"{self.API_PREFIX}/ingest/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] in ("queued", "processing", "completed", "failed")
        assert "progress" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_get_ingest_status_not_found(self, client: TestClient) -> None:
        """GET /ingest/status/{job_id} con ID inexistente retorna 404."""
        response = client.get(
            f"{self.API_PREFIX}/ingest/status/nonexistent-job-id"
        )
        assert response.status_code == 404

    def test_get_ingest_status_after_waiting(
        self, client: TestClient, sample_txt_path: str
    ) -> None:
        """GET /ingest/status/{job_id} tras esperar retorna estado final."""
        ingest_resp = client.post(
            f"{self.API_PREFIX}/ingest/document",
            json={
                "source_type": "file",
                "source_path": sample_txt_path,
            },
        )
        job_id = ingest_resp.json()["job_id"]

        # Esperar un poco para que el background task procese
        time.sleep(1.0)

        response = client.get(f"{self.API_PREFIX}/ingest/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        # Debería haber terminado (completed, failed, o empty)
        assert data["status"] in ("completed", "failed", "empty")


# ── Tests de consulta RAG ───────────────────────────────────────────


class TestQueryAPI:
    """Pruebas para el endpoint de consulta RAG."""

    API_PREFIX = "/api/v1"

    def test_query_returns_response(self, client: TestClient) -> None:
        """POST /query retorna una respuesta con fuentes y metadatos."""
        response = client.post(
            f"{self.API_PREFIX}/query",
            json={
                "query": "¿Qué es Python?",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "metadata" in data
        assert isinstance(data["sources"], list)
        assert "latency" in data["metadata"]
        assert "sources_count" in data["metadata"]

    def test_query_with_filters(self, client: TestClient) -> None:
        """POST /query acepta filtros opcionales."""
        response = client.post(
            f"{self.API_PREFIX}/query",
            json={
                "query": "machine learning",
                "top_k": 5,
                "filters": {"document_id": "test-doc-123"},
            },
        )
        assert response.status_code == 200

    def test_query_with_top_k_boundaries(self, client: TestClient) -> None:
        """POST /query acepta valores de top_k dentro del rango [1, 50]."""
        for k in [1, 25, 50]:
            response = client.post(
                f"{self.API_PREFIX}/query",
                json={"query": "test query", "top_k": k},
            )
            assert response.status_code == 200

    def test_query_top_k_out_of_range(self, client: TestClient) -> None:
        """POST /query con top_k fuera de rango retorna 422."""
        # top_k = 0 (menor que 1)
        response = client.post(
            f"{self.API_PREFIX}/query",
            json={"query": "test", "top_k": 0},
        )
        assert response.status_code == 422

        # top_k = 51 (mayor que 50)
        response = client.post(
            f"{self.API_PREFIX}/query",
            json={"query": "test", "top_k": 100},
        )
        assert response.status_code == 422

    def test_query_empty_string(self, client: TestClient) -> None:
        """POST /query con consulta vacía retorna 422."""
        response = client.post(
            f"{self.API_PREFIX}/query",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_query_response_structure_complete(
        self, client: TestClient
    ) -> None:
        """La respuesta de consulta tiene la estructura completa esperada."""
        response = client.post(
            f"{self.API_PREFIX}/query",
            json={"query": "RAG retrieval augmented generation"},
        )
        data = response.json()

        # Verificar estructura de sources si hay
        for source in data["sources"]:
            assert "chunk_id" in source
            assert "content" in source
            assert "metadata" in source
            assert "score" in source
            assert isinstance(source["score"], (int, float))
            assert 0.0 <= source["score"] <= 1.0

        # Verificar metadatos
        meta = data["metadata"]
        assert meta["sources_count"] == len(data["sources"])
        assert meta["model"] == "deepseek-v4"


# ── Tests de administración ─────────────────────────────────────────


class TestAdminAPI:
    """Pruebas para los endpoints de administración y monitoreo."""

    API_PREFIX = "/api/v1"

    def test_health_check_returns_200(self, client: TestClient) -> None:
        """GET /admin/health retorna 200 con estado de servicios."""
        response = client.get(f"{self.API_PREFIX}/admin/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("ok", "degraded")
        assert "services" in data
        assert "timestamp" in data

        # Verificar servicios esperados
        services = data["services"]
        for svc_name in ("qdrant", "llm", "db"):
            assert svc_name in services, f"Falta servicio '{svc_name}'"
            svc = services[svc_name]
            assert "status" in svc
            assert svc["status"] in ("up", "down")

    def test_audit_logs_returns_list(self, client: TestClient) -> None:
        """GET /admin/audit retorna lista de entradas de auditoría."""
        response = client.get(f"{self.API_PREFIX}/admin/audit")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_audit_logs_with_limit(self, client: TestClient) -> None:
        """GET /admin/audit?limit=10 retorna máximo 10 entradas."""
        response = client.get(f"{self.API_PREFIX}/admin/audit?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 10

    def test_audit_logs_with_action_filter(self, client: TestClient) -> None:
        """GET /admin/audit?action=QUERY_EXECUTE filtra por acción."""
        response = client.get(
            f"{self.API_PREFIX}/admin/audit?action=QUERY_EXECUTE"
        )
        assert response.status_code == 200
        data = response.json()
        for entry in data:
            assert entry["action"] == "QUERY_EXECUTE"

    def test_audit_logs_limit_out_of_range(self, client: TestClient) -> None:
        """GET /admin/audit?limit=1000 retorna 422 (fuera de rango)."""
        response = client.get(f"{self.API_PREFIX}/admin/audit?limit=1000")
        assert response.status_code == 422

    def test_health_check_timestamp_is_iso(self, client: TestClient) -> None:
        """El timestamp del health check debe ser ISO 8601."""
        response = client.get(f"{self.API_PREFIX}/admin/health")
        data = response.json()
        # Verificar que contiene T (formato ISO 8601)
        assert "T" in data["timestamp"]


# ── Tests de integración: flujo completo ────────────────────────────


class TestIntegrationFlow:
    """Pruebas de integración del flujo ingesta → consulta."""

    API_PREFIX = "/api/v1"

    def test_full_ingest_then_query_flow(
        self, client: TestClient, sample_txt_path: str
    ) -> None:
        """Flujo completo: ingerir documento, esperar, luego consultar."""
        # 1. Ingerir documento
        ingest_resp = client.post(
            f"{self.API_PREFIX}/ingest/document",
            json={
                "source_type": "file",
                "source_path": sample_txt_path,
                "metadata": {"project": "integration-test"},
            },
        )
        assert ingest_resp.status_code == 202
        job_id = ingest_resp.json()["job_id"]

        # 2. Esperar procesamiento
        time.sleep(1.5)
        status_resp = client.get(f"{self.API_PREFIX}/ingest/status/{job_id}")
        assert status_resp.status_code == 200

        # 3. Ejecutar consulta (siempre debe responder, incluso sin resultados)
        query_resp = client.post(
            f"{self.API_PREFIX}/query",
            json={
                "query": "Python lenguaje de programación",
                "top_k": 5,
            },
        )
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert "answer" in query_data

    def test_multiple_ingest_jobs(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Múltiples trabajos de ingesta funcionan concurrentemente."""
        # Crear varios archivos
        files = []
        for i in range(3):
            fpath = tmp_path / f"doc_{i}.txt"
            fpath.write_text(f"Documento número {i}. Contenido de prueba.", encoding="utf-8")
            files.append(str(fpath))

        job_ids = []
        for fpath in files:
            resp = client.post(
                f"{self.API_PREFIX}/ingest/document",
                json={"source_type": "file", "source_path": fpath},
            )
            assert resp.status_code == 202
            job_ids.append(resp.json()["job_id"])

        # Todos los IDs deben ser únicos
        assert len(set(job_ids)) == len(job_ids)
