# Modelo de Amenazas — Pymerag (STRIDE)

**Versión:** 1.0  
**Fecha:** 2026-05-17  
**Autor:** Security & Compliance Officer (Agente A6)  
**Alcance:** Sistema completo — API REST, Pipeline RAG, Frontend Streamlit, Infraestructura Docker, Servidor MCP

---

## 1. Resumen Ejecutivo

Este documento presenta el análisis de amenazas del sistema Pymerag utilizando la metodología STRIDE (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege), conforme a las mejores prácticas de OWASP y los requisitos de la Ley 25.326 de Protección de Datos Personales de Argentina.

El sistema Pymerag es un asistente inteligente RAG-MCP (Retrieval-Augmented Generation con Model Context Protocol) para gestión documental. Su arquitectura se compone de:

- **API REST (FastAPI):** Backend principal que expone endpoints de ingesta, consulta, tópicos y administración.
- **Pipeline RAG:** Recuperación híbrida (dense + sparse) sobre Qdrant, generación de respuestas con LLM (llama.cpp).
- **Frontend (Streamlit):** Interfaz web para usuarios finales.
- **Infraestructura Docker:** Contenedores para API, Qdrant, PostgreSQL y llama.cpp.
- **Servidor MCP:** Integración con agentes externos (en desarrollo).

---

## 2. Análisis STRIDE

### 2.1 Spoofing (Suplantación de Identidad)

| ID | Amenaza | Componente Afectado | Severidad | Mitigación | Estado |
|----|---------|---------------------|-----------|------------|--------|
| S-01 | **Falta de autenticación en la API**: Todos los endpoints (`/api/v1/*`) son accesibles sin credenciales. Un atacante puede suplantar a cualquier usuario legítimo. | API (FastAPI) | **Crítica** | Implementar OAuth2/JWT con FastAPI Security. Agregar `Depends(get_current_user)` en todos los endpoints. Mínimo: API Key por header `X-API-Key`. | ❌ Pendiente |
| S-02 | **CORS demasiado permisivo**: `allow_origins` configurable pero con valores por defecto amplios. `allow_credentials=True` + `allow_methods=["*"]` permite ataques CSRF desde orígenes maliciosos. | API (main.py) | **Alta** | Restringir `allow_origins` a dominios específicos de producción. Evaluar necesidad de `allow_credentials=True`. | ⚠️ Parcial |
| S-03 | **Falta de autenticación en frontend**: Streamlit no implementa capa de autenticación. Cualquier persona con acceso a la URL puede ejecutar consultas e ingesta. | Frontend (Streamlit) | **Alta** | Agregar autenticación Streamlit (OAuth, contraseña de aplicación, o integración SSO). | ❌ Pendiente |
| S-04 | **MCP Server sin autenticación**: Los placeholders de MCP (`server.py`, `tools.py`) no implementan seguridad. | MCP Server | **Media** | Implementar token de acceso MCP según especificación del protocolo. | ❌ Pendiente |

### 2.2 Tampering (Manipulación de Datos)

| ID | Amenaza | Componente Afectado | Severidad | Mitigación | Estado |
|----|---------|---------------------|-----------|------------|--------|
| T-01 | **Prompt Injection en consultas RAG**: Las consultas del usuario se concatenan directamente en el prompt del LLM sin sanitización (`_build_prompt` en `routes_query.py:105`). Un atacante puede inyectar instrucciones para eludir restricciones o extraer datos del contexto. | API / RAG Pipeline | **Crítica** | Implementar sanitización de input: delimitar claramente instrucciones del sistema vs. datos de usuario. Usar `\n\n---\n\n` como separador explícito. Validar longitud máxima de consulta. Considerar usar LLM guardrails. | ❌ Pendiente |
| T-02 | **Validación insuficiente en ingesta**: `source_path` en `IngestRequest` acepta rutas arbitrarias sin sanitización (`routes_ingest.py:53`). Un atacante podría leer archivos del sistema (`/etc/passwd`, `../../secrets`). | API (routes_ingest.py) | **Alta** | Validar que la ruta esté dentro del directorio de datos autorizado. Bloquear path traversal (`../`). Usar `os.path.realpath()` para resolver enlaces simbólicos. | ❌ Pendiente |
| T-03 | **Conexiones sin TLS en Docker**: La comunicación entre servicios Docker (API → Qdrant, API → LLM) es en texto plano. Un atacante con acceso a la red `pymerag-net` puede interceptar tráfico. | Infraestructura Docker | **Media** | Habilitar TLS mutuo entre servicios o usar redes overlay encriptadas. Para desarrollo local el riesgo es bajo. | ⚠️ Aceptado (dev) |
| T-04 | **Metadatos de ingesta sin validación**: El campo `metadata` en `IngestRequest` acepta JSON arbitrario sin sanitización. Podría usarse para inyectar payloads maliciosos. | API (routes_ingest.py) | **Baja** | Validar tipos de datos en metadatos. Limitar profundidad del JSON. Sanitizar claves antes de indexar en Qdrant. | ❌ Pendiente |

### 2.3 Repudiation (Repudio / No repudio)

| ID | Amenaza | Componente Afectado | Severidad | Mitigación | Estado |
|----|---------|---------------------|-----------|------------|--------|
| R-01 | **Audit Log sin integridad criptográfica**: Los registros de auditoría (`AuditLog` en PostgreSQL) no tienen firma digital ni hash encadenado. Un administrador malicioso podría modificar o eliminar registros. | Base de datos (PostgreSQL) | **Alta** | Implementar hash encadenado (blockchain-style) en AuditLog. Usar `HMAC-SHA256` con clave separada. Alternativa: append-only table con permisos restringidos. | ❌ Pendiente |
| R-02 | **Falta de trazabilidad de usuario**: El campo `user_id` en `AuditLog` se registra como `"api"` (valor fijo en `routes_query.py:201`). No se puede atribuir acciones a usuarios específicos. | API (routes_query.py) | **Media** | Una vez implementada autenticación (S-01), usar el ID real del usuario autenticado. | ❌ Pendiente |
| R-03 | **Logs de aplicación sin respaldo**: Los logs de Python (`logging`) van a stdout. En Docker se pierden al reiniciar el contenedor. No hay persistencia de logs. | Sistema de logging | **Media** | Configurar driver de logs de Docker (`json-file` o `fluentd`). Alternativa: enviar a Langfuse/ELK stack. | ⚠️ Parcial |

### 2.4 Information Disclosure (Divulgación de Información)

| ID | Amenaza | Componente Afectado | Severidad | Mitigación | Estado |
|----|---------|---------------------|-----------|------------|--------|
| I-01 | **URL de base de datos expuesta en logs**: `main.py:48` registra `settings.database_url` completo, incluyendo credenciales si las tiene. Ej: `postgresql://user:password@host/db`. | API (main.py) | **Crítica** | **URGENTE**: Enmascarar contraseña en logs. Usar `settings.database_url.replace(settings.database_url.password, '***')` o loggear solo host/db. | ❌ Pendiente |
| I-02 | **Consultas de usuarios en logs y auditoría**: `_audit_query` (`routes_query.py:182`) almacena la consulta del usuario (hasta 500 chars) y preview de respuesta (200 chars) en la tabla `audit_logs`. Si las consultas contienen datos personales (nombres, DNI, CUIT), se almacenan en texto plano. | API / DB | **Alta** | Implementar política de no registrar contenido de consultas. Alternativa: almacenar solo hash o versión anonimizada. Agregar filtro de PII antes de almacenar. | ❌ Pendiente |
| I-03 | **Health check expone detalles de infraestructura**: `GET /admin/health` revela nombres de servicios, latencias y mensajes de error. Facilita reconocimiento por atacantes. | API (routes_admin.py) | **Media** | Requerir autenticación para el endpoint de health. Limitar información en respuesta al rol del solicitante. Mantener health básico (`/healthz`) sin detalles. | ❌ Pendiente |
| I-04 | **Endpoints de auditoría sin protección**: `GET /admin/audit` expone todos los registros de auditoría sin restricción de acceso. | API (routes_admin.py) | **Alta** | Requerir autenticación y autorización (rol admin) para acceder a logs de auditoría. | ❌ Pendiente |
| I-05 | **Datos sensibles en Qdrant sin encriptación**: Los chunks de documentos se almacenan en Qdrant en texto plano. Si los documentos contienen datos personales, no están protegidos en reposo. | Qdrant | **Media** | Evaluar Qdrant encryption-at-rest. Para datos sensibles, encriptar el campo `content` antes de indexar (con clave derivada). | ⚠️ Aceptado (fase 1) |
| I-06 | **Error messages con detalles internos**: Los endpoints retornan mensajes de error con detalles de archivo/ruta (`routes_ingest.py:192`). En producción esto revela estructura del filesystem. | API | **Baja** | En producción, retornar mensajes genéricos sin rutas internas. Usar IDs de error para correlación en logs. | ❌ Pendiente |

### 2.5 Denial of Service (Denegación de Servicio)

| ID | Amenaza | Componente Afectado | Severidad | Mitigación | Estado |
|----|---------|---------------------|-----------|------------|--------|
| D-01 | **Falta de rate limiting**: Ningún endpoint implementa limitación de tasa. Un atacante puede saturar la API con consultas repetidas, agotando recursos del LLM y Qdrant. | API (todos los endpoints) | **Alta** | Implementar `slowapi` o middleware de rate limiting. Configurar límites por endpoint: `/query`: 30 req/min, `/ingest`: 10 req/min, `/admin`: 60 req/min. | ❌ Pendiente |
| D-02 | **Sin límite de tamaño en consultas**: `query` en `QueryRequest` solo tiene `min_length=1`, sin `max_length`. Un prompt extremadamente largo (>100K tokens) puede saturar el LLM. | API (routes_query.py) | **Alta** | Agregar `max_length=2000` en `QueryRequest.query`. Limitar `top_k` a 20 en producción. | ❌ Pendiente |
| D-03 | **Sin timeout en ingesta de directorios grandes**: `ingest_directory` procesa archivos secuencialmente sin límite de tiempo ni concurrencia. Un directorio con miles de archivos PDF puede bloquear el worker. | API / Pipeline | **Media** | Agregar timeout por archivo y límite de archivos por trabajo. Implementar procesamiento con cola (Celery/Redis). | ❌ Pendiente |
| D-04 | **Health check sin caching**: Cada llamada a `/admin/health` crea nuevas conexiones a Qdrant y DB. Múltiples health checks simultáneos pueden degradar el rendimiento. | API (routes_admin.py) | **Baja** | Implementar caching de health status con TTL de 10 segundos. Reutilizar conexiones en lugar de crear nuevas cada vez. | ❌ Pendiente |
| D-05 | **Sin límite de conexiones en PostgreSQL**: No hay connection pooling configurado. Cada worker crea su propio `create_engine`. | API / DB | **Media** | Usar `SQLModel` con pool_size y max_overflow configurados. Implementar engine singleton en lugar de crear engines por request. | ❌ Pendiente |

### 2.6 Elevation of Privilege (Elevación de Privilegios)

| ID | Amenaza | Componente Afectado | Severidad | Mitigación | Estado |
|----|---------|---------------------|-----------|------------|--------|
| E-01 | **Sin control de acceso basado en roles (RBAC)**: No hay distinción entre administradores y usuarios. Cualquiera puede acceder a `/admin/*`, ejecutar ingesta, y ver auditoría. | API | **Crítica** | Implementar RBAC: roles `admin`, `user`, `readonly`. Proteger endpoints admin con `Depends(require_role("admin"))`. | ❌ Pendiente |
| E-02 | **Path traversal en ingesta de archivos**: No se valida que `source_path` esté dentro del filesystem autorizado. Teóricamente permite leer cualquier archivo accesible al proceso. | API (routes_ingest.py) | **Alta** | Validar canónico: `Path(source_path).resolve()` debe estar dentro de `settings.data_dir.resolve()`. Rechazar paths con `..`. | ❌ Pendiente |
| E-03 | **Configuración de Docker con privilegios innecesarios**: El contenedor `llm` requiere GPU NVIDIA (`deploy.resources.reservations.devices`). Si se compromete, tiene acceso directo a la GPU del host. | Infraestructura Docker | **Media** | Aislar el contenedor LLM con `security_opt: no-new-privileges:true`. Usar `user: 1000:1000` en lugar de root. Montar modelos como `:ro`. | ⚠️ Parcial |
| E-04 | **PostgreSQL con contraseña por defecto**: El `docker-compose.yml` define `POSTGRES_PASSWORD: "${POSTGRES_PASSWORD:-pymerag}"`. Si no se configura variable de entorno, la contraseña es "pymerag". | Infraestructura Docker | **Alta** | No establecer default de contraseña. Requerir variable de entorno obligatoria. Generar contraseña aleatoria en first-run. | ❌ Pendiente |
| E-05 | **Ejecución como root en contenedor API**: El `Dockerfile` no especifica usuario no-root. El proceso uvicorn corre como root dentro del contenedor. | Dockerfile | **Media** | Agregar `USER 1000:1000` al final del Dockerfile. Asegurar permisos de `/app/data` para el usuario no-root. | ❌ Pendiente |

---

## 3. Resumen de Riesgos

| Nivel | Cantidad | IDs |
|-------|----------|-----|
| 🔴 Crítica | 4 | S-01, T-01, I-01, E-01 |
| 🟠 Alta | 10 | S-02, S-03, T-02, R-01, I-02, I-04, D-01, D-02, E-02, E-04 |
| 🟡 Media | 10 | S-04, T-03, R-02, R-03, I-03, I-05, D-03, D-05, E-03, E-05 |
| 🟢 Baja | 3 | T-04, I-06, D-04 |

---

## 4. Plan de Remediación Priorizado

### Fase 1 — Inmediata (antes de producción)
1. **I-01**: Enmascarar contraseña de base de datos en logs
2. **S-01**: Implementar autenticación básica (API Key o JWT)
3. **E-01**: Implementar RBAC mínimo (admin vs user)
4. **T-01**: Sanitización de prompts contra inyección
5. **E-04**: Eliminar contraseña por defecto en PostgreSQL

### Fase 2 — Corto plazo (siguiente sprint)
6. **D-01**: Rate limiting en todos los endpoints
7. **I-02**: Anonimizar consultas en audit logs
8. **T-02**: Validación de path traversal en ingesta
9. **E-02**: Restricción de acceso a filesystem
10. **I-04**: Proteger endpoint de auditoría

### Fase 3 — Mediano plazo
11. **R-01**: Integridad criptográfica en audit logs
12. **S-03**: Autenticación en frontend Streamlit
13. **D-05**: Connection pooling en PostgreSQL
14. **E-05**: Ejecutar contenedores como no-root

---

## 5. Referencias

- [OWASP Top 10 API Security Risks (2023)](https://owasp.org/API-Security/editions/2023/)
- [STRIDE Methodology — Microsoft](https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats)
- [Ley 25.326 — Protección de Datos Personales (Argentina)](http://servicios.infoleg.gob.ar/infolegInternet/anexos/60000-64999/64790/norma.htm)
- [CWE-89: SQL Injection](https://cwe.mitre.org/data/definitions/89.html)
- [CWE-22: Path Traversal](https://cwe.mitre.org/data/definitions/22.html)
