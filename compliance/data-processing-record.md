# Registro de Actividades de Tratamiento — Pymerag

**Versión:** 1.0  
**Fecha:** 2026-05-17  
**Responsable del Tratamiento:** [A designar]  
**Encargado del Tratamiento:** Pymerag (Sistema RAG-MCP)  
**Legislación Aplicable:** Ley 25.326 de Protección de Datos Personales (Argentina)

---

## 1. Identificación del Responsable y Encargado

| Campo | Valor |
|-------|-------|
| **Razón Social** | Pymerag RAG-MCP Assistant |
| **Responsable del Tratamiento** | [Nombre del responsable legal] |
| **Domicilio Legal** | [Dirección física en Argentina] |
| **Datos de Contacto** | [Email y teléfono del DPO] |
| **Finalidad Principal** | Asistente inteligente para gestión documental con recuperación aumentada por generación (RAG) |
| **Base de Datos** | Pymerag DB (PostgreSQL) + Qdrant Vector Store |
| **Número de Registro AAIP** | [A completar al inscribir la base de datos] |

---

## 2. Cumplimiento por Artículo de la Ley 25.326

### Artículo 3 — Definiciones

> *Define datos personales, datos sensibles, titular, responsable, encargado, etc.*

**Aplicación en Pymerag:**
- **Datos personales tratados:** Texto de documentos ingeridos, consultas de usuarios, metadatos de documentos, registros de auditoría.
- **Datos sensibles:** Pymerag no está diseñado para tratar datos sensibles (salud, ideología, religión, etc.). Si un usuario ingiere documentos con datos sensibles, se considera un uso no previsto.
- **Titular de los datos:** El usuario que ingiere documentos o formula consultas.

---

### Artículo 4 — Calidad de los Datos

> *Los datos deben ser exactos, adecuados, pertinentes y no excesivos.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Exactitud | Los documentos se ingieren tal cual son provistos. No se modifica el contenido original. | ✅ Cumple |
| Adecuación | Solo se almacena el texto extraído y metadatos relevantes para la búsqueda. | ✅ Cumple |
| No excesivos | El chunking limita el contenido indexado a fragmentos de tamaño configurable (512 tokens por defecto). | ✅ Cumple |
| Actualización | No implementado — los documentos ingeridos no tienen mecanismo automático de actualización. | ⚠️ Pendiente |

**Recomendación:** Implementar endpoint `PUT /ingest/document/{id}` para re-ingesta de documentos actualizados, con eliminación de chunks previos.

---

### Artículo 5 — Consentimiento

> *El tratamiento de datos personales es lícito cuando el titular hubiere prestado su consentimiento libre, expreso e informado.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Consentimiento explícito | No implementado. No hay pantalla de consentimiento ni términos de uso en el frontend. | ❌ Pendiente |
| Finalidad específica | La finalidad (asistente RAG para gestión documental) está documentada pero no se informa al usuario. | ❌ Pendiente |
| Revocabilidad | No existe mecanismo para que el titular revoque el consentimiento y solicite eliminación de datos. | ❌ Pendiente |

**Recomendación:**
1. Agregar pantalla de consentimiento en el frontend Streamlit antes del primer uso.
2. Incluir checkbox: "He leído y acepto los términos de uso y la política de privacidad".
3. Implementar endpoint `DELETE /ingest/document/{id}` con eliminación en cascada (PostgreSQL + Qdrant).
4. Registrar el consentimiento en `AuditLog` con timestamp y versión de términos aceptados.

---

### Artículo 6 — Información al Titular

> *El responsable debe informar al titular sobre la finalidad, destinatarios, derechos y existencia de la base de datos.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Política de Privacidad | No existe documento de política de privacidad accesible al usuario. | ❌ Pendiente |
| Información sobre la base de datos | No se informa al titular que sus datos se almacenan en PostgreSQL y Qdrant. | ❌ Pendiente |
| Derechos ARCO | No se informa al titular sobre sus derechos de Acceso, Rectificación, Cancelación y Oposición. | ❌ Pendiente |

**Recomendación:** Crear archivo `PRIVACY.md` enlazado desde el frontend. Incluir: identidad del responsable, finalidad, destinatarios, derechos ARCO, y procedimiento para ejercerlos.

---

### Artículo 7 — Niveles de Protección

> *Los datos personales deben ser destruidos cuando hayan dejado de ser necesarios.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Conservación limitada | Los datos se conservan indefinidamente. No hay política de retención ni purga automática. | ❌ Pendiente |
| Destrucción al dejar de ser necesarios | Existe `delete_by_document` en `QdrantIndexer` pero no hay endpoint expuesto para que el usuario lo solicite. | ⚠️ Parcial |

**Recomendación:** Implementar política de retención configurable (`DATA_RETENTION_DAYS`). Agregar tarea programada que elimine documentos y chunks con antigüedad mayor al período configurado.

---

### Artículo 8 — Seguridad de los Datos

> *El responsable debe adoptar medidas técnicas y organizativas para garantizar la seguridad y confidencialidad de los datos.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Confidencialidad | **Pendiente:** No hay autenticación en la API. Los endpoints son públicos. | ❌ Pendiente |
| Confidencialidad en tránsito | **Pendiente:** Las conexiones entre servicios Docker son en texto plano (sin TLS). | ❌ Pendiente |
| Integridad | Los datos en PostgreSQL tienen integridad transaccional (ACID). Qdrant no garantiza integridad criptográfica. | ⚠️ Parcial |
| Control de acceso | **Pendiente:** No hay RBAC. Todos los endpoints son accesibles sin distinción de roles. | ❌ Pendiente |
| Trazabilidad | **Implementado:** Tabla `audit_logs` registra acciones del sistema (QUERY_EXECUTE, etc.). | ✅ Cumple |
| Protección contra acceso no autorizado | **Pendiente:** No hay autenticación, rate limiting, ni WAF. | ❌ Pendiente |
| Copias de respaldo | No implementado. No hay política de backups para PostgreSQL ni Qdrant. | ❌ Pendiente |

**Ver threat-model.md para el análisis detallado STRIDE.**

---

### Artículo 9 — Deber de Secreto

> *El responsable y encargado deben guardar secreto profesional sobre los datos.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Confidencialidad del personal | No aplica (sistema automatizado). Los logs de auditoría son accesibles sin restricción. | ❌ Pendiente |
| Acuerdos de confidencialidad | No aplica directamente al software. Debe implementarse como política organizacional. | ⚠️ Fuera de alcance |

**Recomendación:** Proteger los endpoints de auditoría (`GET /admin/audit`) con autenticación y autorización de rol `admin`.

---

### Artículo 10 — Deber de Seguridad (Detallado)

Ver sección 2.6 del threat model (STRIDE). Controles técnicos específicos:

| Control | Implementación Técnica | Estado |
|---------|------------------------|--------|
| TLS para API externa | No implementado. Se recomienda reverse proxy con Nginx + Let's Encrypt. | ❌ Pendiente |
| Hashing de contraseñas | No aplica aún (sin autenticación). Se usará `bcrypt` para passwords de usuarios. | 📋 Planificado |
| Sanitización de inputs | Parcial: Pydantic valida tipos pero no contenido. Sin protección anti prompt injection. | ⚠️ Parcial |
| Rate limiting | No implementado. Ver D-01 en threat-model.md. | ❌ Pendiente |
| Logs sin datos sensibles | **Crítico:** `database_url` se loggea con credenciales. Consultas se almacenan en texto plano. | ❌ No cumple |

---

### Artículo 11 — Transferencia Internacional

> *La transferencia de datos personales a países sin protección adecuada requiere consentimiento o excepciones.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Transferencia a terceros países | No se identifican transferencias internacionales automáticas. Los datos residen en infraestructura local (Docker on-premise). | ✅ Cumple |
| Uso de APIs externas | `llama.cpp` es auto-hospedado (no envía datos a OpenAI/Anthropic). `BGE-M3` se ejecuta localmente. | ✅ Cumple |
| Qdrant Cloud | `qdrant_api_key` en configuración permite conexión a Qdrant Cloud. Si se usa, hay transferencia internacional. | ⚠️ Condicional |
| Langfuse | `langfuse_host` configurable. Si apunta a cloud (https://cloud.langfuse.com), hay transferencia a UE (GDPR-adecuado). | ⚠️ Condicional |

**Recomendación:** Si se despliega Qdrant Cloud o Langfuse Cloud, documentar la transferencia y verificar que el país de destino tenga nivel adecuado según Disposición AAIP.

---

### Artículos 12-13 — Videovigilancia y Prestadores

No aplican directamente a Pymerag (no procesa imágenes de videovigilancia ni actúa como prestador de servicios de información crediticia).

---

### Artículo 14 — Derecho de Acceso

> *El titular puede solicitar y obtener información sobre sus datos personales.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Endpoint de acceso | No implementado. No hay endpoint para que un usuario consulte qué datos tiene almacenados el sistema sobre él. | ❌ Pendiente |
| Gratuidad | — | 📋 Planificado |
| Plazo de respuesta | — | 📋 Planificado |

**Recomendación:** Implementar `GET /api/v1/user/data` que retorne documentos ingeridos por el usuario autenticado y registros de auditoría asociados.

---

### Artículo 15 — Derecho de Rectificación

> *El titular puede solicitar la rectificación de datos inexactos.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Corrección de datos | No implementado. Los documentos ingeridos no pueden modificarse. | ❌ Pendiente |
| Corrección de metadatos | No hay endpoint `PATCH` para modificar metadatos de documentos o chunks. | ❌ Pendiente |

**Recomendación:** Implementar `PATCH /ingest/document/{id}` para actualizar metadatos. Para contenido de documentos, el flujo natural es re-ingesta.

---

### Artículo 16 — Derecho de Supresión (Cancelación)

> *El titular puede solicitar la eliminación de sus datos.*

| Requisito | Control Técnico Implementado | Estado |
|-----------|------------------------------|--------|
| Eliminación de documentos | `QdrantIndexer.delete_by_document()` existe pero no está expuesto como endpoint. | ⚠️ Parcial |
| Eliminación en cascada | PostgreSQL tiene foreign key `chunks.document_id → documents.id`. Al eliminar un documento, los chunks en PostgreSQL quedan huérfanos (no hay `ON DELETE CASCADE`). | ❌ No cumple |
| Plazo de ejecución | — | 📋 Planificado |

**Recomendación:**
1. Agregar `ON DELETE CASCADE` en la FK de `chunks.document_id`.
2. Exponer `DELETE /api/v1/ingest/document/{id}` que elimine en PostgreSQL y Qdrant.
3. Registrar la eliminación en `AuditLog` con acción `DOCUMENT_DELETE`.

---

### Artículo 17 — Derecho de Oposición

El titular puede oponerse al tratamiento de sus datos por motivos legítimos. En Pymerag, esto equivale a solicitar la eliminación. Ver Art. 16.

---

### Artículo 18 — Excepciones

No aplican excepciones al consentimiento en Pymerag. El tratamiento actual es para fines privados/de investigación, no cubierto por las excepciones del artículo 18 (fuentes públicas, obligación legal, etc.).

---

### Artículos 19-27 — Sanciones, Acción de Habeas Data, Autoridad de Control

Estos artículos establecen el régimen sancionatorio y la competencia de la AAIP. No requieren controles técnicos adicionales, pero el responsable debe:
- Inscribir la base de datos en el Registro Nacional de Bases de Datos de la AAIP.
- Designar un Delegado de Protección de Datos (DPO).
- Establecer procedimiento interno para responder solicitudes ARCO en el plazo legal.

---

## 3. Inventario de Datos

| Categoría de Datos | Ubicación | Formato | Cifrado en Reposo | Cifrado en Tránsito | Período de Retención |
|-------------------|-----------|---------|-------------------|---------------------|---------------------|
| Texto de documentos (chunks) | Qdrant | Texto plano | ❌ No | ❌ No (intra-Docker) | Indefinido |
| Metadatos de documentos | PostgreSQL (`documents`) | JSON | ❌ No | ❌ No (intra-Docker) | Indefinido |
| Fragmentos (chunks) | PostgreSQL (`chunks`) | Texto | ❌ No | ❌ No (intra-Docker) | Indefinido |
| Consultas de usuarios | PostgreSQL (`audit_logs`) | JSON | ❌ No | ❌ No (intra-Docker) | Indefinido |
| Tópicos descubiertos | PostgreSQL (`topics`) | JSON | ❌ No | ❌ No (intra-Docker) | Indefinido |
| Logs de aplicación | stdout / Docker logs | Texto/JSON | ❌ No | N/A | Hasta rotación de logs |

---

## 4. Subencargados del Tratamiento

| Subencargado | Servicio | País | Garantías |
|-------------|----------|------|-----------|
| Qdrant Cloud (opcional) | Vector Database | Alemania / USA | GDPR + contrato de encargo |
| Langfuse Cloud (opcional) | Observabilidad | UE | GDPR |
| Docker / Infraestructura propia | Orquestación | Argentina (on-premise) | Control físico del responsable |

---

## 5. Evaluación de Impacto en Protección de Datos (EIPD / DPIA)

**¿Es necesaria una EIPD?**  
No se requiere obligatoriamente para Pymerag según la Ley 25.326, ya que no realiza:
- Evaluaciones sistemáticas de aspectos personales (profiling automatizado).
- Tratamiento a gran escala de datos sensibles.
- Vigilancia sistemática de zonas públicas.

Sin embargo, **se recomienda realizar una EIPD simplificada** si el sistema escala para procesar documentos legales, médicos o financieros con datos personales de terceros.

---

## 6. Cumplimiento Técnico — Resumen

| Control | Estado |
|---------|--------|
| Autenticación de usuarios | ❌ No implementado |
| Control de acceso (RBAC) | ❌ No implementado |
| Cifrado en tránsito (TLS) | ❌ No implementado |
| Cifrado en reposo | ❌ No implementado |
| Rate limiting | ❌ No implementado |
| Sanitización de inputs | ⚠️ Parcial |
| Logs sin datos sensibles | ❌ No cumple (I-01) |
| Auditoría y trazabilidad | ✅ Implementado |
| Consentimiento informado | ❌ No implementado |
| Derechos ARCO | ❌ No implementado |
| Transferencia internacional | ✅ No aplica (local) |
| Backups y disaster recovery | ❌ No implementado |

---

## 7. Plan de Acción para Cumplimiento

### Prioridad 1 — Cumplimiento Legal Mínimo
1. Redactar y publicar Política de Privacidad
2. Implementar consentimiento informado en frontend
3. Inscribir base de datos ante AAIP

### Prioridad 2 — Seguridad Técnica Básica
4. Implementar autenticación (JWT)
5. Habilitar TLS en API externa
6. Corregir fuga de credenciales en logs (I-01)
7. Implementar eliminación de datos (derecho de supresión)

### Prioridad 3 — Cumplimiento Integral
8. Implementar rate limiting
9. Implementar endpoints ARCO
10. Establecer política de retención y purga automática
11. Realizar EIPD si se escala el sistema

---

## 8. Referencias Legales

- [Ley 25.326 — Protección de Datos Personales](http://servicios.infoleg.gob.ar/infolegInternet/anexos/60000-64999/64790/norma.htm)
- [Decreto 1558/2001 — Reglamentación Ley 25.326](http://servicios.infoleg.gob.ar/infolegInternet/anexos/70000-74999/70368/norma.htm)
- [Disposición 60-E/2016 AAIP — Transferencia Internacional](https://www.argentina.gob.ar/aaip)
- [Resolución 14/2018 AAIP — Medidas de Seguridad](https://www.argentina.gob.ar/aaip)
