# Pymerag: Asistente Inteligente RAG-MCP para Gestión Documental

[![CI Status](https://github.com/lilidl-nft/pymerag-thesis/actions/workflows/ci.yml/badge.svg)](https://github.com/lilidl-nft/pymerag-thesis/actions)

**Pymerag** es un prototipo de asistente inteligente diseñado para la gestión y consulta de documentación corporativa heterogénea en pequeñas y medianas empresas (PyMEs) tecnológicas. El proyecto implementa una arquitectura híbrida de vanguardia que combina el paradigma de **Retrieval-Augmented Generation (RAG)** con el **Model Context Protocol (MCP)** para ofrecer una solución de búsqueda semántica, resumen y generación de contenido con trazabilidad completa.

Este proyecto forma parte de una investigación de maestría enfocada en la eficiencia de sistemas de gestión de conocimiento mediante modelos de lenguaje de código abierto.

---

## 🧠 Arquitectura del Sistema

El sistema se basa en una organización por capas funcionales que garantizan el desacoplamiento y la escalabilidad:

1.  **Capa de Ingesta:** Procesamiento de documentos (PDF, PPTX, DOCX) mediante **Docling** para una extracción jerárquica de alta fidelidad.
2.  **Capa de Indexación:** Almacenamiento de vectores en una base de datos híbrida (**Qdrant**) utilizando embeddings multilingües (**BGE-M3**).
3.  **Capa de Orquestación (El Cerebro):** Un grafo de estados implementado con **LangGraph** que gestiona el flujo de razonamiento, la recuperación de información y la validación de citas.
4.  **Capa de Interfaz de Herramientas (MCP):** Exposición de capacidades operativas (búsqueda, comparación, resumen) mediante un servidor **FastMCP**, permitiendo que agentes externos utilicen el sistema de forma estandarizada.
5.  **Capa de Presentación:** Interfaz conversacional desarrollada en **Streamlit** con soporte para trazabilidad de fuentes y control de roles.

---

## 🛠 Stack Tecnológico

| Componente | Tecnología | Justificación |
| :--- | :--- | :--- |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) | Alto rendimiento y soporte nativo para asincronía. |
| **Orquestación de Agentes** | [LangGraph](https://langchain-ai.github.io/langgraph/) | Control granular de ciclos y estados en flujos RAG complejos. |
| **Base de Datos Vectorial** | [Qdrant](https://qdrant.tech/) | Búsqueda híbrida (densa + dispersa) y alta eficiencia. |
| **Extracción Documental** | [Docling](https://github.com/docling-project/docling) | Preservación de estructura jerárquica de documentos complejos. |
| **Embeddings** | [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Multilingüe y soporte multivectorial (densos, dispersos, ColBERT). |
| **Inferencia de LLM** | [llama.cpp](https://github.com/ggml-org/llama.cpp) | Inferencia local optimizada (GGUF) sobre GPU NVIDIA. |
| **Protocolo de Herramientas** | [FastMCP](https://github.com/modelcontextprotocol/python-sdk) | Estandarización de la integración de herramientas externas. |
| **Taxonomía Semántica** | [BERTopic](https://maartengr.github.io/BERTopic/) | Descubrimiento de tópicos no supervisado para etiquetado automático. |

---

## 📁 Estructura del Repositorio

```text
pymerag/
├── app/
│   ├── api/            # Endpoints de FastAPI (Ingesta, Consulta, Admin)
│   ├── core/           # Configuración, Seguridad y Logging
│   ├── ingest/         # Pipeline de extracción y chunking
│   ├── rag/            # Grafo de razonamiento, Retriever y LLM
│   ├── mcp_server/     # Servidor de herramientas MCP
│   ├── topics/         # Motor de descubrimiento de taxonomía (BERTopic)
│   └── models/         # Modelos de datos (SQLModel)
├── frontend/           # Interfaz de usuario en Streamlit
├── data/               # Corpus de pruebas y Golden Set de evaluación
├── eval/               # Framework de evaluación (RAGAS)
├── tests/              # Suite de pruebas unitarias e integración
├── .github/workflows/  # Pipelines de CI/CD (Linting y Tests)
├── docker-compose.yml  # Orquestación de servicios mediante Docker
└── pyproject.toml      # Gestión de dependencias (uv/pip)
```

---

## 🚀 Instalación y Despliegue

El sistema está diseñado para ser desplegado de forma aislada utilizando **Docker**.

### Requisitos previos
*   [Docker](https://www.docker.com/) y [Docker Compose](https://docs.docker.com/compose/)
*   GPU NVIDIA con soporte CUDA (recomendado RTX 3090 para inferencia local)

### Pasos para el despliegue
1.  Clona el repositorio:
    ```bash
    git clone https://github.com/lilidl-nft/pymerag-thesis.git
    cd pymerag-thesis
    ```
2.  Configura las variables de entorno:
    ```bash
    cp .env.example .env
    # Edita .env con tus credenciales y configuraciones
    ```
3.  Levanta el stack completo:
    ```bash
    docker-compose up -d
    ```

---

## 🧪 Evaluación y Calidad

El proyecto incluye un pipeline de evaluación automática basado en el framework **RAGAS**, midiendo métricas críticas como:
*   **Faithfulness:** Fidelidad de la respuesta a los documentos recuperados.
*   **Answer Relevancy:** Relevancia de la respuesta respecto a la consulta.
*   **Context Precision/Recall:** Calidad de la recuperación de fragmentos.

La integración continua (CI) mediante GitHub Actions garantiza que cada cambio cumpla con los estándares de calidad mediante la ejecución de `ruff` para linting y `pytest` para pruebas funcenciales.

---
*Este proyecto es parte de una investigación académica. Para consultas técnicas, contactar al autor.*
