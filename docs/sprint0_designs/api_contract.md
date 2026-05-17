# Pymerag Core API Contract (OpenAPI 3.1)

## Base URL: `/api/v1`

### 1. Ingestion Module
- **POST `/ingest/document`**
  - **Description**: Trigger ingestion for a single document or a directory.
  - **Request Body**:
    ```json
    {
      "source_type": "file | directory | url",
      "source_path": "string",
      "metadata": { "key": "value" }
    }
    ```
  - **Response (202 Accepted)**:
    ```json
    { "job_id": "uuid", "status": "queued" }
    ```

- **GET `/ingest/status/{job_id}`**
  - **Description**: Check progress of an ingestion job.
  - **Response (200 OK)**:
    ```json
    { "job_id": "uuid", "status": "processing | completed | failed", "progress": 0.75 }
    ```

### 2. RAG / Query Module
- **POST `/query`**
  - **Description**: Execute a RAG-based query.
  - **Request Body**:
    ```json
    {
      "query": "string",
      "stream": true,
      "top_k": 5,
      "filters": { "metadata": { "key": "value" } }
    }
    ```
  - **Response (200 OK)**:
    ```json
    {
      "answer": "string",
      "sources": [
        { "chunk_id": "uuid", "content": "string", "metadata": {}, "score": 0.98 }
      ],
      "metadata": { "latency": 1.2 }
    }
    ```

### 3. Topic Module
- **GET `/topics`**
  - **Description**: Retrieve the current topic landscape.
  - **Response (200 OK)**:
    ```json
    {
      "topics": [
        { "id": "uuid", "name": "string", "description": "string", "count": 10 }
      ]
    }
    ```

### 4. Admin & Audit
- **GET `/admin/health`**
  - **Response (200 OK)**: `{ "status": "ok", "services": { "qdrant": "up", "llm": "up" } }`

- **GET `/admin/audit`**
  - **Description**: Retrieve system audit logs.
  - **Response (200 OK)**:
    ```json
    [
      { "id": "uuid", "timestamp": "ISO8601", "action": "ingest", "target": "doc_123" }
    ]
    ```
