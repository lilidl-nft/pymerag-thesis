# Pymerag Database Schema (SQLModel)

The system uses a relational database (PostgreSQL/SQLite) for metadata and structured information, while Qdrant handles the vector embeddings.

## 1. `Document` Model
Stores high-level information about ingested files.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `id` | UUID | Primary Key | Unique identifier for the document. |
| `name` | String | Not Null | Filename or display name. |
| `path` | String | Not Null | Original source path/URL. |
| `file_type` | String | | e.g., 'pdf', 'docx', 'txt'. |
| `metadata` | JSONB | | Custom metadata from Docling/User. |
| `created_at` | DateTime | Default: Now | Ingestion timestamp. |
| `updated_at` | DateTime | | Last modification timestamp. |

## 2. `Chunk` Model
Stores the mapping between text segments and the vector database.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `id` | UUID | Primary Key | Unique identifier for the chunk. |
| `document_id`| UUID | Foreign Key (Document) | Link to the parent document. |
| `content` | Text | Not Null | The actual text content of the chunk. |
| `qdrant_id` | String | Unique | The ID used in the Qdrant vector store. |
| `start_index` | Integer | | Character offset start. |
| `end_index` | Integer | | Character offset end. |
| `embedding_model`| String | | Which model produced this embedding. |

## 3. `Topic` Model
Stores results from the BERTopic pipeline.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `id` | UUID | Primary Key | Unique identifier. |
| `name` | String | Not Null | Topic label/name. |
| `description` | Text | | Automated or manual description. |
| `representative_chunks` | JSONB | | List of chunk IDs representing the topic. |

## 4. `AuditLog` Model
Tracks all critical system actions.

| Field | Type | Constraints | Description |
|---|---|---|---|
| `id` | UUID | Primary Key | Unique identifier. |
| `timestamp` | DateTime | Not Null | When the action occurred. |
| `user_id` | String | | User or system process ID. |
| `action` | String | Not Null | e.g., 'INGEST_START', 'QUERY_EXECUTE', 'MCP_TOOL_CALL'. |
| `target_id` | UUID | Optional | The ID of the resource affected. |
| `payload` | JSONB | | Snapshot of the request/response. |
