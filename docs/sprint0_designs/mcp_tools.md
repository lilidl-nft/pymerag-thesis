# Pymerag MCP Tool Definitions

The Pymerag MCP server exposes the following tools to LLM clients (like Claude Desktop or custom agents).

## 1. `pymerag_search`
- **Description**: Performs a hybrid semantic and keyword search across the indexed document collection.
- **Arguments**:
  - `query` (string, required): The search query.
  - `top_k` (integer, optional, default: 5): Number of chunks to return.
  - `metadata_filter` (object, optional): Filter results by document metadata.
- **Returns**: A list of text chunks with their relevance scores and source metadata.

## 2. `pymerag_summarize`
- **Description**: Generates a summary of a specific document or a collection of related chunks.
- **Arguments**:
  - `document_id` (string, optional): Specific document to summarize.
  - `context_chunks` (array of strings, optional): Specific chunks to summarize.
  - `style` (string, optional): 'concise', 'detailed', 'bullet_points'.
- **Returns**: A summary string.

## 3. `pymerag_compare`
- **Description**: Compares information across two or more documents to find similarities or contradictions.
- **Arguments**:
  - `document_ids` (array of strings, required): List of document IDs.
  - `comparison_criteria` (string, required): What to compare (e.g., "pricing", "technical specs").
- **Returns**: A comparative analysis report.

## 4. `pymerag_generate_with_evidence`
- **Description**: Generates a high-fidelity answer to a query, ensuring every claim is backed by a source citation.
- **Arguments**:
  - `query` (string, required): The question to answer.
  - `strict_mode` (boolean, default: true): If true, the LLM must refuse to answer if no direct evidence is found.
- **Returns**: An answer string containing in-line citations (e.g., "[doc_1, chunk_42]").
