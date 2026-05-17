# Pymerag Squad Backlog (Post-Sprint 0)

This backlog is organized by agent assignment. Each agent should pick up their designated tasks and implement them following the Sprint 0 designs.

## 🚀 Agent A2: Ingestion & Indexing Specialist
**Goal**: Build a robust pipeline to convert documents to searchable chunks and vectors.

- [ ] **Task A2.1**: Implement `Docling` integration for PDF/Docx parsing.
- [ ] **Task A2.2**: Implement chunking strategy (semantic or fixed-size with overlap).
- [ ] **Task A2.3**: Implement BGE-M3 embedding generation logic.
- [ ] **Task A2.4**: Implement Qdrant client for hybrid search indexing (Dense + Sparse).
- [ ] **Task A2.5**: Create the `IngestionService` in `app/ingest/`.

## 🧠 Agent A3: RAG & LangGraph Architect
**Goal**: Build the intelligence layer that retrieves and synthesizes information.

- [ ] **Task A3.1**: Implement the Retrieval tool (connect to Qdrant via A2's service).
- [ ] **Task A3.2**: Design and implement the LangGraph state machine for the RAG loop.
- [ ] **Task A3.3**: Implement "Self-Correction" nodes (Re-ranking / Query expansion).
- [ ] **Task A3.4**: Integrate `llama.cpp` or local LLM for response generation.
- [ ] **Task A3.5**: Create the `RAGService` in `app/rag/`.

## 🔌 Agent A4: MCP Orchestration Engineer
**Goal**: Expose Pymerag capabilities to the outside world via MCP.

- [ ] **Task A4.1**: Set up `FastMCP` server in `app/mcp_server/`.
- [ ] **Task A4.2**: Implement `pymerag_search` MCP tool.
- [ ] **Task A4.3**: Implement `pymerag_summarize` MCP tool.
- [ ] **Task A4.4**: Implement `pymerag_compare` and `pymerag_generate_with_evidence` tools.
- [ ] **Task A4.5**: Ensure MCP tools call the correct services (A2, A3, A5).

## 📊 Agent A5: Topic Modeling & Analysis Expert
**Goal**: Provide high-level insights and document organization.

- [ ] **Task A5.1**: Implement `BERTopic` pipeline for asynchronous topic extraction.
- [ ] **Task A5.2**: Implement the `TopicService` in `app/topics/`.
- [ ] **Task A5.3**: Create the API endpoints for topic retrieval.
- [ ] **Task A5.4**: Develop logic to link chunks/documents to discovered topics.

## 🎨 Agent A6: Frontend & Integration Lead
**Goal**: Provide a user-facing interface and ensure end-to-end connectivity.

- [ ] **Task A6.1**: Set up the Streamlit application in `frontend/`.
- [ ] **Task A6.2**: Implement the Chat UI for RAG queries.
- [ ] **Task A6.3**: Implement the Document Management/Ingestion dashboard.
- [ ] **Task A6.4**: Implement the Topic Visualization dashboard.
- [ ] **Task A6.5**: Integrate API calls to `app/api/v1`.

---
**Note to Agents**: Refer to `docs/sprint0_designs/` for all interface and schema definitions.
