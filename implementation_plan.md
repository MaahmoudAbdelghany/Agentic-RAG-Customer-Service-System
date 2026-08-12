# Agentic RAG Customer Service System — Implementation Plan

Build a production-ready, bilingual (Arabic + English) RAG-based AI customer service agent for e-commerce. The system retrieves accurate company knowledge, tracks orders, and escalates to humans — deployed as a generic, reusable platform with a synthetic demo knowledge base.

---

## Design Decisions Summary

| Decision | Choice |
|---|---|
| **Platform Type** | Generic, reusable with demo KB |
| **Demo Domain** | E-commerce (Shopify-style) |
| **Deployment** | Local-first (AWS scripts later) |
| **LLM** | Groq `llama-3.3-70b-versatile` |
| **Agent Framework** | LangGraph |
| **Vector DB** | ChromaDB |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (384d, local) |
| **Tools** | KB Retrieval, Order Tracking (mock), Human Handoff |
| **Interface** | FastAPI + Streamlit |
| **Memory** | LangGraph MemorySaver + SQLite |
| **Chunking** | Recursive (800 tokens, 100 overlap) |
| **Guardrails** | Off-topic, hallucination, PII |
| **Language** | Arabic + English |
| **Package Manager** | uv |

---

## Proposed Changes

### Phase 1: Project Scaffolding & Configuration ✅

#### [NEW] [pyproject.toml](file:///d:/AI Projects/Agentic RAG Customer Service System/pyproject.toml)
- uv-managed project with all dependencies:
  - `langgraph`, `langchain`, `langchain-groq`, `langchain-community`, `langchain-chroma`
  - `chromadb`, `sentence-transformers`
  - `fastapi`, `uvicorn`, `streamlit`
  - `python-dotenv`, `pydantic`, `pydantic-settings`

#### [NEW] [.env.example](file:///d:/AI Projects/Agentic RAG Customer Service System/.env.example)
- Template with: `GROQ_API_KEY`, `COMPANY_NAME`, `DEFAULT_LANGUAGE`, `CHROMA_PERSIST_DIR`, etc.

#### [NEW] [.gitignore](file:///d:/AI Projects/Agentic RAG Customer Service System/.gitignore)
- Standard Python + `.env` + `__pycache__` + `chroma_db/` + `.venv/`

#### [NEW] [README.md](file:///d:/AI Projects/Agentic RAG Customer Service System/README.md)
- Project overview, setup instructions, architecture diagram, usage guide.

---

### Phase 2: Configuration Module ✅

#### [NEW] [src/config/settings.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/config/settings.py)
- Pydantic `Settings` class loading from `.env`:
  - `GROQ_API_KEY`, `GROQ_MODEL` (default: `llama-3.3-70b-versatile`)
  - `EMBEDDING_MODEL` (default: `paraphrase-multilingual-MiniLM-L12-v2`)
  - `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION_NAME`
  - `CHUNK_SIZE` (800), `CHUNK_OVERLAP` (100)
  - `COMPANY_NAME`, `DEFAULT_LANGUAGE` (en/ar)
  - `TOP_K_RESULTS` (5), `SIMILARITY_THRESHOLD` (0.7)

#### [NEW] [src/config/__init__.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/config/__init__.py)

---

### Phase 3: Demo Knowledge Base (Synthetic Data)

#### [NEW] [data/knowledge_base/return_policy.md](file:///d:/AI Projects/Agentic RAG Customer Service System/data/knowledge_base/return_policy.md)
- Realistic return/refund policy for "TechStyle" e-commerce (English + Arabic sections)

#### [NEW] [data/knowledge_base/shipping_policy.md](file:///d:/AI Projects/Agentic RAG Customer Service System/data/knowledge_base/shipping_policy.md)
- Shipping methods, timelines, fees, international shipping, tracking info

#### [NEW] [data/knowledge_base/warranty_policy.md](file:///d:/AI Projects/Agentic RAG Customer Service System/data/knowledge_base/warranty_policy.md)
- Warranty terms, claim process, exclusions

#### [NEW] [data/knowledge_base/faq.md](file:///d:/AI Projects/Agentic RAG Customer Service System/data/knowledge_base/faq.md)
- 20-30 frequently asked questions with answers (bilingual)

#### [NEW] [data/knowledge_base/product_catalog.md](file:///d:/AI Projects/Agentic RAG Customer Service System/data/knowledge_base/product_catalog.md)
- Sample product descriptions with categories, prices, features

#### [NEW] [data/mock_orders.json](file:///d:/AI Projects/Agentic RAG Customer Service System/data/mock_orders.json)
- 15-20 mock orders with order IDs, customer emails, statuses (processing, shipped, delivered, returned), tracking numbers, items, timestamps

---

### Phase 4: RAG Pipeline (Document Processing + Vector Store)

#### [NEW] [src/rag/__init__.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/rag/__init__.py)

#### [NEW] [src/rag/document_loader.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/rag/document_loader.py)
- `load_documents(directory: str) -> list[Document]`: Loads all `.md`, `.txt`, `.pdf` files from a directory using LangChain document loaders
- Extracts metadata: `source_file`, `section`, `language` (detect Arabic vs English)
- Supports recursive directory scanning

#### [NEW] [src/rag/chunker.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/rag/chunker.py)
- `chunk_documents(documents: list[Document]) -> list[Document]`: Applies `RecursiveCharacterTextSplitter` with configurable `chunk_size=800`, `chunk_overlap=100`
- Preserves and enriches metadata per chunk (source, chunk_index, language)

#### [NEW] [src/rag/embeddings.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/rag/embeddings.py)
- `get_embedding_model() -> Embeddings`: Returns `HuggingFaceEmbeddings` with `paraphrase-multilingual-MiniLM-L12-v2`
- Singleton pattern to avoid re-loading the model on every call

#### [NEW] [src/rag/vector_store.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/rag/vector_store.py)
- `get_vector_store() -> Chroma`: Creates/loads a persistent ChromaDB collection
- `ingest_documents(documents: list[Document])`: Embeds and stores document chunks
- `similarity_search(query: str, k: int = 5) -> list[Document]`: Retrieves relevant chunks
- `reset_collection()`: Clears and re-ingests (for knowledge base updates)

#### [NEW] [src/rag/ingest.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/rag/ingest.py)
- CLI script: `python -m src.rag.ingest` — runs the full pipeline: load → chunk → embed → store
- Progress logging, chunk count reporting, duplicate detection

---

### Phase 5: Agent Tools

#### [NEW] [src/tools/__init__.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/tools/__init__.py)

#### [NEW] [src/tools/knowledge_retrieval.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/tools/knowledge_retrieval.py)
- `knowledge_retrieval_tool`: LangChain `@tool` decorated function
- Input: `query: str` (the customer's question)
- Searches ChromaDB, returns top-k relevant chunks with source metadata
- Formats results as structured context for the LLM

#### [NEW] [src/tools/order_tracking.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/tools/order_tracking.py)
- `order_status_tool`: LangChain `@tool` decorated function
- Input: `order_id: str` or `customer_email: str`
- Loads mock orders from `data/mock_orders.json`
- Returns: order status, tracking number, estimated delivery, items list
- Handles: order not found, invalid ID format

#### [NEW] [src/tools/human_handoff.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/tools/human_handoff.py)
- `human_handoff_tool`: LangChain `@tool` decorated function
- Input: `reason: str`, `conversation_summary: str`
- Logs the escalation event (to analytics DB)
- Returns a polite message to the customer indicating transfer to a human agent
- Triggers: anger/frustration detection, inability to answer, explicit user request

---

### Phase 6: LangGraph Agent

#### [NEW] [src/agent/__init__.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/agent/__init__.py)

#### [NEW] [src/agent/prompts.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/agent/prompts.py)
- `SYSTEM_PROMPT`: Bilingual system prompt template (parameterized with `{company_name}`)
  - Identity, role, and behavioral constraints
  - Tool usage instructions (when to use each tool)
  - Guardrails: no fabrication, no off-topic, PII warnings
  - Language handling: respond in the language the customer uses
- `GUARDRAIL_PROMPT`: Additional instructions for off-topic detection

#### [NEW] [src/agent/state.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/agent/state.py)
- `AgentState(TypedDict)`: Defines the LangGraph state schema
  - `messages: list[BaseMessage]` — conversation history
  - `language: str` — detected language (en/ar)
  - `escalated: bool` — whether handoff was triggered

#### [NEW] [src/agent/graph.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/agent/graph.py)
- Core LangGraph `StateGraph` definition:
  - **Nodes:**
    - `agent`: Calls the Groq LLM with tools bound, processes user message
    - `tools`: Executes tool calls (KB retrieval, order tracking, handoff)
  - **Edges:**
    - `agent → tools` (if tool call requested)
    - `tools → agent` (return tool results)
    - `agent → END` (if final response ready)
  - **Checkpointer:** `MemorySaver` (in-memory) with SQLite persistence option
- `build_agent() -> CompiledGraph`: Factory function that assembles and compiles the graph
- `run_agent(user_input: str, session_id: str) -> str`: High-level function to invoke the agent

---

### Phase 7: Guardrails Module

#### [NEW] [src/guardrails/__init__.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/guardrails/__init__.py)

#### [NEW] [src/guardrails/validators.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/guardrails/validators.py)
- `is_off_topic(query: str) -> bool`: Uses a lightweight check (keyword heuristics + LLM-based classification) to detect off-topic queries
- `detect_pii(text: str) -> list[str]`: Regex-based detection for emails, phone numbers, credit cards, SSNs in user input. Returns list of PII types found
- `validate_response(response: str, context: list[str]) -> bool`: Checks that the agent's response is grounded in the retrieved context (anti-hallucination)

---

### Phase 8: FastAPI Backend

#### [NEW] [src/api/__init__.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/api/__init__.py)

#### [NEW] [src/api/main.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/api/main.py)
- FastAPI app with CORS middleware
- `POST /api/chat` — Main chat endpoint
  - Request: `{ "message": str, "session_id": str }`
  - Response: `{ "response": str, "session_id": str, "tool_used": str | null, "language": str }`
- `GET /api/health` — Health check
- `POST /api/ingest` — Trigger knowledge base re-ingestion
- `GET /api/analytics/summary` — Returns conversation stats

#### [NEW] [src/api/models.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/api/models.py)
- Pydantic request/response models: `ChatRequest`, `ChatResponse`, `HealthResponse`, `AnalyticsSummary`

---

### Phase 9: Analytics & Logging

#### [NEW] [src/analytics/__init__.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/analytics/__init__.py)

#### [NEW] [src/analytics/logger.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/analytics/logger.py)
- `ConversationLogger` class using SQLite:
  - `log_message(session_id, role, content, tool_used, language, timestamp)`
  - `log_escalation(session_id, reason, timestamp)`
  - `log_unanswered(session_id, query, timestamp)`

#### [NEW] [src/analytics/metrics.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/analytics/metrics.py)
- `get_deflection_rate() -> float`: % of conversations resolved without human handoff
- `get_unanswered_queries() -> list[dict]`: Queries the bot couldn't answer
- `get_tool_usage_breakdown() -> dict`: Count of each tool invocation
- `get_conversation_count(period: str) -> int`: Total conversations in time period
- `get_daily_stats(days: int) -> list[dict]`: Daily conversation/escalation/unanswered counts

---

### Phase 10: Streamlit UI

#### [NEW] [src/ui/Home.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/ui/Home.py)
- Main chat interface:
  - Chat input with message history display
  - Session ID management (auto-generated or user-provided)
  - Language auto-detection indicator (🇬🇧/🇸🇦)
  - Tool usage indicators (icons showing when KB/order/handoff tools were used)
  - Streamlit `st.chat_message` components with proper avatars
  - Sidebar with company branding and quick-action buttons

#### [NEW] [src/ui/pages/Analytics.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/ui/pages/Analytics.py)
- Analytics dashboard page:
  - Deflection rate gauge/metric
  - Daily conversation volume chart
  - Tool usage pie chart
  - Unanswered queries table (sortable, filterable)
  - Recent escalations log

#### [NEW] [src/ui/pages/Knowledge_Base.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/ui/pages/Knowledge_Base.py)
- Knowledge base management:
  - View currently ingested documents
  - Upload new documents (drag & drop)
  - Trigger re-ingestion
  - View chunk count and collection stats

---

### Phase 11: Entry Points & Scripts

#### [NEW] [src/__init__.py](file:///d:/AI Projects/Agentic RAG Customer Service System/src/__init__.py)

#### [NEW] [scripts/run_api.py](file:///d:/AI Projects/Agentic RAG Customer Service System/scripts/run_api.py)
- Starts the FastAPI server via uvicorn

#### [NEW] [scripts/run_ui.py](file:///d:/AI Projects/Agentic RAG Customer Service System/scripts/run_ui.py)
- Starts the Streamlit UI

#### [NEW] [scripts/ingest_data.py](file:///d:/AI Projects/Agentic RAG Customer Service System/scripts/ingest_data.py)
- Runs the document ingestion pipeline from the CLI

---

## Verification Plan

### Automated Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test suites
uv run pytest tests/test_rag.py -v         # Document loading, chunking, embedding
uv run pytest tests/test_tools.py -v       # Tool execution (KB, order, handoff)
uv run pytest tests/test_agent.py -v       # Agent graph execution
uv run pytest tests/test_guardrails.py -v  # Off-topic, PII, hallucination checks
uv run pytest tests/test_api.py -v         # API endpoint tests
```

### Manual Verification

1. **Ingestion Pipeline**: Run `uv run python scripts/ingest_data.py` and verify chunk count, collection stats
2. **Chat Flow**: Open Streamlit UI, test multi-turn conversations in both English and Arabic
3. **Tool Routing**: Verify the agent correctly routes to KB retrieval for policy questions, order tracking for order queries, and handoff for angry/unanswerable queries
4. **Guardrails**: Test off-topic queries ("Who won the World Cup?"), PII input (credit card numbers), and hallucination prompts ("What is your policy on alien returns?")
5. **Analytics**: Verify conversation logs appear in the analytics dashboard after testing
6. **Bilingual**: Send messages in Arabic and verify the bot responds in Arabic using the Arabic KB content

### Test Scenarios

| Scenario | Expected Tool | Expected Behavior |
|---|---|---|
| "What is your return policy?" | `knowledge_retrieval` | Returns policy from KB |
| "Where is my order ORD-001?" | `order_status` | Returns mock order details |
| "I want to speak to a manager!" | `human_handoff` | Escalates with apology |
| "ما هي سياسة الإرجاع؟" | `knowledge_retrieval` | Returns Arabic policy |
| "What's the weather today?" | None (guardrail) | Politely refuses, off-topic |
| "My card number is 4111..." | None (PII guard) | Warns about sharing PII |
